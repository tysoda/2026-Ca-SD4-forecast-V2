#!/usr/bin/env python3
"""
fetch_optiq_precincts.py
========================
Precinct-level SD4 ballot returns from Civiq/Optiq's map tiles.

The precinct map at abev.civiq.vote is served as Mapbox vector tiles:
  https://abev.civiq.vote/api/tiles/STATE%20SENATE%204/{z}/{x}/{y}.pbf
Each tile has a "precincts" layer with, per precinct: precinct_shapefile,
county_name, {dem,rep,other,total}_{registered,returned}, margin_pct.
Tiles aren't limited to SD4 — they hold every precinct in their area.

This script:
  1. fetches the tiles covering SD4 at one zoom level,
  2. decodes them and merges precincts that straddle tile edges,
  3. keeps SD4 precincts: all of the 9 whole counties, and in the 4 split
     counties those whose centre falls inside sd4_boundary.json,
  4. checks county totals against Optiq's county feed (geography-breakdown)
     and, if they don't match, retries at a higher zoom,
  5. appends today's snapshot to optiq_precincts.csv and writes the shapes
     to sd4_precincts.geojson (only when the precinct set changes).

Usage
  python3 fetch_optiq_precincts.py            # fetch, validate, write
  python3 fetch_optiq_precincts.py --dry-run  # fetch and validate only
  python3 fetch_optiq_precincts.py --tiles-dir DIR --zoom 6
        # use already-downloaded tiles named {z}-{x}-{y}.pbf (for testing)
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import ssl
import sys
import time
import urllib.request
from pathlib import Path

import mapbox_vector_tile as mvt
import pandas as pd

REPO = Path(__file__).resolve().parent
BASE = "https://abev.civiq.vote/api"
SLUG = "STATE%20SENATE%204"
TILE_URL = f"{BASE}/tiles/{SLUG}/{{z}}/{{x}}/{{y}}.pbf"
GEO_URL = f"{BASE}/geography-breakdown/{SLUG}?level=county_name"
OUT_CSV = REPO / "optiq_precincts.csv"
OUT_GEOJSON = REPO / "sd4_precincts.geojson"
BOUNDARY = REPO / "sd4_boundary.json"

SD4_COUNTIES = {"ALPINE", "AMADOR", "CALAVERAS", "EL DORADO", "INYO", "MADERA", "MARIPOSA",
                "MERCED", "MONO", "NEVADA", "PLACER", "STANISLAUS", "TUOLUMNE"}
SPLIT_COUNTIES = {"MADERA", "MERCED", "NEVADA", "PLACER"}
FIELDS = ["dem_registered", "rep_registered", "other_registered", "total_registered",
          "dem_returned", "rep_returned", "other_returned", "total_returned"]
ZOOMS = (6, 7, 8)            # tried in order until county totals match Optiq
MATCH_TOLERANCE = 0.02       # whole counties: |precinct sum − Optiq| / Optiq


# ── Tiles ────────────────────────────────────────────────────────────────────
def tile_xy(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    lr = math.radians(lat)
    y = int((1 - math.asinh(math.tan(lr)) / math.pi) / 2 * n)
    return x, y


def tiles_for_bbox(bbox: tuple[float, float, float, float], z: int) -> list[tuple[int, int, int]]:
    lon0, lat0, lon1, lat1 = bbox
    x0, y0 = tile_xy(lon0, lat1, z)   # top-left
    x1, y1 = tile_xy(lon1, lat0, z)   # bottom-right
    return [(z, x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def to_lonlat(px: float, py: float, z: int, x: int, y: int, extent: int) -> tuple[float, float]:
    n = 2 ** z
    lon = (x + px / extent) / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + py / extent) / n))))
    return round(lon, 6), round(lat, 6)


def _ctx():
    ca = "/root/.ccr/ca-bundle.crt"
    return ssl.create_default_context(cafile=ca if os.path.exists(ca) else None)


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (SD4 ballot tracker)"})
    with urllib.request.urlopen(req, timeout=30, context=_ctx()) as r:
        return r.read()


def load_tile(z, x, y, tiles_dir: Path | None) -> bytes | None:
    if tiles_dir:
        p = tiles_dir / f"{z}-{x}-{y}.pbf"
        return p.read_bytes() if p.exists() else None
    try:
        return _get(TILE_URL.format(z=z, x=x, y=y))
    except Exception as e:
        if getattr(e, "code", None) in (204, 404):
            return None
        raise


# ── Decode and merge ─────────────────────────────────────────────────────────
def decode_tiles(tiles, tiles_dir: Path | None) -> dict:
    """{(county, precinct): {"props": {...}, "polygons": [[ring, ...], ...]}}"""
    out: dict = {}
    for z, x, y in tiles:
        data = load_tile(z, x, y, tiles_dir)
        if not data:
            continue
        t = mvt.decode(data, default_options={"y_coord_down": True})
        layer = t.get("precincts")
        if not layer:
            continue
        ext = layer.get("extent", 4096)
        for f in layer["features"]:
            p = f["properties"]
            county = str(p.get("county_name") or "").upper().strip()
            pid = str(p.get("precinct_shapefile") or "").strip()
            if not county or not pid:
                continue
            g = f["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            polys = [[[to_lonlat(px, py, z, x, y, ext) for px, py in ring] for ring in poly] for poly in polys]
            rec = out.setdefault((county, pid), {"props": p, "polygons": []})
            rec["polygons"].extend(polys)   # pieces clipped at tile edges are kept together
        if not tiles_dir:
            time.sleep(0.3)                 # be gentle with the server
    return out


def point_in_polygon(lon: float, lat: float, ring: list) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]; xj, yj = ring[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def centroid(polygons: list) -> tuple[float, float]:
    """Area-weighted centre of the outer rings (falls back to mean of points)."""
    ax = ay = a_tot = 0.0
    for poly in polygons:
        ring = poly[0]
        a = cx = cy = 0.0
        for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
            cr = x0 * y1 - x1 * y0
            a += cr; cx += (x0 + x1) * cr; cy += (y0 + y1) * cr
        if abs(a) > 1e-12:
            ax += cx / 3; ay += cy / 3; a_tot += a
    if abs(a_tot) > 1e-12:
        return ax / a_tot, ay / a_tot
    pts = [pt for poly in polygons for pt in poly[0]]
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def select_sd4(precincts: dict, boundary_ring: list) -> pd.DataFrame:
    rows = []
    for (county, pid), rec in precincts.items():
        if county not in SD4_COUNTIES:
            continue
        lon, lat = centroid(rec["polygons"])
        if county in SPLIT_COUNTIES and not point_in_polygon(lon, lat, boundary_ring):
            continue
        p = rec["props"]
        rows.append({"county": county.title(), "precinct": pid, "lon": round(lon, 5), "lat": round(lat, 5),
                     **{k: int(p.get(k) or 0) for k in FIELDS}})
    return pd.DataFrame(rows, columns=["county", "precinct", "lon", "lat"] + FIELDS)


# ── Validation against Optiq county totals ───────────────────────────────────
def optiq_county_totals(geo_rows: list | None = None) -> dict:
    rows = geo_rows if geo_rows is not None else json.loads(_get(GEO_URL))
    return {str(r["geography"]).title(): {"registered": int(r.get("totalRegistered") or 0),
                                          "returned": int(r.get("totalReturned") or 0)} for r in rows}


def validate(df: pd.DataFrame, optiq: dict) -> tuple[bool, pd.DataFrame]:
    g = df.groupby("county")[["total_registered", "total_returned"]].sum()
    rows, ok = [], True
    for c, o in sorted(optiq.items()):
        reg = int(g.loc[c, "total_registered"]) if c in g.index else 0
        ret = int(g.loc[c, "total_returned"]) if c in g.index else 0
        diff = (reg - o["registered"]) / o["registered"] if o["registered"] else 0.0
        split = c.upper() in SPLIT_COUNTIES
        if not split and abs(diff) > MATCH_TOLERANCE:
            ok = False
        rows.append({"county": c, "precincts": int((df["county"] == c).sum()), "reg_precincts": reg,
                     "reg_optiq": o["registered"], "reg_diff": round(diff, 4),
                     "ret_precincts": ret, "ret_optiq": o["returned"], "split": split})
    return ok, pd.DataFrame(rows)


# ── Output ───────────────────────────────────────────────────────────────────
def write_outputs(df: pd.DataFrame, precincts: dict, snapshot: str) -> None:
    snap = df.drop(columns=["lon", "lat"]).assign(snapshot_date=snapshot)
    snap = snap[["snapshot_date", "county", "precinct"] + FIELDS]
    if OUT_CSV.exists():
        old = pd.read_csv(OUT_CSV, dtype={"precinct": str})
        snap = pd.concat([old[old["snapshot_date"] != snapshot], snap], ignore_index=True)
    snap.to_csv(OUT_CSV, index=False)
    print(f"  ✓ {len(df)} precincts for {snapshot} → {OUT_CSV.name} ({len(snap)} rows total)")

    keys = set(zip(df["county"].str.upper(), df["precinct"]))
    feats = []
    for (county, pid), rec in sorted(precincts.items()):
        if (county, pid) not in keys:
            continue
        r = df[(df["county"].str.upper() == county) & (df["precinct"] == pid)].iloc[0]
        coords = [[[list(pt) for pt in ring] for ring in poly] for poly in rec["polygons"]]
        feats.append({"type": "Feature", "id": f"{county.title()}|{pid}",
                      "properties": {"county": county.title(), "precinct": pid,
                                     "lon": float(r["lon"]), "lat": float(r["lat"])},
                      "geometry": {"type": "MultiPolygon", "coordinates": coords}})
    gj = {"type": "FeatureCollection", "features": feats}
    if OUT_GEOJSON.exists():
        old_ids = {f["id"] for f in json.loads(OUT_GEOJSON.read_text())["features"]}
        if old_ids == {f["id"] for f in feats}:
            print(f"  · {OUT_GEOJSON.name} unchanged (same precinct set)")
            return
    OUT_GEOJSON.write_text(json.dumps(gj, separators=(",", ":")))
    print(f"  ✓ {len(feats)} precinct shapes → {OUT_GEOJSON.name}")


def main():
    ap = argparse.ArgumentParser(description="SD4 precinct ballot returns from Civiq map tiles.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--zoom", type=int, help="Use only this zoom level.")
    ap.add_argument("--tiles-dir", type=Path, help="Read tiles from DIR/{z}-{x}-{y}.pbf instead of fetching.")
    ap.add_argument("--geo-json", type=Path, help="Read the Optiq county breakdown from a file (testing).")
    args = ap.parse_args()

    ring = [tuple(p) for p in json.loads(BOUNDARY.read_text())["coordinates"][0]]
    lons, lats = [p[0] for p in ring], [p[1] for p in ring]
    bbox = (min(lons) - 0.05, min(lats) - 0.05, max(lons) + 0.05, max(lats) + 0.05)
    optiq = optiq_county_totals(json.loads(args.geo_json.read_text()) if args.geo_json else None)
    snapshot = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-8))).date().isoformat()

    for z in ([args.zoom] if args.zoom else ZOOMS):
        tiles = tiles_for_bbox(bbox, z)
        print(f"Zoom {z}: {len(tiles)} tiles")
        precincts = decode_tiles(tiles, args.tiles_dir)
        df = select_sd4(precincts, ring)
        ok, report = validate(df, optiq)
        print(report.to_string(index=False))
        if ok and len(df):
            if args.dry_run:
                print("Dry run — nothing written.")
            else:
                write_outputs(df, precincts, snapshot)
            return
        print(f"  ✗ county totals don't match Optiq at zoom {z}" + (" — trying higher zoom" if not args.zoom else ""))
    print("ERROR: precinct totals never matched Optiq's county totals; nothing written.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
