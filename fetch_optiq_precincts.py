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
     and, if they don't match, retries at a higher zoom; if no zoom matches
     every county, keeps the best one as long as it covers ≥95% of the
     whole counties' registration (some precincts may have no map shape),
  5. appends today's snapshot to optiq_precincts.csv, per-county coverage to
     optiq_precincts_coverage.csv, and writes the shapes to
     sd4_precincts.geojson (only when the precinct set changes).

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
OUT_COVERAGE = REPO / "optiq_precincts_coverage.csv"
BOUNDARY = REPO / "sd4_boundary.json"

SD4_COUNTIES = {"ALPINE", "AMADOR", "CALAVERAS", "EL DORADO", "INYO", "MADERA", "MARIPOSA",
                "MERCED", "MONO", "NEVADA", "PLACER", "STANISLAUS", "TUOLUMNE"}
SPLIT_COUNTIES = {"MADERA", "MERCED", "NEVADA", "PLACER"}
FIELDS = ["dem_registered", "rep_registered", "other_registered", "total_registered",
          "dem_returned", "rep_returned", "other_returned", "total_returned"]
ZOOMS = (6, 7, 8)            # tried in order until county totals match Optiq
MATCH_TOLERANCE = 0.02       # whole counties: |precinct sum − Optiq| / Optiq
MIN_COVERAGE = 0.95          # if no zoom matches every county, save the best one if it covers this much


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
def decode_tiles(tiles, tiles_dir: Path | None, stats: dict | None = None) -> dict:
    """{(county, precinct): {"props": {...}, "polygons": [[ring, ...], ...]}}

    A precinct cut by a tile edge appears in each tile with identical figures,
    so pieces are merged on (county, precinct id, figures). Different precincts
    that share a shapefile id (consolidated precincts) keep separate entries,
    labelled id, id#2, ….
    """
    stats = stats if stats is not None else {}
    stats.update(features=0, no_county=0, no_id=0)
    groups: dict = {}
    for z, x, y in tiles:
        data = load_tile(z, x, y, tiles_dir)
        if not data:
            continue
        t = mvt.decode(data, default_options={"y_coord_down": True})
        layer = t.get("precincts")
        if not layer:
            continue
        ext = layer.get("extent", 4096)
        seen_here: dict = {}
        for f in layer["features"]:
            stats["features"] += 1
            p = f["properties"]
            county = str(p.get("county_name") or "").upper().strip()
            pid = str(p.get("precinct_shapefile") or "").strip()
            if not county:
                stats["no_county"] += 1      # kept; county is inferred from neighbours in select_sd4
            if not pid:
                stats["no_id"] += 1
                pid = "(no id)"
            sig = tuple(int(p.get(k) or 0) for k in FIELDS)
            # The same key twice in one tile means two different precincts
            n = seen_here.get((county, pid, sig), 0)
            seen_here[(county, pid, sig)] = n + 1
            sig = sig + (n,)
            g = f["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            polys = [[[to_lonlat(px, py, z, x, y, ext) for px, py in ring] for ring in poly] for poly in polys]
            rec = groups.setdefault((county, pid), {}).setdefault(sig, {"props": p, "polygons": []})
            rec["polygons"].extend(polys)
        if not tiles_dir:
            time.sleep(0.3)                 # be gentle with the server
    out = {}
    shared = 0
    for (county, pid), by_sig in groups.items():
        if len(by_sig) > 1:
            shared += 1
        for i, sig in enumerate(sorted(by_sig, key=lambda s: (-s[3], s[-1]))):
            out[(county, pid if i == 0 else f"{pid}#{i + 1}")] = by_sig[sig]
    stats["shared_ids"] = shared
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


def select_sd4(precincts: dict, boundary_ring: list, stats: dict | None = None) -> pd.DataFrame:
    centres = {key: centroid(rec["polygons"]) for key, rec in precincts.items()}
    # Precincts with no county label take the county of the nearest labelled precinct
    labelled = [(c, lon, lat) for (c, _), (lon, lat) in centres.items() if c]
    assigned: dict = {}
    for (county, pid), (lon, lat) in list(centres.items()):
        if county or not labelled:
            continue
        k = math.cos(math.radians(lat))
        near = min(labelled, key=lambda r: ((r[1] - lon) * k) ** 2 + (r[2] - lat) ** 2)[0]
        assigned[near] = assigned.get(near, 0) + 1
        rec, c = precincts.pop((county, pid)), centres.pop((county, pid))
        new = (near, pid)
        if new in precincts and precincts[new]["props"] == rec["props"]:
            precincts[new]["polygons"].extend(rec["polygons"])   # same precinct, label missing in one tile
            continue
        while new in precincts:
            new = (near, new[1] + "+")
        precincts[new], centres[new] = rec, c
    if stats is not None:
        stats["assigned"] = assigned
    rows = []
    for (county, pid), rec in precincts.items():
        if county not in SD4_COUNTIES:
            continue
        lon, lat = centres[(county, pid)]
        if county in SPLIT_COUNTIES and not point_in_polygon(lon, lat, boundary_ring):
            continue
        p = rec["props"]
        rows.append({"county": county.title(), "precinct": pid, "lon": round(lon, 5), "lat": round(lat, 5),
                     **{k: int(p.get(k) or 0) for k in FIELDS}})
    return pd.DataFrame(rows, columns=["county", "precinct", "lon", "lat"] + FIELDS)


# ── Validation against Optiq county totals ───────────────────────────────────
def optiq_county_totals(geo_rows: list | None = None) -> dict:
    rows = geo_rows if geo_rows is not None else json.loads(_get(GEO_URL))
    out = {}
    for r in rows:
        o = {"registered": int(r.get("totalRegistered") or 0), "returned": int(r.get("totalReturned") or 0)}
        for party, key in (("dem", "dem"), ("rep", "rep"), ("other", "npp")):
            o[f"{party}_registered"] = int(r.get(f"{key}Registered") or 0)
            o[f"{party}_returned"] = int(r.get(f"{key}Returned") or 0)
        out[str(r["geography"]).title()] = o
    return out


def validate(df: pd.DataFrame, optiq: dict) -> tuple[bool, float, pd.DataFrame]:
    """Compare precinct sums with Optiq's county totals.

    Returns (all whole counties within tolerance, whole-county registered
    coverage, per-county report).
    """
    g = df.groupby("county")[FIELDS].sum()
    rows, ok = [], True
    whole_reg = whole_optiq = 0
    for c, o in sorted(optiq.items()):
        reg = int(g.loc[c, "total_registered"]) if c in g.index else 0
        ret = int(g.loc[c, "total_returned"]) if c in g.index else 0
        diff = (reg - o["registered"]) / o["registered"] if o["registered"] else 0.0
        split = c.upper() in SPLIT_COUNTIES
        if not split:
            whole_reg += reg; whole_optiq += o["registered"]
            if abs(diff) > MATCH_TOLERANCE:
                ok = False
        rows.append({"county": c, "precincts": int((df["county"] == c).sum()), "reg_precincts": reg,
                     "reg_optiq": o["registered"], "reg_diff": round(diff, 4),
                     "ret_precincts": ret, "ret_optiq": o["returned"], "split": split,
                     # Party figures for the part of the county not on the map (Optiq − precinct sum)
                     **{f"gap_{k}": (o.get(k, 0) - (int(g.loc[c, k]) if c in g.index else 0))
                        for k in FIELDS if not k.startswith("total")}})
    coverage = whole_reg / whole_optiq if whole_optiq else 0.0
    return ok, coverage, pd.DataFrame(rows)


# ── Output ───────────────────────────────────────────────────────────────────
SIMPLIFY_DEG = 0.0004   # ~40 m: plenty for a precinct map, and keeps the file ~2–3 MB


def _rdp(pts: list, eps: float) -> list:
    """Douglas–Peucker line simplification (open line)."""
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        (x0, y0), (x1, y1) = pts[a], pts[b]
        dx, dy = x1 - x0, y1 - y0
        n = math.hypot(dx, dy) or 1e-12
        dmax, im = 0.0, None
        for i in range(a + 1, b):
            d = abs(dy * (pts[i][0] - x0) - dx * (pts[i][1] - y0)) / n
            if d > dmax:
                dmax, im = d, i
        if im is not None and dmax > eps:
            keep[im] = True
            stack += [(a, im), (im, b)]
    return [p for p, k in zip(pts, keep) if k]


def _slim(ring: list) -> list:
    """Simplify a closed ring (split in two so the closed ends don't collapse it)."""
    if len(ring) >= 8:
        m = len(ring) // 2
        out = _rdp(ring[:m + 1], SIMPLIFY_DEG)[:-1] + _rdp(ring[m:], SIMPLIFY_DEG)
        ring = out if len(out) >= 4 else ring
    out = []
    for lon, lat in ring:
        pt = [round(lon, 4), round(lat, 4)]
        if not out or pt != out[-1]:
            out.append(pt)
    return out


def write_outputs(df: pd.DataFrame, precincts: dict, snapshot: str, report: pd.DataFrame) -> None:
    cov = report.assign(snapshot_date=snapshot)[
        ["snapshot_date", "county", "split", "precincts", "reg_precincts", "reg_optiq", "ret_precincts", "ret_optiq"]
        + [c for c in report.columns if c.startswith("gap_")]]
    if OUT_COVERAGE.exists():
        oldc = pd.read_csv(OUT_COVERAGE)
        cov = pd.concat([oldc[oldc["snapshot_date"] != snapshot], cov], ignore_index=True)
    cov.to_csv(OUT_COVERAGE, index=False)

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
        coords = [[_slim(ring) for ring in poly] for poly in rec["polygons"]]
        feats.append({"type": "Feature", "id": f"{county.title()}|{pid}",
                      "properties": {"county": county.title(), "precinct": pid,
                                     "lon": float(r["lon"]), "lat": float(r["lat"])},
                      "geometry": {"type": "MultiPolygon", "coordinates": coords}})
    gj = {"type": "FeatureCollection", "format": 2, "features": feats}
    if OUT_GEOJSON.exists():
        old = json.loads(OUT_GEOJSON.read_text())
        if old.get("format") == 2 and {f["id"] for f in old["features"]} == {f["id"] for f in feats}:
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

    best = None
    for z in ([args.zoom] if args.zoom else ZOOMS):
        tiles = tiles_for_bbox(bbox, z)
        stats: dict = {}
        precincts = decode_tiles(tiles, args.tiles_dir, stats)
        df = select_sd4(precincts, ring, stats)
        ok, coverage, report = validate(df, optiq)
        print(f"Zoom {z}: {len(tiles)} tiles, {stats['features']} features "
              f"({stats['no_county']} without a county, {stats['no_id']} without a precinct id, "
              f"{stats['shared_ids']} ids shared by more than one precinct)")
        if stats.get("assigned"):
            print("  Unlabelled precincts given their nearest neighbour's county: "
                  + ", ".join(f"{c.title()} {n}" for c, n in sorted(stats["assigned"].items())))
        print(report[[c for c in report.columns if not c.startswith("gap_")]].to_string(index=False))
        print(f"  Whole-county registration coverage: {coverage:.2%}")
        if best is None or coverage > best[0] + 1e-9 or (abs(coverage - best[0]) < 1e-9 and len(df) > len(best[2])):
            best = (coverage, z, df, precincts, report)
        if ok and len(df):
            break
        print(f"  ✗ some whole counties are more than {MATCH_TOLERANCE:.0%} off Optiq at zoom {z}")

    coverage, z, df, precincts, report = best
    off = report[~report["split"] & (report["reg_diff"].abs() > MATCH_TOLERANCE)]
    if coverage < MIN_COVERAGE or df.empty:
        print(f"ERROR: best coverage {coverage:.1%} (zoom {z}) is below {MIN_COVERAGE:.0%}; nothing written.",
              file=sys.stderr)
        sys.exit(1)
    if len(off):
        print(f"WARNING: saving zoom {z} ({coverage:.2%} coverage) — precinct totals short of Optiq in: "
              + ", ".join(f"{r.county} ({r.reg_diff:+.1%})" for r in off.itertuples()))
    if args.dry_run:
        print("Dry run — nothing written.")
    else:
        write_outputs(df, precincts, snapshot, report)

if __name__ == "__main__":
    main()
