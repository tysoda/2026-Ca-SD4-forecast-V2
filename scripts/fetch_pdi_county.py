#!/usr/bin/env python3
"""
fetch_pdi_county.py
===================
Fetches PDI early-vote county-level data for the 9 SD4 counties that lie
wholly (or substantially) within SD4, using the unauthenticated JSON API:

  GET https://politicaldata.com/early-vote-counts
      ?districtId=<id>&_data=routes/early-vote-counts._index

The other 4 SD4 counties (Madera, Merced, Nevada, Placer) span multiple
senate districts; PDI only exposes whole-county totals for those IDs, so
they are excluded here — use ballot_returns.csv for their SD4-slice data.

Output: pdi_county_daily.csv
  county, election, code, days_before_eday, date, total, D, R, O

Run daily — PDI's 30-day window shifts forward as the election approaches,
exposing historical data for past elections one day at a time.
"""

import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

# 9 wholly-in-SD4 counties with confirmed PDI districtIds
COUNTY_IDS = {
    "Alpine":     10566,
    "Amador":     10603,
    "Calaveras":  10960,
    "El Dorado":  12473,
    "Inyo":       13087,
    "Mariposa":   13661,
    "Mono":       13866,
    "Stanislaus": 15743,
    "Tuolumne":   16048,
}

BASE_URL = (
    "https://politicaldata.com/early-vote-counts"
    "?districtId={id}&_data=routes/early-vote-counts._index"
)
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

OUT = Path(__file__).parent.parent / "pdi_county_daily.csv"
FIELDS = ["county", "election", "code", "days_before_eday", "date", "total", "D", "R", "O"]


def fetch_county(district_id: int) -> list[dict]:
    url = BASE_URL.format(id=district_id)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    if not data.get("success"):
        raise RuntimeError(f"API returned success=false for districtId={district_id}")
    rows = []
    for e in data.get("ellectionData", []):
        for c in e["cumulativeCounts"]:
            rows.append({
                "election": e["electionName"],
                "code":     e["electionCode"],
                "days_before_eday": c["DayUntilElection"],
                "date":     c["ElectionDate"],
                "total":    c["Total"] or 0,
                "D":        c["Dems"] or 0,
                "R":        c["Reps"] or 0,
                "O":        c["Others"] or 0,
            })
    return rows


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(open(path)))


def merge_rows(existing: list[dict], fresh_by_county: dict[str, list[dict]]) -> list[dict]:
    """Merge fresh rows into existing, preferring non-zero values from fresh."""
    # Key: (county, code, days_before_eday)
    merged = {
        (r["county"], r["code"], str(r["days_before_eday"])): r
        for r in existing
    }
    for county, rows in fresh_by_county.items():
        for r in rows:
            key = (county, r["code"], str(r["days_before_eday"]))
            existing_row = merged.get(key)
            if existing_row is None or int(r["total"]) > 0:
                merged[key] = {"county": county, **r}
    # Sort: county, code, days descending
    return sorted(
        merged.values(),
        key=lambda r: (r["county"], r["code"], -int(r["days_before_eday"])),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    existing = load_existing(OUT)
    fresh_by_county: dict[str, list[dict]] = {}
    errors = []

    for county, did in COUNTY_IDS.items():
        print(f"  Fetching {county} (districtId={did}) ...", end=" ", flush=True)
        try:
            rows = fetch_county(did)
            non_zero = sum(1 for r in rows if int(r["total"]) > 0)
            print(f"{len(rows)} rows, {non_zero} non-zero")
            fresh_by_county[county] = rows
        except Exception as e:
            print(f"ERROR: {e}")
            errors.append((county, e))
        time.sleep(0.2)  # be polite

    merged = merge_rows(existing, fresh_by_county)
    write_csv(OUT, merged)

    non_zero_total = sum(1 for r in merged if int(r["total"]) > 0)
    elections = sorted({r["code"] for r in merged if int(r["total"]) > 0})
    counties_fetched = sorted(fresh_by_county.keys())
    print(f"\nWrote {len(merged)} rows ({non_zero_total} non-zero) to {OUT.name}")
    print(f"Counties: {', '.join(counties_fetched)}")
    print(f"Elections with data: {', '.join(elections)}")

    if errors:
        print(f"\nErrors ({len(errors)}):", file=sys.stderr)
        for county, e in errors:
            print(f"  {county}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    print("Fetching PDI county-level VBM data ...")
    main()
