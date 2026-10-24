#!/usr/bin/env python3
"""
fetch_pdi_daily.py
==================
Fetches SD4 district-level data from the PDI early vote tracker via the
unauthenticated JSON API and merges it into pdi_statewide_daily.csv.

  GET https://politicaldata.com/early-vote-counts
      ?districtId=10320&_data=routes/early-vote-counts._index

Run daily — as the election approaches, PDI's 30-day window shifts forward
and exposes more historical data for past elections, which we use to
calibrate per-county window fractions.
"""

import csv
import json
import sys
import urllib.request
from pathlib import Path

SD4_ID = 10320
URL = (
    f"https://politicaldata.com/early-vote-counts"
    f"?districtId={SD4_ID}&_data=routes/early-vote-counts._index"
)
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
OUT = Path(__file__).parent.parent / "pdi_statewide_daily.csv"
FIELDS = ["election", "code", "days_before_eday", "date", "total", "D", "R", "O"]


def fetch_election_data(url: str) -> list:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    if not data.get("success"):
        raise RuntimeError(f"API returned success=false for {url}")
    return data["ellectionData"]


def extract_rows(election_data: list) -> list[dict]:
    rows = []
    for e in election_data:
        for c in e["cumulativeCounts"]:
            rows.append(
                {
                    "election": e["electionName"],
                    "code": e["electionCode"],
                    "days_before_eday": c["DayUntilElection"],
                    "date": c["ElectionDate"],
                    "total": c["Total"] or 0,
                    "D": c["Dems"] or 0,
                    "R": c["Reps"] or 0,
                    "O": c["Others"] or 0,
                }
            )
    return rows


def merge_rows(existing: list[dict], fresh: list[dict]) -> list[dict]:
    """Merge fresh rows into existing, preferring non-zero values from fresh."""
    # Key: (code, days_before_eday)
    merged = {(r["code"], str(r["days_before_eday"])): r for r in existing}
    for r in fresh:
        key = (r["code"], str(r["days_before_eday"]))
        existing_row = merged.get(key)
        # Overwrite if fresh has data or no existing row
        if existing_row is None or int(r["total"]) > 0:
            merged[key] = r
    # Sort: by election code then days descending
    return sorted(merged.values(), key=lambda r: (r["code"], -int(r["days_before_eday"])))


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(open(path)))


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    print(f"Fetching SD4 (districtId={SD4_ID}) ...")
    try:
        election_data = fetch_election_data(URL)
    except Exception as e:
        print(f"ERROR fetching data: {e}", file=sys.stderr)
        sys.exit(1)

    fresh = extract_rows(election_data)
    existing = load_existing(OUT)
    merged = merge_rows(existing, fresh)
    write_csv(OUT, merged)

    non_zero = sum(1 for r in merged if int(r["total"] or 0) > 0)
    elections = sorted({r["code"] for r in merged if int(r["total"] or 0) > 0})
    print(f"Wrote {len(merged)} rows ({non_zero} non-zero) to {OUT.name}")
    print(f"Elections with data: {', '.join(elections)}")


if __name__ == "__main__":
    main()
