#!/usr/bin/env python3
"""
fetch_pdi_daily.py
==================
Fetches the PDI early vote tracker page, extracts the window.__remixContext
JSON, and overwrites pdi_statewide_daily.csv with the latest cumulative
daily VBM return totals for all available elections (2022G, 2024G, etc.).

Run daily — as the election approaches, PDI's 30-day window shifts forward
and exposes more historical data for past elections, which we use to
calibrate per-county window fractions.
"""

import csv
import json
import re
import sys
import urllib.request
from pathlib import Path

URL = "https://politicaldata.com/early-vote-counts"
OUT = Path(__file__).parent.parent / "pdi_statewide_daily.csv"
FIELDS = ["election", "code", "days_before_eday", "date", "total", "D", "R", "O"]


def fetch_remix_context(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8")

    match = re.search(r"window\.__remixContext\s*=\s*(.*)", html)
    if not match:
        raise RuntimeError("window.__remixContext not found in page")

    raw = match.group(1)
    # PDI server-renders escaped JSON into the page
    unescaped = raw.encode().decode("unicode_escape")

    # Find the balanced JSON object
    depth, end = 0, 0
    for i, c in enumerate(unescaped):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    return json.loads(unescaped[:end])


def extract_rows(data: dict) -> list[dict]:
    election_data = (
        data["state"]["loaderData"]["routes/early-vote-counts._index"]["ellectionData"]
    )
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
    print(f"Fetching {URL} ...")
    try:
        data = fetch_remix_context(URL)
    except Exception as e:
        print(f"ERROR fetching page: {e}", file=sys.stderr)
        sys.exit(1)

    fresh = extract_rows(data)
    existing = load_existing(OUT)
    merged = merge_rows(existing, fresh)
    write_csv(OUT, merged)

    non_zero = sum(1 for r in merged if int(r["total"]) > 0)
    elections = sorted({r["code"] for r in merged if int(r["total"]) > 0})
    print(f"Wrote {len(merged)} rows ({non_zero} non-zero) to {OUT.name}")
    print(f"Elections with data: {', '.join(elections)}")


if __name__ == "__main__":
    main()
