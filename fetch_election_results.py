"""
fetch_election_results.py
Fetches live SD4 State Senate and Governor election night results from
the California SOS API and writes them to:
  - election_night.csv   (SD4 State Senate results by county)
  - governor_night.csv   (Governor results by county)

Usage:
  python3 fetch_election_results.py           # fetch and save
  python3 fetch_election_results.py --dry-run # print JSON, don't save

API endpoints used:
  https://api.sos.ca.gov/returns/state-senate/district/4/county/{county}
  https://api.sos.ca.gov/returns/governor/county/{county}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent
API_BASE = "https://api.sos.ca.gov/returns"
TIMEOUT  = 15   # seconds per request
PAUSE    = 0.3  # seconds between requests (be polite to the API)

SD4_COUNTIES = [
    "alpine", "amador", "calaveras", "el-dorado", "inyo",
    "madera", "mariposa", "merced", "mono", "nevada",
    "placer", "stanislaus", "tuolumne",
]

# Map API county slug → display name used in election_night.csv
COUNTY_DISPLAY = {
    "alpine":      "Alpine",
    "amador":      "Amador",
    "calaveras":   "Calaveras",
    "el-dorado":   "El Dorado",
    "inyo":        "Inyo",
    "madera":      "Madera",
    "mariposa":    "Mariposa",
    "merced":      "Merced",
    "mono":        "Mono",
    "nevada":      "Nevada",
    "placer":      "Placer",
    "stanislaus":  "Stanislaus",
    "tuolumne":    "Tuolumne",
}

# Total precincts per county (for the template / sanity check)
PRECINCTS_TOTAL = {
    "Alpine": 6, "Amador": 68, "Calaveras": 81, "El Dorado": 155,
    "Inyo": 42, "Madera": 113, "Mariposa": 34, "Merced": 128,
    "Mono": 27, "Nevada": 70, "Placer": 233, "Stanislaus": 230,
    "Tuolumne": 69,
}

# Party name patterns → D / R / O
def _party_label(name: str) -> str:
    n = name.lower()
    if "dem" in n:
        return "D"
    if "rep" in n or "gop" in n:
        return "R"
    return "O"


# ── API fetching ──────────────────────────────────────────────────────────────

def fetch_county(contest_path: str, county_slug: str) -> dict | None:
    """Fetch one county result from the SOS API. Returns raw JSON dict or None."""
    url = f"{API_BASE}/{contest_path}/county/{county_slug}"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        # API returns a list with one element, or a dict
        if isinstance(data, list):
            return data[0] if data else None
        return data
    except Exception as e:
        print(f"  ⚠  {county_slug}: {e}", file=sys.stderr)
        return None


def parse_reporting(reporting_str: str) -> tuple[int, int]:
    """Parse '42% (42 of 100) precincts reporting' → (42, 100)."""
    try:
        inner = reporting_str.split("(")[1].split(")")[0]   # '42 of 100'
        parts = inner.split(" of ")
        return int(parts[0].replace(",", "")), int(parts[1].replace(",", ""))
    except Exception:
        return 0, 0


def parse_votes(candidates: list, party: str) -> int:
    """Sum votes for candidates matching party label."""
    total = 0
    for c in candidates:
        if _party_label(c.get("Party", "")+c.get("Name", "")) == party:
            v = c.get("Votes", "0")
            try:
                total += int(str(v).replace(",", ""))
            except ValueError:
                pass
    return total


def fetch_race(contest_path: str, counties: list[str], label: str) -> pd.DataFrame:
    """Fetch all counties for one race; return a DataFrame."""
    rows = []
    now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\nFetching {label}…")
    for slug in counties:
        display = COUNTY_DISPLAY.get(slug, slug.title())
        data = fetch_county(contest_path, slug)
        if data is None:
            print(f"  {display}: no data")
            rows.append({
                "county": display,
                "precincts_reporting": 0,
                "precincts_total": PRECINCTS_TOTAL.get(display, 0),
                "dem_votes": None, "rep_votes": None, "other_votes": None,
                "mail_dem": None, "mail_rep": None, "mail_other": None,
                "source": "api.sos.ca.gov",
                "last_updated": now_ts,
            })
            time.sleep(PAUSE)
            continue

        reporting_str = data.get("Reporting", "")
        prec_rep, prec_tot = parse_reporting(reporting_str)
        candidates = data.get("candidates", [])

        dem   = parse_votes(candidates, "D")
        rep   = parse_votes(candidates, "R")
        other = parse_votes(candidates, "O")

        pct_rep   = float(data.get("ReportingPct", "0").rstrip("%") or 0)
        report_time = data.get("ReportingTime", now_ts)

        print(f"  {display}: {prec_rep}/{prec_tot} precincts · D {dem:,} R {rep:,}")

        rows.append({
            "county":              display,
            "precincts_reporting": prec_rep,
            "precincts_total":     prec_tot or PRECINCTS_TOTAL.get(display, 0),
            "dem_votes":           dem   if dem + rep + other > 0 else None,
            "rep_votes":           rep   if dem + rep + other > 0 else None,
            "other_votes":         other if dem + rep + other > 0 else None,
            "mail_dem":            None,
            "mail_rep":            None,
            "mail_other":          None,
            "source":              "api.sos.ca.gov",
            "last_updated":        report_time or now_ts,
        })
        time.sleep(PAUSE)

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch CA SOS election night results for SD4.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print fetched data; don't write CSV files.")
    args = parser.parse_args()

    # SD4 State Senate
    df_sd4 = fetch_race("state-senate/district/4", SD4_COUNTIES, "SD4 State Senate")

    # Governor
    df_gov = fetch_race("governor", SD4_COUNTIES, "Governor")

    if args.dry_run:
        print("\n── SD4 State Senate ──")
        print(df_sd4.to_string(index=False))
        print("\n── Governor ──")
        print(df_gov.to_string(index=False))
        return

    sd4_path = DATA_DIR / "election_night.csv"
    gov_path = DATA_DIR / "governor_night.csv"

    df_sd4.to_csv(sd4_path, index=False)
    df_gov.to_csv(gov_path, index=False)

    print(f"\n✅ Saved {sd4_path}")
    print(f"✅ Saved {gov_path}")


if __name__ == "__main__":
    main()
