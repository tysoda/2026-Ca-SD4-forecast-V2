#!/usr/bin/env python3
"""
fetch_ballot_returns.py
=======================
Downloads (or reads a local copy of) the California SOS VBM ballot return
statistics XLSX and appends the SD4 county figures to ballot_returns.csv.

Usage
-----
  # Download from SOS and append:
  python3 fetch_ballot_returns.py

  # Use a locally-downloaded file instead (e.g. when proxy blocks the CDN):
  python3 fetch_ballot_returns.py --file /path/to/bsr-statistics.xlsx

  # Dry-run — print what would be appended without writing:
  python3 fetch_ballot_returns.py --dry-run

The script is idempotent: if a row for the snapshot date already exists in
ballot_returns.csv it skips rather than duplicating.
"""

import argparse
import csv
import io
import os
import pathlib
import urllib.request
import ssl
import datetime
from typing import Optional

import openpyxl

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_DIR   = pathlib.Path(__file__).parent
DATA_DIR   = REPO_DIR                          # CSVs live at repo root
OUTPUT_CSV = DATA_DIR / "ballot_returns.csv"
PARTIAL_CSV = DATA_DIR / "partial_county_fractions.csv"

SOS_URL = (
    "https://elections.cdn.sos.ca.gov/"
    "statewide-elections/2026-general/bsr-statistics.xlsx"
)

# ── SD4 counties (all 13) ──────────────────────────────────────────────────────
SD4_COUNTIES = {
    "Alpine", "Amador", "Calaveras", "El Dorado", "Inyo",
    "Madera", "Mariposa", "Merced", "Mono", "Nevada",
    "Placer", "Stanislaus", "Tuolumne",
}

# ── Column indices (0-based) in the data rows ─────────────────────────────────
# Row 4 (index 3) is the header row:
# col 0  COUNTY
# col 1  County Type
# col 2  Total Voters Issued VBM Ballots
# col 3  Drop Box
# col 4  Drop Off Location
# col 5  Vote Center Drop Off
# col 6  Mail
# col 7  FAX
# col 8  Other
# col 9  Total Returned VBM Ballots
# col 10 Total Accepted VBM Ballots
# col 11 Accepted % of Voter-Returned Ballots
COL_COUNTY   = 0
COL_ISSUED   = 2
COL_RETURNED = 9
COL_ACCEPTED = 10


def load_partial_fractions() -> dict:
    fractions = {}
    with open(PARTIAL_CSV, newline="") as f:
        for row in csv.DictReader(f):
            fractions[row["county"]] = float(row["sd4_fraction"])
    return fractions


def download_xlsx() -> bytes:
    """Download the SOS XLSX; raises on failure."""
    # Use system CA bundle if available (CCR proxy)
    ca_bundle = "/root/.ccr/ca-bundle.crt"
    if os.path.exists(ca_bundle):
        ctx = ssl.create_default_context(cafile=ca_bundle)
    else:
        ctx = ssl.create_default_context()

    proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy_url:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"https": proxy_url, "http": proxy_url}),
            urllib.request.HTTPSHandler(context=ctx),
        )
    else:
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ctx)
        )

    with opener.open(SOS_URL, timeout=30) as resp:
        return resp.read()


def parse_xlsx(data: bytes, fractions: dict) -> tuple[datetime.date, list[dict]]:
    """
    Parse the SOS XLSX bytes.

    Returns
    -------
    snapshot_date : datetime.date  (from cell A1)
    rows          : list of dicts, one per SD4 county, with keys:
                    county, vbm_issued, vbm_returned, vbm_accepted
                    Values for split counties are already scaled to the SD4 portion.
    """
    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
    ws = wb["VBM Ballot Statistics"]

    # Cell A1 contains the snapshot datetime
    raw_date = ws.cell(1, 1).value
    if isinstance(raw_date, datetime.datetime):
        snapshot_date = raw_date.date()
    elif isinstance(raw_date, datetime.date):
        snapshot_date = raw_date
    else:
        # Fallback: today
        snapshot_date = datetime.date.today()

    rows_out = []
    for row in ws.iter_rows(min_row=5, values_only=True):   # data starts row 5
        county = row[COL_COUNTY]
        if not county or county not in SD4_COUNTIES:
            continue

        fraction = fractions.get(county, 1.0)

        def scale(val) -> int:
            if val is None:
                return 0
            return round(float(val) * fraction)

        rows_out.append({
            "county":       county,
            "vbm_issued":   scale(row[COL_ISSUED]),
            "vbm_returned": scale(row[COL_RETURNED]),
            "vbm_accepted": scale(row[COL_ACCEPTED]),
        })

    return snapshot_date, rows_out


def load_existing_keys() -> set:
    """Return set of (county, snapshot_date) already in ballot_returns.csv."""
    if not OUTPUT_CSV.exists():
        return set()
    with open(OUTPUT_CSV, newline="") as f:
        return {(r["county"], r["snapshot_date"]) for r in csv.DictReader(f)}


def append_rows(
    snapshot_date: datetime.date,
    rows: list[dict],
    dry_run: bool = False,
) -> int:
    """Append new rows; return count written (0 if all already present)."""
    existing = load_existing_keys()
    date_str = snapshot_date.isoformat()

    to_write = [
        r for r in rows
        if (r["county"], date_str) not in existing
    ]

    if not to_write:
        print(f"  ✓ All rows for {date_str} already present — nothing to append.")
        return 0

    fieldnames = ["county", "snapshot_date", "vbm_issued", "vbm_returned", "vbm_accepted"]
    write_header = not OUTPUT_CSV.exists()

    if dry_run:
        print(f"  DRY RUN — would append {len(to_write)} rows for {date_str}:")
        for r in sorted(to_write, key=lambda x: x["county"]):
            print(
                f"    {r['county']:12s}  issued={r['vbm_issued']:>7,}  "
                f"returned={r['vbm_returned']:>6,}  accepted={r['vbm_accepted']:>6,}"
            )
        return len(to_write)

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in sorted(to_write, key=lambda x: x["county"]):
            writer.writerow({
                "county":        r["county"],
                "snapshot_date": date_str,
                "vbm_issued":    r["vbm_issued"],
                "vbm_returned":  r["vbm_returned"],
                "vbm_accepted":  r["vbm_accepted"],
            })

    print(f"  ✓ Appended {len(to_write)} rows for {date_str} → {OUTPUT_CSV}")
    return len(to_write)


def main():
    parser = argparse.ArgumentParser(description="Fetch SOS VBM ballot return stats for SD4.")
    parser.add_argument("--file", metavar="PATH", help="Use a local XLSX instead of downloading.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be written without writing.")
    args = parser.parse_args()

    fractions = load_partial_fractions()

    if args.file:
        print(f"Reading local file: {args.file}")
        with open(args.file, "rb") as f:
            data = f.read()
    else:
        print(f"Downloading from SOS…")
        try:
            data = download_xlsx()
            print("  ✓ Download complete")
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            print("  Tip: download manually and re-run with --file /path/to/bsr-statistics.xlsx")
            raise SystemExit(1)

    snapshot_date, rows = parse_xlsx(data, fractions)
    print(f"  Snapshot date: {snapshot_date}  |  SD4 counties found: {len(rows)}")

    append_rows(snapshot_date, rows, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
