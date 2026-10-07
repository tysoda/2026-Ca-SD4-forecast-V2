#!/usr/bin/env python3
"""
fetch_optiq_crosstabs.py
Daily snapshot of SD4 VBM crosstab data from Optiq Data API.

Appends a new row for each (party, age, gender, ethnicity) cell with today's
snapshot_date, issued count, and returned count. Idempotent — safe to re-run;
existing rows for today are replaced (delete-then-reinsert) to handle mid-day
re-runs.

Output: optiq_crosstabs.csv
"""

import json
import os
import ssl
import sys
import urllib.request
from datetime import date

import pandas as pd

DISTRICT_SLUG = "STATE%20SENATE%204"
API_URL = f"https://abev.optiqdata.com/api/ballot-returns/{DISTRICT_SLUG}"
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "optiq_crosstabs.csv")
SNAPSHOT_DATE = date.today().isoformat()  # e.g. "2026-10-07"

# SD4 party registration as of 60-day general election report (September 4, 2026).
SD4_PARTY_REG = {
    "D": 207213,
    "R": 250162,
    "O": 178612,  # All non-D/R: NPP (121,703) + AIP (34,830) + Lib (8,975) + Green (2,802) + P&F (3,456) + Other (4,439) + Unknown (2,407)
}


def fetch_rows(url: str) -> list[dict]:
    ca_bundle = "/root/.ccr/ca-bundle.crt"
    ctx = ssl.create_default_context(cafile=ca_bundle if os.path.exists(ca_bundle) else None)
    try:
        with urllib.request.urlopen(url, timeout=15, context=ctx) as resp:
            data = json.loads(resp.read())
            return data.get("ballotReturns", [])
    except Exception as exc:
        print(f"ERROR fetching Optiq API: {exc}", file=sys.stderr)
        sys.exit(1)


def process_rows(raw_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_rows)
    df["voters"] = pd.to_numeric(df["voters"], errors="coerce").fillna(0).astype(int)
    df["returned"] = df["returnDate"].notna().astype(int)

    # Aggregate by (party, age, gender, ethnicity): sum issued and returned
    grouped = (
        df.groupby(["party", "age", "gender", "ethnicity"])
        .apply(lambda g: pd.Series({
            "issued":   g["voters"].sum(),
            "returned": g.loc[g["returnDate"].notna(), "voters"].sum(),
        }))
        .reset_index()
    )
    grouped["snapshot_date"] = SNAPSHOT_DATE
    grouped["return_rate"] = (
        grouped["returned"] / grouped["issued"].replace(0, pd.NA)
    ).round(6)

    return grouped[["snapshot_date", "party", "age", "gender", "ethnicity",
                     "issued", "returned", "return_rate"]]


def load_existing(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path, dtype=str)
    return pd.DataFrame(columns=["snapshot_date", "party", "age", "gender",
                                  "ethnicity", "issued", "returned", "return_rate"])


def main():
    print(f"Fetching Optiq crosstabs for {DISTRICT_SLUG} (snapshot_date={SNAPSHOT_DATE})…")
    raw = fetch_rows(API_URL)
    print(f"  {len(raw):,} rows received from API")

    today_df = process_rows(raw)
    print(f"  {len(today_df)} distinct (party × age × gender × ethnicity) cells")

    existing = load_existing(OUTPUT_CSV)

    # Drop any existing rows for today (idempotent re-run)
    if len(existing) > 0 and "snapshot_date" in existing.columns:
        existing = existing[existing["snapshot_date"] != SNAPSHOT_DATE]

    combined = pd.concat([existing, today_df], ignore_index=True)

    # Coerce numeric columns back to correct types before saving
    for col in ("issued", "returned"):
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0).astype(int)
    combined["return_rate"] = pd.to_numeric(combined["return_rate"], errors="coerce").round(6)

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"  Written {len(combined)} rows to {OUTPUT_CSV}")

    # Quick summary
    party_labels = {"D": "Democrat", "R": "Republican", "O": "Other/NPP"}
    print("\nParty summary for today:")
    for party in ["D", "R", "O"]:
        pslice = today_df[today_df["party"] == party]
        issued = int(pslice["issued"].sum())
        returned = int(pslice["returned"].sum())
        rate = returned / issued if issued > 0 else 0
        reg = SD4_PARTY_REG.get(party, 0)
        reg_share = issued / sum(SD4_PARTY_REG.values()) if sum(SD4_PARTY_REG.values()) > 0 else 0
        print(f"  {party_labels.get(party, party):15s}: {returned:>7,} returned / {issued:>7,} issued "
              f"({rate:.1%} rate) | reg share: {issued/sum(today_df['issued'].groupby(today_df['party']).sum()):.1%} of issued")


if __name__ == "__main__":
    main()
