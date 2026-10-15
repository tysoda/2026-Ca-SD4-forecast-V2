"""
data.py — loaders for the SD4 ballot return tracker.

Everything here is plain pandas (no Streamlit), so it can be imported from
notebooks or tests. The app wraps these in st.cache_data.

Sources (all at repo root, refreshed by GitHub Actions):
  ballot_returns.csv        SOS BSR county VBM stats (SD4 portion for split counties)
  optiq_crosstabs.csv       Optiq party × age × gender × ethnicity snapshots
  optiq_county_party.csv    Optiq county × party snapshots (written by fetch_optiq_crosstabs.py)
  vbm_2022_baseline.csv     2022 SOS BSR baseline, SD4 portion
  sd4_party_registration.csv  SOS 60-day report party registration
  vbm_2022_daily.csv        OPTIONAL — real 2022 return curve; replaces the linear pace baseline
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent

SD4_COUNTIES = [
    "Alpine", "Amador", "Calaveras", "El Dorado", "Inyo", "Madera", "Mariposa",
    "Merced", "Mono", "Nevada", "Placer", "Stanislaus", "Tuolumne",
]
PARTIAL_COUNTIES = {"Madera", "Merced", "Nevada", "Placer"}

PARTIES = ["D", "R", "O"]
PARTY_LABELS = {"D": "Democrat", "R": "Republican", "O": "Other / NPP"}
PARTY_COLORS = {"D": "#1565c0", "R": "#b71c1c", "O": "#666666"}

# Optiq dimension codes → labels (age codes as used in the forecast app)
DIM_LABELS = {
    "party":     PARTY_LABELS,
    "age":       {"1": "18–34", "3": "35–49", "4": "50–64", "6": "65+"},
    "gender":    {"F": "Female", "M": "Male", "U": "Unknown"},
    "ethnicity": {"A": "Asian", "B": "Black", "H": "Hispanic", "W": "White"},
}
DIM_NAMES = {"party": "Party", "age": "Age", "gender": "Gender", "ethnicity": "Ethnicity"}

EDAY_2026 = pd.Timestamp("2026-11-03")
EDAY_2022 = pd.Timestamp("2022-11-08")
MAILOUT_2022 = pd.Timestamp("2022-10-10")

OPTIQ_SLUG = "STATE%20SENATE%204"
OPTIQ_BASE = "https://abev.optiqdata.com/api"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _drop_stale_snapshots(df: pd.DataFrame, date_col: str, value_cols: list[str],
                          group_cols: list[str] | None = None) -> pd.DataFrame:
    """Drop snapshots whose figures are identical to the previous snapshot.

    Both feeds are polled on a schedule, so weekends and non-updating days
    produce repeated rows stamped with a new date. Keeping them would show
    fake zero-velocity days. Comparison is on the whole snapshot (all groups).
    """
    if df.empty:
        return df
    keys = (group_cols or []) + value_cols
    sigs = (
        df.sort_values([date_col] + (group_cols or []))
          .groupby(date_col)[keys]
          .apply(lambda g: hash(tuple(map(tuple, g.astype(str).values))))
          .sort_index()
    )
    stale = sigs[sigs.eq(sigs.shift())].index
    return df[~df[date_col].isin(stale)].copy()


def _ssl_ctx() -> ssl.SSLContext:
    ca = "/root/.ccr/ca-bundle.crt"
    return ssl.create_default_context(cafile=ca if os.path.exists(ca) else None)


def _get_json(url: str, timeout: int = 10):
    with urllib.request.urlopen(url, timeout=timeout, context=_ssl_ctx()) as r:
        return json.loads(r.read())


# ── SOS ballot returns ───────────────────────────────────────────────────────
def load_sos_returns(drop_stale: bool = True) -> pd.DataFrame:
    """County × snapshot VBM issued/returned/accepted (SD4 portion)."""
    path = REPO_DIR / "ballot_returns.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["snapshot_date"])
    for c in ("vbm_issued", "vbm_returned", "vbm_accepted"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df = df[df["county"].isin(SD4_COUNTIES)]
    if drop_stale:
        df = _drop_stale_snapshots(df, "snapshot_date",
                                   ["vbm_issued", "vbm_returned"], ["county"])
    return df.sort_values(["snapshot_date", "county"]).reset_index(drop=True)


def load_2022_baseline() -> pd.DataFrame:
    path = REPO_DIR / "vbm_2022_baseline.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).set_index("county")


def load_2022_daily() -> pd.DataFrame | None:
    """Optional real 2022 return curve.

    Expected schema (one row per county per day):
      county, days_before_eday, vbm_returned
    SD4-portion counts. When present, the pace baseline uses it instead of
    linear interpolation.
    """
    path = REPO_DIR / "vbm_2022_daily.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    need = {"county", "days_before_eday", "vbm_returned"}
    return df if need.issubset(df.columns) else None


def load_party_registration() -> dict:
    path = REPO_DIR / "sd4_party_registration.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {"counts": dict(zip(df["party"], df["registered"].astype(int)))}
    out["report_date"] = str(df["report_date"].iloc[0]) if "report_date" in df else ""
    return out


# ── Optiq crosstabs ──────────────────────────────────────────────────────────
def load_optiq_crosstabs(drop_stale: bool = True) -> pd.DataFrame:
    path = REPO_DIR / "optiq_crosstabs.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"party": str, "age": str, "gender": str, "ethnicity": str})
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    for c in ("issued", "returned"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    if drop_stale:
        df = _drop_stale_snapshots(df, "snapshot_date", ["issued", "returned"],
                                   ["party", "age", "gender", "ethnicity"])
    return df.reset_index(drop=True)


def fetch_optiq_live() -> pd.DataFrame:
    """Live row-level Optiq pull, aggregated to cells with a return date.

    Returns columns: party, age, gender, ethnicity, return_date (NaT if not
    returned), voters. Empty frame if the API is unreachable.
    """
    try:
        rows = _get_json(f"{OPTIQ_BASE}/ballot-returns/{OPTIQ_SLUG}").get("ballotReturns", [])
    except Exception:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["voters"] = pd.to_numeric(df["voters"], errors="coerce").fillna(0).astype(int)
    df["return_date"] = pd.to_datetime(df["returnDate"], errors="coerce").dt.tz_localize(None).dt.normalize()
    for c in ("party", "age", "gender", "ethnicity"):
        df[c] = df[c].astype(str)
    return df[["party", "age", "gender", "ethnicity", "return_date", "voters"]]


def live_to_snapshot(live: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Collapse a live pull to the optiq_crosstabs.csv shape."""
    g = live.assign(ret=np.where(live["return_date"].notna(), live["voters"], 0))
    out = (g.groupby(["party", "age", "gender", "ethnicity"], as_index=False)
             .agg(issued=("voters", "sum"), returned=("ret", "sum")))
    out["snapshot_date"] = snapshot_date
    return out


# ── Optiq county × party ─────────────────────────────────────────────────────
def _parse_geo_rows(rows: list[dict]) -> pd.DataFrame:
    """geography-breakdown rows → one row per county × party (D/R/O).

    O uses Optiq's npp fields when present, else total − D − R.
    latest_return_date is Optiq's most recent ballot return date for the county.
    """
    recs = []
    for r in rows:
        cn = str(r.get("geography", "")).title()
        if cn not in SD4_COUNTIES:
            continue
        tot_reg = int(r.get("totalRegistered") or 0)
        tot_ret = int(r.get("totalReturned") or 0)
        d_reg, d_ret = int(r.get("demRegistered") or 0), int(r.get("demReturned") or 0)
        r_reg, r_ret = int(r.get("repRegistered") or 0), int(r.get("repReturned") or 0)
        o_reg = r.get("nppRegistered")
        o_ret = r.get("nppReturned")
        o_reg = int(o_reg) if o_reg is not None else max(tot_reg - d_reg - r_reg, 0)
        o_ret = int(o_ret) if o_ret is not None else max(tot_ret - d_ret - r_ret, 0)
        latest = r.get("latestReturnDate")
        for p, reg, ret in (("D", d_reg, d_ret), ("R", r_reg, r_ret), ("O", o_reg, o_ret)):
            recs.append({"county": cn, "party": p, "registered": reg, "returned": ret,
                         "latest_return_date": latest})
    df = pd.DataFrame(recs)
    if not df.empty:
        df["latest_return_date"] = pd.to_datetime(df["latest_return_date"], errors="coerce")
    return df


def load_optiq_county_party(drop_stale: bool = True) -> pd.DataFrame:
    """Persisted county × party snapshots (may not exist until the Action runs)."""
    path = REPO_DIR / "optiq_county_party.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["snapshot_date"])
    if "latest_return_date" in df:
        df["latest_return_date"] = pd.to_datetime(df["latest_return_date"], errors="coerce")
    for c in ("registered", "returned"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    if drop_stale:
        df = _drop_stale_snapshots(df, "snapshot_date", ["registered", "returned"],
                                   ["county", "party"])
    return df.reset_index(drop=True)


def fetch_optiq_county_party_live() -> pd.DataFrame:
    try:
        rows = _get_json(f"{OPTIQ_BASE}/geography-breakdown/{OPTIQ_SLUG}?level=county_name")
    except Exception:
        return pd.DataFrame()
    return _parse_geo_rows(rows)


def load_forecast_params() -> dict | None:
    """forecast_params.json from the forecast app (structural model output)."""
    path = REPO_DIR / "forecast_params.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
