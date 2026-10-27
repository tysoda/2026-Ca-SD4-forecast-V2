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
OPTIQ_BASE = "https://abev.civiq.vote/api"


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


def fetch_sos_live(days_back: int = 3) -> pd.DataFrame:
    """Newest SOS report straight from the SOS site (dated files for the last few
    days plus the undated one), in the ballot_returns.csv shape. Lets the app show a
    new report as soon as SOS posts it, without waiting for the GitHub Action.
    Empty frame if nothing can be fetched.
    """
    import datetime
    import sys
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    try:
        import importlib
        import fetch_ballot_returns as F
        # Streamlit keeps imported modules between reruns; reload so a newer version on disk
        # (after a git pull) is used rather than the copy loaded before the update
        F = importlib.reload(F)
        fractions = F.load_partial_fractions()
    except Exception:
        return pd.DataFrame()
    today = pd.Timestamp.now(tz="America/Los_Angeles").date()
    best = None
    for i in range(days_back + 1):
        d = today - datetime.timedelta(days=i)
        snap = rows = None
        for url in F.dated_urls(d):        # two-digit and four-digit year file names
            try:
                snap, rows = F.parse_xlsx(F.download_xlsx(url), fractions, fallback_date=d)
                break
            except Exception:
                continue
        if rows is None:
            continue
        if rows and (best is None or snap > best[0]):
            best = (snap, rows)
        if best is not None:
            break                      # newest dated file found
    if best is None:
        return pd.DataFrame()
    df = pd.DataFrame(best[1])
    df["snapshot_date"] = pd.Timestamp(best[0])
    return df


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


def fetch_optiq_live() -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Live row-level Optiq pull, aggregated to cells with a return date.

    Returns (frame, latest_update). Frame columns: party, age, gender, ethnicity,
    return_date (NaT if not returned), voters; empty if the API is unreachable.
    latest_update is Optiq's last data-load time (Pacific), or None.
    """
    try:
        payload = _get_json(f"{OPTIQ_BASE}/ballot-returns/{OPTIQ_SLUG}")
    except Exception:
        return pd.DataFrame(), None
    rows = payload.get("ballotReturns", [])
    latest = _to_pacific(payload.get("latestUpdate"))
    if not rows:
        return pd.DataFrame(), latest
    df = pd.DataFrame(rows)
    df["voters"] = pd.to_numeric(df["voters"], errors="coerce").fillna(0).astype(int)
    df["return_date"] = pd.to_datetime(df["returnDate"], errors="coerce").dt.tz_localize(None).dt.normalize()
    for c in ("party", "age", "gender", "ethnicity"):
        df[c] = df[c].astype(str)
    return df[["party", "age", "gender", "ethnicity", "return_date", "voters"]], latest


def _to_pacific(ts) -> pd.Timestamp | None:
    if not ts:
        return None
    try:
        t = pd.Timestamp(ts)
    except Exception:
        return None
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return t.tz_convert("America/Los_Angeles").tz_localize(None)


def load_optiq_drops() -> pd.DataFrame:
    """optiq_county_drops.csv: one row per change in a county's returned count, with the
    Optiq load time (Pacific) that first showed it. First rows per county have no time."""
    path = REPO_DIR / "optiq_county_drops.csv"
    if not path.exists():
        return pd.DataFrame(columns=["county", "returned", "loaded_at"])
    df = pd.read_csv(path, dtype={"county": str})
    for c in ("returned", "D", "R", "O"):
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    df["loaded_at"] = df["loaded_at"].map(_to_pacific)
    return df


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


# ── 2022 benchmarks ──────────────────────────────────────────────────────────
def load_pdi_2022() -> pd.DataFrame | None:
    """2022 SD4 daily VBM returns by party, read from PDI's tracker chart
    (average of three readings — estimates, roughly ±300 per party per day).
    Columns: days_before_eday, D, R, O, total, date_2022.
    """
    path = REPO_DIR / "pdi_2022_daily_sd4.csv"
    if not path.exists():
        return None
    return pd.read_csv(path).sort_values("days_before_eday", ascending=False).reset_index(drop=True)


def load_registration_2022() -> pd.DataFrame | None:
    """SD4 party registration estimate from the SOS 15-day report (Oct 24, 2022).
    Split counties: county total × SD4 fraction, with the 2026 SD4-portion party mix.
    """
    path = REPO_DIR / "sd4_party_registration_2022.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ── Optiq precincts (from Civiq's map tiles, via fetch_optiq_precincts.py) ───
def load_optiq_precincts() -> pd.DataFrame:
    """Precinct × snapshot D/R/O registered and returned (SD4 precincts only)."""
    path = REPO_DIR / "optiq_precincts.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"precinct": str}, parse_dates=["snapshot_date"])
    for c in df.columns:
        if c.endswith(("_registered", "_returned")):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


def load_precinct_coverage() -> pd.DataFrame:
    """Per-county precinct sums vs Optiq county totals, one block per snapshot."""
    path = REPO_DIR / "optiq_precincts_coverage.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["snapshot_date"])


def load_precinct_shapes() -> dict | None:
    """sd4_precincts.geojson — feature id is 'County|precinct'."""
    path = REPO_DIR / "sd4_precincts.geojson"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_sd4_boundary() -> list | None:
    path = REPO_DIR / "sd4_boundary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["coordinates"][0]


def load_last_saved() -> dict:
    """When each GitHub Action last committed new data (UTC → Pacific), from
    last_saved_sos.txt / last_saved_optiq.txt. Missing until the Action next saves."""
    out = {}
    for key in ("sos", "optiq"):
        path = REPO_DIR / f"last_saved_{key}.txt"
        try:
            ts = pd.Timestamp(path.read_text().strip())
            out[key] = ts.tz_convert("America/Los_Angeles").tz_localize(None)
        except Exception:
            out[key] = None
    return out


# ── PDI overrides for counties whose Optiq feed has stalled ──────────────────
# PDI district ids (from scripts/fetch_pdi_county.py). Returned ballots for these
# counties come from PDI's 2026 General early-vote counts instead of Optiq.
PDI_OVERRIDE_IDS = {"Alpine": 10566}
PDI_URL = "https://politicaldata.com/early-vote-counts?districtId={id}&_data=routes/early-vote-counts._index"


def _pdi_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("days_before_eday", "total", "D", "R", "O"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df = df[df["total"] > 0]           # zeros are days PDI hasn't released, not zero ballots
    df["date"] = EDAY_2026 - pd.to_timedelta(df["days_before_eday"], unit="D")
    return df.sort_values("days_before_eday", ascending=False).reset_index(drop=True)


def fetch_pdi_overrides() -> dict:
    """{county: cumulative 2026 General returns by day (date, total, D, R, O)} for the
    override counties — live from PDI, falling back to pdi_county_daily.csv."""
    out = {}
    saved = None
    for county, did in PDI_OVERRIDE_IDS.items():
        df = pd.DataFrame()
        try:
            req = urllib.request.Request(PDI_URL.format(id=did),
                                         headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=20, context=_ssl_ctx()) as r:
                payload = json.loads(r.read())
            rows = [{"days_before_eday": c["DayUntilElection"], "total": c["Total"] or 0,
                     "D": c["Dems"] or 0, "R": c["Reps"] or 0, "O": c["Others"] or 0}
                    for e in payload.get("ellectionData", []) if e.get("electionCode") == "26G"
                    for c in e.get("cumulativeCounts", [])]
            df = _pdi_frame(rows)
            df.attrs["source"] = "live"
        except Exception:
            pass
        if df.empty:
            if saved is None:
                path = REPO_DIR / "pdi_county_daily.csv"
                saved = pd.read_csv(path) if path.exists() else pd.DataFrame()
            if not saved.empty:
                df = _pdi_frame(saved[(saved["county"] == county) & (saved["code"] == "26G")]
                                .to_dict("records"))
                df.attrs["source"] = "saved"
        if not df.empty:
            out[county] = df
    return out


def apply_pdi_overrides(cp: pd.DataFrame, pdi: dict) -> pd.DataFrame:
    """Replace override counties' returned (by party) in Optiq county × party rows with PDI's
    cumulative count as of each row's snapshot date. Registered stays Optiq's."""
    if cp.empty or not pdi:
        return cp
    cp = cp.copy()
    if "source" not in cp:
        cp["source"] = "Optiq"
    for county, s in pdi.items():
        for d in cp.loc[cp["county"] == county, "snapshot_date"].unique():
            upto = s[s["date"] <= pd.Timestamp(d)]
            if upto.empty:
                continue
            last = upto.iloc[-1]
            rose_on = upto[upto["total"] == last["total"]]["date"].min()   # day the count last rose
            for p in ("D", "R", "O"):
                m = (cp["county"] == county) & (cp["snapshot_date"] == d) & (cp["party"] == p)
                cp.loc[m, "returned"] = int(last[p])
                if "latest_return_date" in cp:
                    cp.loc[m, "latest_return_date"] = rose_on
                cp.loc[m, "source"] = "PDI"
    return cp
