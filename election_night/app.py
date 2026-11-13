"""
SD4 Election Night Tracker — standalone Streamlit app.

Run locally:     streamlit run election_night/app.py
Streamlit Cloud: same repo, main file path = election_night/app.py

Lifted from the forecast app's 🌙 Election Night tab. The pre-results prior is
rebuilt from forecast_params.json by prior.py (forecast-app defaults), so this
app doesn't import election_forecast.py. Live results come from the repo-root
election_night.csv / governor_night.csv written by fetch_election_results.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib
import os
import subprocess
import tempfile
import time as _time

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import streamlit as st

import prior as P
import remaining as REM

# Streamlit keeps imported helper modules between reruns; reload after a git pull.
P = importlib.reload(P)
REM = importlib.reload(REM)

DATA_DIR = P.REPO_DIR
WIN_THRESHOLD = P.WIN_THRESHOLD

st.set_page_config(page_title="SD4 Election Night", page_icon="🌙", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  h1, h2, h3, h4 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.02em; }
  .block-container { padding-top: 2rem; padding-bottom: 3rem; }
  .section-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase; color: #888;
    border-bottom: 1px solid #e0e0e0; padding-bottom: 0.4rem; margin-bottom: 1rem;
  }
  .stat-card { background: #f7f7f5; border-left: 3px solid #1a6b3c; padding: 0.85rem 1rem; border-radius: 2px; margin-bottom: 0.5rem; }
  .stat-card .label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #666; margin-bottom: 0.2rem; }
  .stat-card .value { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 600; color: #111; line-height: 1; }
  .stat-card .sub { font-size: 0.72rem; color: #888; margin-top: 0.2rem; }
  .win-hero { background: linear-gradient(135deg, #0f3d24 0%, #1a6b3c 100%); color: white; padding: 1.5rem 1.75rem; border-radius: 4px; text-align: center; }
  .win-hero .wlabel { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.14em; text-transform: uppercase; opacity: 0.7; margin-bottom: 0.5rem; }
  .win-hero .wvalue { font-family: 'IBM Plex Mono', monospace; font-size: 3.2rem; font-weight: 600; line-height: 1; }
  .win-hero .wsub { font-size: 0.75rem; opacity: 0.6; margin-top: 0.5rem; }
  .styled-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  .styled-table th { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.06em; text-transform: uppercase; color: #666; padding: 0.4rem 0.6rem; border-bottom: 2px solid #ddd; text-align: right; }
  .styled-table th:first-child { text-align: left; }
  .styled-table td { padding: 0.4rem 0.6rem; border-bottom: 1px solid #f0f0f0; text-align: right; font-variant-numeric: tabular-nums; }
  .styled-table td:first-child { text-align: left; font-weight: 500; }
  .styled-table tr:hover td { background: #fafaf8; }
</style>
""", unsafe_allow_html=True)


def fmt_pct(v, d=1): return f"{v*100:.{d}f}%"
def fmt_margin(v, d=1):
    m = (v * 2 - 1) * 100
    return f"D +{m:.{d}f}" if m >= 0 else f"R +{-m:.{d}f}"


# ── Prior: forecast parameters + simulation ──────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_params():
    return P.load_params()

@st.cache_data(show_spinner=False)
def _load_bg():
    return P.load_bg_county_leans()

@st.cache_data(show_spinner="Running simulations…")
def _simulate(n, env, lean_method, district_sd, county_sd, state_sd, counties_frozen, bg_frozen):
    return P.run_simulation(n, env, lean_method, P.SEED, district_sd, county_sd, state_sd,
                            counties_frozen, (), bg_county_leans_frozen=bg_frozen)

PARAMS          = _load_params()
COUNTIES        = PARAMS["counties"]
BG_COUNTY_LEANS = _load_bg()
model_env_val   = PARAMS["state_env"]
STATE_ENV_SD    = PARAMS["state_env_sd"]

with st.sidebar:
    st.markdown("### ⚙️ Forecast prior")
    st.caption("Defaults match the forecast app's defaults.")
    n_sims = st.number_input("Simulations", min_value=1000, max_value=100_000,
                             value=P.N_DEFAULT, step=1000)
    override_env = st.toggle("Override model environment", value=False)
    if override_env:
        _model_margin = round((model_env_val * 2 - 1) * 100, 1)
        _override_margin = st.number_input("Manual margin (D − R)", value=_model_margin,
                                           step=0.1, format="%.1f")
        forecast_env = (_override_margin / 100 + 1) / 2
        st.caption(f"Manual override · Model was: {fmt_margin(model_env_val)}")
    else:
        forecast_env = model_env_val
        st.caption(f"Model forecast: {fmt_margin(model_env_val)} (SD={fmt_pct(STATE_ENV_SD)})")

    _lean_opts = ["Linear", "Average"] + (["Block Group"] if BG_COUNTY_LEANS is not None else [])
    _lean_default = _lean_opts.index("Block Group") if "Block Group" in _lean_opts else 0
    lean_method = st.radio("Lean Method", _lean_opts, index=_lean_default, horizontal=True)
    st.caption(f"Parameters: {PARAMS['source']}")

_res = _simulate(
    n_sims, forecast_env, lean_method,
    PARAMS["district_turnout_sd"], PARAMS["county_turnout_sd"], STATE_ENV_SD,
    P.counties_to_frozen(COUNTIES),
    P.bg_frozen(BG_COUNTY_LEANS) if lean_method == "Block Group" else None,
)
district_share = _res["district_share"]
county_shares  = _res["county_shares"]

# The tab read the model environment from the forecast app's session state.
st.session_state["model_forecast_env"] = model_env_val * 100


# ── Live-results fetch (shared across sessions via a marker file's mtime) ────
_FETCHER       = DATA_DIR / "fetch_election_results.py"
_FETCH_MARKER  = Path(tempfile.gettempdir()) / "sd4_election_night_last_fetch"
_FETCH_EVERY_S = 600
_FETCH_TIMEOUT = 120   # 13 SD4 + 58 governor county calls; 60 s could cut it off

def _seconds_since_fetch() -> float:
    try:
        return _time.time() - _FETCH_MARKER.stat().st_mtime
    except FileNotFoundError:
        return float("inf")

def _run_fetcher() -> None:
    """Pull fresh SOS results into election_night.csv / governor_night.csv."""
    _FETCH_MARKER.touch()   # mark the attempt first so a failing API isn't retried every rerun
    try:
        subprocess.run([sys.executable, str(_FETCHER)], cwd=str(DATA_DIR),
                       timeout=_FETCH_TIMEOUT, capture_output=True)
    except Exception:
        pass   # non-fatal — stale data still shows


# ══════════════════════════════════════════════════════════════════════════════
st.title("🌙 SD4 Election Night")
st.markdown(
    "Live results for SD4 counties. Enter results manually or step through a mock simulation "
    "to see how win probability updates as votes come in."
)
st.caption(
    f"Prior: Monte Carlo · {int(n_sims):,} simulations · Lean: {lean_method} · "
    f"State env: {fmt_pct(forecast_env)} (SD={fmt_pct(STATE_ENV_SD)}) · Win threshold: 50%"
)

# ── Mock scenario controls ────────────────────────────────────────────────
MOCK_DIR = DATA_DIR / "mock_results"
MOCK_SCENARIOS = {
    "— none —":      None,
    "Narrow win":    "narrow_win",
    "Comfortable win": "comfortable_win",
    "Narrow loss":   "narrow_loss",
}
WAVE_LABELS_DISPLAY = [
    "8:00 PM", "9:00 PM", "11:00 PM",
    "9:00 AM D+1", "4:00 PM D+1",
    "4:00 PM D+2", "4:00 PM D+3", "4:00 PM D+4",
    "4:00 PM D+5", "4:00 PM D+6", "4:00 PM D+7",
]
WAVE_FILENAMES = [
    "800_PM.csv", "900_PM.csv", "1100_PM.csv",
    "900_AM_D1.csv", "400_PM_D1.csv",
    "400_PM_D2.csv", "400_PM_D3.csv", "400_PM_D4.csv",
    "400_PM_D5.csv", "400_PM_D6.csv", "400_PM_D7.csv",
]

rc1, rc2 = st.columns([1, 1])
with rc1:
    auto_refresh = st.toggle("Auto-refresh", value=True, key="night_auto_refresh")
with rc2:
    manual_refresh = st.button("🔄 Refresh", key="night_manual_refresh")

with st.expander("🧪 Mock scenario (testing only)", expanded=False):
    mc1, mc2 = st.columns([2, 2])
    with mc1:
        mock_scenario_label = st.selectbox(
            "Mock scenario", list(MOCK_SCENARIOS.keys()), key="night_mock_scenario"
        )
    with mc2:
        mock_wave_idx = st.select_slider(
            "Reporting wave",
            options=list(range(len(WAVE_LABELS_DISPLAY))),
            format_func=lambda i: WAVE_LABELS_DISPLAY[i],
            value=0,
            key="night_mock_wave",
            disabled=(mock_scenario_label == "— none —"),
        )

# ── Load / initialise election night data ─────────────────────────────────
night_path = DATA_DIR / "election_night.csv"
gov_path   = DATA_DIR / "governor_night.csv"

def _mtime(p) -> float:
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0

@st.cache_data(show_spinner=False, ttl=60)
def load_night_csv(p, mtime=0.0):
    # mtime is part of the cache key, so a rewritten CSV is re-read at once
    try:
        return pd.read_csv(p)
    except Exception:
        return None

_mock_active = mock_scenario_label != "— none —"

_NOW_PT = pd.Timestamp.now(tz="America/Los_Angeles").tz_localize(None)

def _parse_sos_time(s):
    """'November 4, 2026, 4:22 p.m.' (SOS) or ISO UTC → naive Pacific timestamp, else NaT."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return pd.NaT
    t = str(s).strip().replace("a.m.", "AM").replace("p.m.", "PM")
    try:
        return pd.to_datetime(t, format="%B %d, %Y, %I:%M %p")
    except Exception:
        pass
    try:
        return pd.to_datetime(t, utc=True).tz_convert("America/Los_Angeles").tz_localize(None)
    except Exception:
        return pd.NaT

def _fmt_pt(ts) -> str:
    if pd.isna(ts):
        return "—"
    return f"{ts:%b} {ts.day} {ts:%I:%M}".replace(" 0", " ") + ("am" if ts.hour < 12 else "pm")

if manual_refresh:
    with st.spinner("Fetching SOS results…"):
        _run_fetcher()
    st.rerun()

# First load of a session: fetch if nobody has fetched in the last 10 minutes
if auto_refresh and not _mock_active and _seconds_since_fetch() >= _FETCH_EVERY_S:
    with st.spinner("Fetching SOS results…"):
        _run_fetcher()

@st.cache_data(show_spinner=False, ttl=300)
def load_mock_wave(scenario_key, wave_filename):
    """Load a wave CSV from mock_results/<scenario>/<wave>.csv"""
    try:
        p = MOCK_DIR / scenario_key / wave_filename
        return pd.read_csv(p)
    except Exception:
        return None

df_night_base = load_night_csv(str(night_path), _mtime(night_path))
if df_night_base is None:
    st.warning("election_night.csv not found. Create it from the template (see repo).")
    st.stop()

# Apply mock wave if selected
mock_key = MOCK_SCENARIOS[mock_scenario_label]
if mock_key is not None:
    wave_file = WAVE_FILENAMES[mock_wave_idx]
    df_mock = load_mock_wave(mock_key, wave_file)
    if df_mock is not None:
        df_night = df_mock.copy()
        st.info(
            f"📋 Mock: **{mock_scenario_label}** · "
            f"**{WAVE_LABELS_DISPLAY[mock_wave_idx]}** wave — simulated results for testing."
        )
    else:
        df_night = df_night_base.copy()
        st.warning(
            f"Mock wave file not found: mock_results/{mock_key}/{wave_file}. "
            "Run `python3 generate_mock_results.py --scenario all` to generate them."
        )
else:
    df_night = df_night_base.copy()

# ── District summary ──────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">District Summary</div>', unsafe_allow_html=True)

total_dem  = 0; total_rep  = 0; total_other = 0; total_prec = 0; total_prec_all = 0
counties_reporting = 0

for _, row in df_night.iterrows():
    d = row.get("dem_votes", 0);  d = 0 if pd.isna(d) else int(d)
    r = row.get("rep_votes", 0);  r = 0 if pd.isna(r) else int(r)
    o = row.get("other_votes", 0); o = 0 if pd.isna(o) else int(o)
    pr = row.get("precincts_reporting", 0); pr = 0 if pd.isna(pr) else int(pr)
    pt = row.get("precincts_total", 1);     pt = 1 if pd.isna(pt) or pt == 0 else int(pt)
    total_dem  += d; total_rep  += r; total_other += o
    total_prec += pr; total_prec_all += pt
    if d + r + o > 0:
        counties_reporting += 1

total_votes = total_dem + total_rep + total_other
dist_dem_share = total_dem / total_votes if total_votes > 0 else None
pct_precincts  = total_prec / total_prec_all if total_prec_all > 0 else 0

# Compare reported dem share to model's projected district share (mean of simulations)
fp_env = st.session_state.get("model_forecast_env", 59.80) / 100
model_district_mean = float(np.mean(district_share))

dist_est_total = sum(
    int(cd.get("registration", 0) * cd.get("turnout", 0))
    for cd in COUNTIES.values()
)
total_cast = total_dem + total_rep + total_other

# ── Remaining votes: SOS unprocessed-ballot estimates when entered ────────
# unprocessed_ballots.csv holds the SOS "Unprocessed Ballots Status" figures
# (whole county); split counties are scaled by their SD4 share of registration.
@st.cache_data(show_spinner=False)
def _load_unprocessed(p, mtime=0.0):
    return REM.load_unprocessed(p)

@st.cache_data(show_spinner=False)
def _load_hist_prepped(p, mtime=0.0):
    try:
        return REM.prep_history(pd.read_csv(p))
    except Exception:
        return REM.prep_history(None)

@st.cache_data(show_spinner=False)
def _load_fractions(p, mtime=0.0):
    try:
        f = pd.read_csv(p)
        return dict(zip(f["county"], f["sd4_fraction"].astype(float)))
    except Exception:
        return {}

_unproc_path = DATA_DIR / "unprocessed_ballots.csv"
_frac_path   = DATA_DIR / "partial_county_fractions.csv"
UNPROCESSED  = {} if _mock_active else _load_unprocessed(str(_unproc_path), _mtime(_unproc_path))
SD4_FRACTION = _load_fractions(str(_frac_path), _mtime(_frac_path))
_hist_p      = DATA_DIR / "election_night_history.csv"
HIST_PREP    = REM.prep_history(None) if _mock_active else _load_hist_prepped(str(_hist_p), _mtime(_hist_p))
SOS_AGED     = {}   # county → votes subtracted for batches posted after the SOS as_of time

def county_remaining(cn: str, counted: int):
    """(remaining votes, source) for a county: SOS unprocessed × SD4 share minus SD4 votes
    counted since the figure's as_of time, else model estimate."""
    if cn in UNPROCESSED:
        rem, since = REM.sos_remaining(UNPROCESSED[cn], SD4_FRACTION.get(cn, 1.0), counted, HIST_PREP, cn)
        SOS_AGED[cn] = since
        return rem, "sos"
    cd = COUNTIES.get(cn, {})
    est = int(cd.get("registration", 0) * cd.get("turnout", 0))
    return (max(0, est - counted), "model") if est > 0 else (None, "none")

def _row_votes(row):
    v = [row.get(c, 0) for c in ("dem_votes", "rep_votes", "other_votes")]
    return sum(0 if pd.isna(x) else int(x) for x in v)

_rem_by_county = {row.get("county", ""): county_remaining(row.get("county", ""), _row_votes(row))
                  for _, row in df_night.iterrows()}
dist_est_remaining = sum(r for r, _ in _rem_by_county.values() if r is not None)
n_sos_remaining = sum(1 for _, src in _rem_by_county.values() if src == "sos")

_sc = "padding:0.6rem 0.7rem"           # tighter card padding
_lbl = "font-size:0.62rem;font-weight:600;letter-spacing:0.07em;text-transform:uppercase;color:#666;margin-bottom:0.15rem"
_val = "font-family:'IBM Plex Mono',monospace;font-size:1.1rem;font-weight:600;color:#111;line-height:1"
_sub = "font-size:0.62rem;color:#888;margin-top:0.15rem"

def _ns_card(label, value, sub="", val_color=None, border_color=None):
    bc = f"border-color:{border_color}" if border_color else ""
    vc = f"color:{val_color}" if val_color else ""
    s  = f'<div style="{_sub}">{sub}</div>' if sub else ""
    return (f'<div class="stat-card" style="{_sc};{bc}">'
            f'<div style="{_lbl}">{label}</div>'
            f'<div style="{_val};{vc}">{value}</div>{s}</div>')

ns1, ns2, ns3, ns4, ns5, ns6 = st.columns(6)
with ns1:
    v = f"{dist_dem_share*100:.1f}%" if dist_dem_share else "—"
    color = ("#1a6b3c" if dist_dem_share and dist_dem_share >= 0.50 else "#b91c1c") if dist_dem_share else "#888"
    st.markdown(_ns_card("Dem Vote Share", v, "Combined election night", val_color=color, border_color=color), unsafe_allow_html=True)
with ns2:
    if total_votes > 0:
        st.markdown(_ns_card("Vote Totals", f"{total_dem:,} D", f"{total_rep:,} R"
                             + (f" · {total_other:,} other" if total_other else "")), unsafe_allow_html=True)
    else:
        st.markdown(_ns_card("Vote Totals", "—", "No results yet"), unsafe_allow_html=True)
with ns3:
    if total_votes > 0:
        _m = total_dem - total_rep
        _mc = "#1a6b3c" if _m >= 0 else "#b91c1c"
        st.markdown(_ns_card("Vote Margin", f"{'D' if _m >= 0 else 'R'} +{abs(_m):,}",
                             f"{abs(_m) / total_votes * 100:.1f}pt · {counties_reporting}/13 counties in",
                             val_color=_mc, border_color=_mc), unsafe_allow_html=True)
    else:
        st.markdown(_ns_card("Vote Margin", "—", f"{counties_reporting}/13 counties in"), unsafe_allow_html=True)
with ns4:
    if dist_dem_share:
        diff = dist_dem_share - model_district_mean
        arrow = "▲" if diff >= 0 else "▼"
        c2 = "#1a6b3c" if diff >= 0 else "#b91c1c"
        st.markdown(_ns_card("vs Model Forecast", f"{arrow} {abs(diff)*100:.1f}pt", f"Model median: {model_district_mean*100:.1f}%", val_color=c2), unsafe_allow_html=True)
    else:
        st.markdown(_ns_card("vs Model Forecast", "—", "No results yet"), unsafe_allow_html=True)
with ns5:
    st.markdown(_ns_card("Total Votes Cast", f"{total_cast:,}"), unsafe_allow_html=True)
with ns6:
    rem_str = f"~{dist_est_remaining:,}" if dist_est_total > 0 else "—"
    rem_sub = (f"SOS unprocessed for {n_sos_remaining}/13 counties" if n_sos_remaining
               else f"of ~{dist_est_total:,} est. total")
    st.markdown(_ns_card("Est. Votes Remaining", rem_str, rem_sub), unsafe_allow_html=True)

# ── Mail-ballot timing model (used by live win prob + targets) ───────────
# Reporting speed: fraction of final county votes in at each of 11 checkpoints.
_SPEED = {
    #                  8pm   9pm  11pm  9am+1 4pm+1  4pm+2 4pm+3 4pm+4 4pm+5 4pm+6 4pm+7
    "Alpine":    [0.80, 0.92, 0.96, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
    "Mono":      [0.78, 0.91, 0.95, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
    "Mariposa":  [0.70, 0.86, 0.93, 0.97, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
    "Inyo":      [0.55, 0.75, 0.88, 0.94, 0.97, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99],
    "Amador":    [0.52, 0.72, 0.86, 0.93, 0.97, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99],
    "Calaveras": [0.45, 0.68, 0.82, 0.90, 0.95, 0.97, 0.98, 0.99, 0.99, 0.99, 0.99],
    "Tuolumne":  [0.40, 0.62, 0.78, 0.87, 0.93, 0.96, 0.97, 0.98, 0.99, 0.99, 0.99],
    "El Dorado": [0.65, 0.78, 0.90, 0.97, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
    "Placer":    [0.60, 0.72, 0.82, 0.85, 0.87, 0.89, 0.90, 0.91, 0.93, 0.95, 0.96],
    "Madera":    [0.38, 0.55, 0.65, 0.68, 0.72, 0.76, 0.81, 0.86, 0.90, 0.93, 0.95],
    "Merced":    [0.22, 0.38, 0.44, 0.46, 0.50, 0.56, 0.63, 0.70, 0.78, 0.85, 0.90],
    "Stanislaus":[0.32, 0.48, 0.62, 0.63, 0.65, 0.67, 0.73, 0.78, 0.84, 0.89, 0.93],
    "Nevada":    [0.12, 0.20, 0.35, 0.38, 0.42, 0.46, 0.52, 0.59, 0.65, 0.70, 0.75],
}
_MAIL_PCT = {
    "Alpine":0.72, "Amador":0.70, "Calaveras":0.73, "El Dorado":0.68,
    "Inyo":0.71, "Madera":0.62, "Mariposa":0.74, "Merced":0.60,
    "Mono":0.75, "Nevada":0.76, "Placer":0.69, "Stanislaus":0.63,
    "Tuolumne":0.72,
}
_MAIL_BOOST = 0.04  # fallback: fixed 4pp Dem mail advantage

# ── Dynamic mail-ballot Dem boost from Optiq outstanding ballots ──────────
# Use Optiq crosstab data (D/R/O issued vs returned) to estimate what the
# outstanding (unreturned) VBM ballots look like by party, then calculate
# an expected Dem share of outstanding mail, compared to Dem share of total
# mail issued — the gap becomes our _mail_boost_live.
#
# Method:
#   outstanding_D = issued_D - returned_D  (Dems who haven't voted yet)
#   outstanding_R = issued_R - returned_R
#   outstanding_O = issued_O - returned_O
#   outstanding_dem_share = outstanding_D / (outstanding_D + outstanding_R + outstanding_O)
#   overall_dem_share     = issued_D / total_issued
#
# The "boost" is: how much more Democratic are the outstanding ballots vs
# the overall electorate? This drives how reported share understates final.
#
# Historical vote share by party is assumed from forecast lean:
#   Dem support: D votes 95% Dem, R 5% Dem, O 40% Dem (rough priors)
_PARTY_DEM_VOTE_SHARE = {"D": 0.95, "R": 0.05, "O": 0.38}

@st.cache_data(ttl=3600)
def _outstanding_mail_boost(crosstab_csv: str) -> float:
    """
    Estimate mail-ballot Dem advantage from Optiq party composition.
    Returns the implied boost (Dem mail share - overall mail share),
    scaled to vote share (not registration share).
    Falls back to _MAIL_BOOST if data unavailable.
    """
    try:
        ct = pd.read_csv(crosstab_csv)
    except Exception:
        return _MAIL_BOOST
    if ct.empty or "snapshot_date" not in ct.columns:
        return _MAIL_BOOST
    latest_ct = ct[ct["snapshot_date"] == ct["snapshot_date"].max()]
    if latest_ct.empty:
        return _MAIL_BOOST
    # Aggregate to party level
    party_agg = (latest_ct.groupby("party")[["issued","returned"]]
                 .sum().reset_index())
    party_agg = party_agg[party_agg["party"].isin(["D","R","O"])]
    if party_agg.empty:
        return _MAIL_BOOST
    pa = party_agg.set_index("party")
    for p in ["D","R","O"]:
        if p not in pa.index:
            pa.loc[p] = {"issued": 0, "returned": 0}
    pa["outstanding"] = (pa["issued"] - pa["returned"]).clip(lower=0)
    total_outstanding = pa["outstanding"].sum()
    total_issued      = pa["issued"].sum()
    if total_outstanding < 1000 or total_issued == 0:
        return _MAIL_BOOST  # too few outstanding ballots to be meaningful
    # Weighted Dem vote share of outstanding vs overall issued
    out_dem_vote  = sum(pa.loc[p,"outstanding"] * _PARTY_DEM_VOTE_SHARE.get(p,0.5)
                        for p in ["D","R","O"])
    iss_dem_vote  = sum(pa.loc[p,"issued"]      * _PARTY_DEM_VOTE_SHARE.get(p,0.5)
                        for p in ["D","R","O"])
    out_dem_share = out_dem_vote / total_outstanding if total_outstanding > 0 else 0.5
    iss_dem_share = iss_dem_vote / total_issued      if total_issued      > 0 else 0.5
    boost = float(np.clip(out_dem_share - iss_dem_share, -0.08, 0.12))
    return boost if abs(boost) > 0.005 else _MAIL_BOOST

_CROSSTAB_CSV_PATH = str(DATA_DIR / "optiq_crosstabs.csv")
_mail_boost_live   = _outstanding_mail_boost(_CROSSTAB_CSV_PATH)

def _implied_final_share(reported: float, county: str, f: float) -> float:
    """Back-calculate the implied FINAL Dem share from a reported share.
    Inverts _wave_share() analytically.
    f = fraction of estimated total votes that are reported.
    Uses _mail_boost_live (from Optiq outstanding-ballot party composition)
    when available; falls back to static _MAIL_BOOST = 0.04.
    """
    boost = _mail_boost_live
    mp = _MAIL_PCT.get(county, 0.68)
    if f <= 0 or f > 1:
        return reported
    if f < mp:
        implied = reported - boost * (1 - mp)
    else:
        implied = reported - boost * mp * (1 - f) / f
    return float(np.clip(implied, 0.0, 1.0))

def _wave_share(final_share: float, county: str, wave_idx: int) -> float:
    """Estimate the REPORTED Dem share at a given wave checkpoint."""
    boost = _mail_boost_live
    f  = _SPEED.get(county, [0.5]*N_WAVES)[wave_idx]
    mp = _MAIL_PCT.get(county, 0.68)
    mail_share = min(1.0, final_share + boost * (1 - mp))
    ed_share   = max(0.0, final_share - boost * mp)
    if f <= 0:
        return final_share
    if f >= mp:
        ed_frac_in = (f - mp) / (1 - mp) if (1 - mp) > 0 else 1.0
        votes_in   = mp * mail_share + ed_frac_in * (1 - mp) * ed_share
    else:
        mail_frac_in = f / mp
        votes_in     = mail_frac_in * mp * mail_share
    return votes_in / f

# ── Live win probability: late-vote model (default) ───────────────────────
# final = votes counted + remaining votes × late-vote Dem share, per county.
# See late_model.py for how the late-vote share and its uncertainty are set.
import late_model as LM
LM = importlib.reload(LM)
import batches as BATCH
BATCH = importlib.reload(BATCH)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Live Win Probability</div>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def _load_csv(p, mtime=0.0):
    try:
        return pd.read_csv(p)
    except Exception:
        return None

_ms_path, _fd_path = DATA_DIR / "mail_splits.csv", DATA_DIR / "first_drop.csv"
_hist_path = DATA_DIR / "election_night_history.csv"
_mail_splits = None if _mock_active else _load_csv(str(_ms_path), _mtime(_ms_path))
_first_drop  = None if _mock_active else _load_csv(str(_fd_path), _mtime(_fd_path))
_late_b = None
if not _mock_active:
    _hb = BATCH.compute_batches(_load_csv(str(_hist_path), _mtime(_hist_path)))
    if not _hb.empty:
        _hb["_posted"] = [_parse_sos_time(x) for x in _hb["last_updated"]]
        _late_b = _hb[_hb["_posted"] >= LM.LATE_COUNT_START]

with st.expander("⚙️ Late-vote model settings", expanded=False):
    lm1, lm2, lm3, lm4 = st.columns(4)
    _lm_shared = lm1.slider("District-wide late-lean uncertainty (SD)", 0.0, 5.0, 2.0, 0.5, format="%.1f pt",
                            help="One shift applied to every county's late votes together") / 100
    _lm_county = lm2.slider("County late-lean uncertainty (SD)", 0.0, 6.0, 3.0, 0.5, format="%.1f pt",
                            help="Independent per county; shrinks as late batches arrive") / 100
    _lm_rem = lm3.slider("Remaining-vote size uncertainty (SD)", 0, 40, 20, 5, format="%d%%",
                         help="For model estimates; SOS unprocessed figures use 7%") / 100
    _lm_k = lm4.slider("Weight on mail split / first drop (votes)", 500, 10000, 3000, 500,
                       help="How many late-batch votes it takes to outweigh the prior lean equally")
_lm_s = LM.LateSettings(prior_weight=float(_lm_k), county_sd=_lm_county, shared_sd=_lm_shared,
                        rem_sd_model=_lm_rem)

if total_votes > 0 and counties_reporting > 0:
    _ci = LM.county_inputs(df_night, _rem_by_county, _mail_splits, _first_drop, _late_b,
                           mock_mail=_mock_active, s=_lm_s)
    _lm = LM.simulate(_ci, _lm_s)
    wp_l = _lm["win_prob"]
    col_l = "#1a6b3c" if wp_l >= 0.60 else ("#d97706" if wp_l >= 0.40 else "#b91c1c")
    lw1, lw2, lw3 = st.columns([1, 1, 1])
    with lw1:
        st.markdown(
            f'<div class="win-hero" style="background:linear-gradient(135deg,{col_l} 0%,{col_l}cc 100%)">'
            f'<div class="wlabel">Win Probability</div><div class="wvalue">{wp_l*100:.0f}%</div>'
            f'<div class="wsub">Late-vote model</div></div>', unsafe_allow_html=True)
    with lw2:
        st.markdown(f'<div class="stat-card"><div class="label">Projected Final Dem Share</div>'
                    f'<div class="value">{_lm["share_mean"]*100:.1f}%</div>'
                    f'<div class="sub">90% range {_lm["share_p05"]*100:.1f}–{_lm["share_p95"]*100:.1f}%</div></div>',
                    unsafe_allow_html=True)
        _mm = _lm["margin_median"]
        st.markdown(f'<div class="stat-card"><div class="label">Projected Final Margin</div>'
                    f'<div class="value" style="font-size:1.3rem">{"D" if _mm >= 0 else "R"} +{abs(_mm):,.0f}</div>'
                    f'<div class="sub">90% range {_lm["margin_p05"]:+,.0f} to {_lm["margin_p95"]:+,.0f} (D − R)</div></div>',
                    unsafe_allow_html=True)
    with lw3:
        st.markdown(f'<div class="stat-card"><div class="label">Dem Share Needed of Remaining</div>'
                    f'<div class="value">{_lm["needed_share"]*100:.1f}%</div>'
                    f'<div class="sub">of ~{_lm["remaining"]:,.0f} remaining votes</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-card"><div class="label">Expected Dem Share of Remaining</div>'
                    f'<div class="value">{_lm["expected_late_share"]*100:.1f}%</div>'
                    f'<div class="sub">weighted late-vote lean across counties</div></div>', unsafe_allow_html=True)

    _lrows = ""
    for _, c in _ci.sort_values("remaining", ascending=False).iterrows():
        _late = f"{c.late_share*100:.1f}% of {c.late_votes:,}" if c.late_votes > 0 else "—"
        _net = c.exp_net
        _lrows += (
            f"<tr><td>{c.county}</td><td>{c.prior_source}</td>"
            f"<td>{'—' if np.isnan(c.prior) else f'{c.prior*100:.1f}%'}</td><td>{_late}</td>"
            f"<td><strong>{'—' if np.isnan(c.lean) else f'{c.lean*100:.1f}%'}</strong></td>"
            f"<td>{c.remaining:,.0f}{'*' if c.remaining_source == 'sos' else ''}</td>"
            f"<td style='color:{'#1a6b3c' if _net >= 0 else '#b91c1c'}'>{'D' if _net >= 0 else 'R'} +{abs(_net):,.0f}</td></tr>"
        )
    with st.expander("Late-vote lean by county", expanded=True):
        st.markdown(
            '<div style="overflow-x:auto"><table class="styled-table" style="min-width:760px">'
            '<thead><tr><th>County</th><th>Lean Source</th><th>Mail / First-Drop Dem Share</th>'
            '<th>Post-Election-Day Batches</th><th>Late-Vote Lean Used</th><th>Remaining Votes</th>'
            '<th>Expected Net</th></tr></thead>'
            f'<tbody>{_lrows}</tbody></table></div>'
            '<p style="font-size:0.72rem;color:#999;margin-top:0.4rem">'
            'Lean source: <b>county mail split</b> = the county\'s own vote-by-mail results for SD4 (fetch_mail_splits.py); '
            '<b>first drop</b> = its first election-night report, almost all early mail, for counties that publish totals '
            'only. Post-election-day batches (posted after 5 am on 4 Nov) are blended in, weighted by votes. '
            'Two-party shares. Remaining votes as in County Results (* = SOS unprocessed figure).</p>',
            unsafe_allow_html=True)
    st.caption(
        "Simulates each county's remaining votes at its late-vote lean, with a district-wide shift (counties' late "
        "votes leaning the same way together), a county-level shift and uncertainty in how many votes remain. "
        "Settings above."
    )
else:
    prior_wp = float(np.mean(district_share >= WIN_THRESHOLD))
    col_pw, _ = st.columns([1, 3])
    with col_pw:
        st.markdown(
            f'<div class="win-hero"><div class="wlabel">Pre-results Win Probability</div>'
            f'<div class="wvalue">{prior_wp*100:.0f}%</div>'
            f'<div class="wsub">No results entered yet</div></div>',
            unsafe_allow_html=True
        )
    st.caption("Enter results in the county table below or load a mock scenario to see live filtering.")

with st.expander("Previous method: mail-adjusted estimate (for comparison)", expanded=False):
    # ── Outstanding-ballot model info ─────────────────────────────────────────
    _boost_src = "Optiq party composition" if abs(_mail_boost_live - _MAIL_BOOST) > 0.001 else "static fallback"
    st.caption(
        f"✉️ Mail-ballot Dem advantage used in adjustments: **{_mail_boost_live:+.1%}** "
        f"({_boost_src}). "
        + ("Derived from D/R/O outstanding VBM party composition (Optiq)."
           if _boost_src != "static fallback"
           else "Optiq crosstab data not yet available — using fixed +4 pp assumption.")
    )

    # ── Live win probability ───────────────────────────────────────────────────

    if total_votes > 0 and counties_reporting > 0:
        # Build a mask: for counties with results, filter simulations where county share ≈ reported
        # For counties without results, use full simulation range
        live_mask = np.ones(len(district_share), dtype=bool)

        # ── Mail-adjusted win probability (Option 2) ──────────────────────────
        # For each reporting county, invert the mail-ballot timing model to recover
        # the implied true FINAL share.  Then re-run a mini Monte Carlo using:
        #   • implied final (± lean_sd) for reporting counties
        #   • full simulation distribution for non-reporting counties
        # This removes the early-count mail-ballot inflation that makes the raw
        # probability misleadingly high when only mail votes are in.
        rng_adj = np.random.default_rng(42)
        N_ADJ   = int(n_sims)

        # Per-county: collect implied finals and their uncertainty
        implied_finals   = {}   # cn -> implied final share (point estimate)
        county_est_total = {}   # cn -> estimated total votes

        for _, row in df_night.iterrows():
            cn  = row.get("county")
            d2  = row.get("dem_votes",  0); d2  = 0 if pd.isna(d2)  else int(d2)
            r2  = row.get("rep_votes",  0); r2  = 0 if pd.isna(r2)  else int(r2)
            o2  = row.get("other_votes",0); o2  = 0 if pd.isna(o2)  else int(o2)
            tv2 = d2 + r2 + o2
            if tv2 == 0 or cn not in county_shares:
                continue
            reported_share = d2 / tv2
            # Filter to simulations within ±2 lean SDs of reported share
            lean_sd = COUNTIES.get(cn, {}).get("lean_sd", 0.02)
            tol = max(0.03, lean_sd * 2.0)
            live_mask &= np.abs(county_shares[cn] - reported_share) <= tol

            # Compute fraction reported for this county
            cd = COUNTIES.get(cn, {})
            est_total = int(cd.get("registration", 0) * cd.get("turnout", 0))
            f_reported = tv2 / est_total if est_total > 0 else 0.5
            f_reported = min(max(f_reported, 0.01), 1.0)

            implied_final = _implied_final_share(reported_share, cn, f_reported)
            implied_finals[cn]   = implied_final
            county_est_total[cn] = min(tv2, est_total)   # votes counted so far

        # Build adjusted district-share distribution
        adj_district_dem  = np.zeros(N_ADJ)
        adj_district_total = np.zeros(N_ADJ)

        for cn, cd in COUNTIES.items():
            est_total = int(cd.get("registration", 0) * cd.get("turnout", 0))
            if est_total == 0:
                continue
            lean_sd = cd.get("lean_sd", 0.02)
            if cn in implied_finals:
                # Reporting county: centre on implied final, add lean_sd uncertainty
                # as we have partial info but not perfect knowledge of final outcome
                remaining_frac = max(0.0, 1.0 - (county_est_total.get(cn, est_total) /
                                                   max(1, est_total)))
                # Uncertainty shrinks as more votes are in; add remaining-vote noise
                noise_sd = lean_sd * (remaining_frac ** 0.5 + 0.3)
                county_sims = rng_adj.normal(implied_finals[cn], noise_sd, N_ADJ)
                county_sims = np.clip(county_sims, 0.0, 1.0)
            else:
                # Non-reporting county: use full simulation distribution
                county_sims = county_shares[cn][:N_ADJ] if len(county_shares[cn]) >= N_ADJ else \
                              np.resize(county_shares[cn], N_ADJ)

            adj_district_dem   += county_sims * est_total
            adj_district_total += est_total

        adj_district_share = adj_district_dem / adj_district_total
        adj_wp   = float(np.mean(adj_district_share >= WIN_THRESHOLD))
        adj_mean = float(np.mean(adj_district_share))

        n_live = int(live_mask.sum())
        if n_live >= 50:
            live_share = district_share[live_mask]
            live_wp    = float(np.mean(live_share >= WIN_THRESHOLD))
            live_mean  = float(np.mean(live_share))

            lw1, lw2, lw3 = st.columns([1, 2, 2])
            with lw1:
                color_adj = "#1a6b3c" if adj_wp >= 0.60 else ("#d97706" if adj_wp >= 0.40 else "#b91c1c")
                st.markdown(
                    f'<div class="win-hero" style="background:linear-gradient(135deg,{color_adj} 0%,{color_adj}cc 100%)">'
                    f'<div class="wlabel">Win Probability</div>'
                    f'<div class="wvalue">{adj_wp*100:.0f}%</div>'
                    f'<div class="wsub">Mail-adjusted estimate</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="stat-card" style="margin-top:0.5rem;font-size:0.82rem">'
                    f'<div class="label">Raw (unadjusted)</div>'
                    f'<div class="value" style="font-size:1.1rem">{live_wp*100:.0f}%</div>'
                    f'<div class="sub">{n_live:,} matching simulations</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with lw2:
                st.markdown(f'<div class="stat-card"><div class="label">Implied Final Share</div><div class="value">{adj_mean*100:.1f}%</div><div class="sub">Mail-adjusted district estimate</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="stat-card"><div class="label">5th–95th Pct Range</div><div class="value" style="font-size:1.2rem">{np.percentile(adj_district_share,5)*100:.1f}–{np.percentile(adj_district_share,95)*100:.1f}%</div></div>', unsafe_allow_html=True)
                if implied_finals:
                    imp_rows = "".join(
                        f"<tr><td>{cn}</td><td>{implied_finals[cn]*100:.1f}%</td></tr>"
                        for cn in sorted(implied_finals)
                    )
                    st.markdown(
                        f'<details style="font-size:0.78rem;margin-top:0.4rem"><summary style="cursor:pointer;color:#555">Implied finals by county</summary>'
                        f'<table style="width:100%;border-collapse:collapse;margin-top:0.3rem">'
                        f'<thead><tr><th style="text-align:left;padding:2px 6px;font-size:0.75rem">County</th>'
                        f'<th style="text-align:left;padding:2px 6px;font-size:0.75rem">Implied Final</th></tr></thead>'
                        f'<tbody>{imp_rows}</tbody></table></details>',
                        unsafe_allow_html=True
                    )
            with lw3:
                fig_lw, ax_lw = plt.subplots(figsize=(5, 2.5))
                fig_lw.patch.set_facecolor("#f7f7f5"); ax_lw.set_facecolor("#f7f7f5")
                ax_lw.hist(adj_district_share*100, bins=40, color="#1a6b3c", alpha=0.75, edgecolor="none", label="Mail-adjusted")
                ax_lw.hist(live_share*100, bins=40, color="#888", alpha=0.35, edgecolor="none", label="Raw (filtered sims)")
                ax_lw.axvline(50, color="#b91c1c", linewidth=1.5, linestyle="--")
                if dist_dem_share:
                    ax_lw.axvline(dist_dem_share*100, color="#d97706", linewidth=1.5, linestyle="-", label=f"Reported {dist_dem_share*100:.1f}%")
                ax_lw.set_xlabel("Dem Vote Share (%)", fontsize=7)
                ax_lw.xaxis.set_major_formatter(mtick.PercentFormatter())
                ax_lw.spines[["top","right","left"]].set_visible(False)
                ax_lw.tick_params(labelsize=6)
                ax_lw.legend(fontsize=6, framealpha=0)
                plt.tight_layout()
                st.pyplot(fig_lw, width="stretch"); plt.close()
            st.caption(
                "**Mail-adjusted** win probability strips out early mail-ballot lean "
                "by back-calculating each county's implied final share from the current "
                "reported share and fraction of votes in.  Raw probability shown for reference."
            )
        else:
            st.info(f"Only {n_live} simulations match current results — too few to estimate win probability. Results may be partial or extreme.")
            # Still show adjusted estimate even when filter is too narrow
            color_adj = "#1a6b3c" if adj_wp >= 0.60 else ("#d97706" if adj_wp >= 0.40 else "#b91c1c")
            col_adj1, col_adj2, _ = st.columns([1, 1, 2])
            with col_adj1:
                st.markdown(
                    f'<div class="win-hero" style="background:linear-gradient(135deg,{color_adj} 0%,{color_adj}cc 100%)">'
                    f'<div class="wlabel">Mail-Adjusted Win Prob</div>'
                    f'<div class="wvalue">{adj_wp*100:.0f}%</div>'
                    f'<div class="wsub">Implied finals estimate</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col_adj2:
                st.markdown(f'<div class="stat-card"><div class="label">Pre-results Win Probability</div><div class="value">{float(np.mean(district_share>=WIN_THRESHOLD))*100:.0f}%</div><div class="sub">From prior simulation (no live filter)</div></div>', unsafe_allow_html=True)
            st.caption("Mail-adjusted estimate still available even when simulation filter is too narrow.")
    else:
        prior_wp = float(np.mean(district_share >= WIN_THRESHOLD))
        col_pw, _ = st.columns([1, 3])
        with col_pw:
            st.markdown(
                f'<div class="win-hero"><div class="wlabel">Pre-results Win Probability</div>'
                f'<div class="wvalue">{prior_wp*100:.0f}%</div>'
                f'<div class="wsub">No results entered yet</div></div>',
                unsafe_allow_html=True
            )
        st.caption("Enter results in the county table below or load a mock scenario to see live filtering.")

# ── Pre-compute win-target medians (used in county results table) ──────────
_win_mask_pre = district_share >= WIN_THRESHOLD
_win_tgt_p50  = {
    cn: float(np.median(county_shares[cn][_win_mask_pre]))
    for cn in county_shares
    if _win_mask_pre.sum() > 0
}

# ── County results table ───────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">County Results</div>', unsafe_allow_html=True)


crows = ""
for _, row in df_night.iterrows():
    cn   = row.get("county", "")
    d3   = row.get("dem_votes",  0); d3 = 0 if pd.isna(d3)  else int(d3)
    r3   = row.get("rep_votes",  0); r3 = 0 if pd.isna(r3)  else int(r3)
    o3   = row.get("other_votes",0); o3 = 0 if pd.isna(o3)  else int(o3)
    tv3  = d3 + r3 + o3
    pr3  = row.get("precincts_reporting", 0); pr3 = 0 if pd.isna(pr3) else int(pr3)
    pt3  = row.get("precincts_total", 1);     pt3 = 1 if (pd.isna(pt3) or pt3==0) else int(pt3)
    pct3 = pr3 / pt3 if pt3 > 0 else 0

    county_data   = COUNTIES.get(cn, {})
    est_total     = int(county_data.get("registration", 0) * county_data.get("turnout", 0))
    votes_remaining, rem_src = county_remaining(cn, tv3)

    win_tgt3 = _win_tgt_p50.get(cn)
    if tv3 > 0:
        rep_share3 = d3 / tv3
        model_share3 = forecast_env + county_data.get("lean_lin", 0)
        diff3 = rep_share3 - model_share3
        share_str = f"{rep_share3*100:.1f}%"
        diff_str  = f"{'▲' if diff3>=0 else '▼'} {abs(diff3)*100:.1f}pt"
        diff_col  = "#1a6b3c" if diff3 >= 0 else "#b91c1c"
        status_icon = "🟢" if rep_share3 >= 0.50 else "🔴"
        if win_tgt3 is not None:
            tgt_diff3 = rep_share3 - win_tgt3
            tgt_str   = f"{'▲' if tgt_diff3>=0 else '▼'} {abs(tgt_diff3)*100:.1f}pt"
            tgt_col   = "#1a6b3c" if tgt_diff3 >= 0 else "#b91c1c"
        else:
            tgt_str = "—"; tgt_col = "#888"
    else:
        share_str = "—"; diff_str = "—"; diff_col = "#888"; status_icon = "⚪"
        tgt_str = "—"; tgt_col = "#888"

    upd_ts   = _parse_sos_time(row.get("last_updated"))
    is_new   = pd.notna(upd_ts) and (_NOW_PT - upd_ts) <= pd.Timedelta(hours=1)
    pct_str  = (f"<span style='color:#1a6b3c;font-weight:600'>🆕 {_fmt_pt(upd_ts)}</span>" if is_new
                else (_fmt_pt(upd_ts) if not _mock_active else str(row.get("last_updated", "—"))))
    rem_str  = ((f"{votes_remaining:,}*" if rem_src == "sos" else f"~{votes_remaining:,}")
                if votes_remaining is not None else "—")
    crows += (
        f"<tr><td>{status_icon} {cn}</td>"
        f"<td>{d3:,}</td><td>{r3:,}</td>"
        f"<td>{share_str}</td>"
        f"<td style='color:{diff_col}'>{diff_str}</td>"
        f"<td style='color:{tgt_col}'>{tgt_str}</td>"
        f"<td>{pct_str}</td>"
        f"<td>{rem_str}</td></tr>"
    )

st.markdown(
    f'<table class="styled-table">'
    f'<thead><tr><th>County</th><th>Dem Votes</th><th>Rep Votes</th>'
    f'<th>Dem Share</th><th>vs Forecast</th><th>vs Win target</th><th>Last Update</th><th>Est. Votes Remaining</th>'
    f'</tr></thead><tbody>{crows}</tbody></table>'
    f'<p style="font-size:0.72rem;color:#999;margin-top:0.4rem">'
    f'🟢 leading · 🔴 trailing · ⚪ no results · vs Forecast uses linear lean · '
    f'vs Win target = actual share vs median share in winning simulations · '
    f'Last Update = the county\'s latest SOS report (🆕 within the past hour) · '
    f'Est. Votes Remaining = (registration × expected turnout) − votes counted, or * SOS unprocessed '
    f'ballots × SD4 share of the county where entered in unprocessed_ballots.csv, minus SD4 votes the '
    f'county has posted since that figure\'s as_of time'
    + (" (" + ", ".join(f"{c} −{v:,}" for c, v in SOS_AGED.items() if v > 0) + ")"
       if any(v > 0 for v in SOS_AGED.values()) else "")
    + '</p>',
    unsafe_allow_html=True
)

# ── Late count: new batches ────────────────────────────────────────────────
# Each time a county's count changes, the fetcher appends a row to
# election_night_history.csv. A batch is what one county added in one update;
# comparing its Dem share with the county's running share shows whether the
# late count is leaning one way.
import batches as BATCH
BATCH = importlib.reload(BATCH)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Late Count — New Batches</div>', unsafe_allow_html=True)

hist_path = DATA_DIR / "election_night_history.csv"

@st.cache_data(show_spinner=False)
def _load_history(p, mtime=0.0):
    try:
        return pd.read_csv(p)
    except Exception:
        return None

if _mock_active:
    st.caption("Not shown while a mock scenario is selected.")
else:
    _hist = _load_history(str(hist_path), _mtime(hist_path))
    _b = BATCH.compute_batches(_hist)
    _sb = BATCH.since_baseline(_hist) if _hist is not None else None
    if _b.empty or not _sb:
        st.caption("No county has posted a new batch since the first saved count. Batches appear here "
                   "as counties update (election_night_history.csv).")
    else:
        _bt = _fmt_pt(_parse_sos_time(_sb["baseline_at"]))
        _dsn, _dsb = _sb["dem_share_new"], _sb["dem_share_baseline"]
        _lean = ("more Democratic" if _dsn > _dsb else "more Republican") + " than the count before them"
        _net = _sb["net_dem"]
        st.markdown(
            f"Since the first saved count (**{_bt}**), counties have added **{_sb['new_votes']:,}** votes: "
            f"**{_dsn*100:.1f}% Dem**, {_lean} ({_dsb*100:.1f}%). "
            f"Net: **{'D' if _net >= 0 else 'R'} +{abs(_net):,}**."
        )
        def _share_cell(new, before):
            if pd.isna(new):
                return "—"
            c = "#1a6b3c" if new >= before else "#b91c1c"
            return f"<span style='color:{c};font-weight:600'>{new*100:.1f}%</span>"
        brows = ""
        _b["_posted"] = [_parse_sos_time(x) for x in _b["last_updated"]]
        for _, r in _b.sort_values(["_posted", "fetched_at"], ascending=False, na_position="last").iterrows():
            ts = r._posted
            new_tag = "🆕 " if pd.notna(ts) and (_NOW_PT - ts) <= pd.Timedelta(hours=1) else ""
            brows += (
                f"<tr><td>{new_tag}{r.county}</td><td>{_fmt_pt(ts)}</td><td>{int(r.new_votes):,}</td>"
                f"<td>{_share_cell(r.dem_share_new, r.dem_share_before)}</td>"
                f"<td>{r.dem_share_before*100:.1f}%</td><td>{r.dem_share_after*100:.1f}%</td>"
                f"<td style='color:{'#1a6b3c' if r.net_dem >= 0 else '#b91c1c'}'>"
                f"{'D' if r.net_dem >= 0 else 'R'} +{abs(int(r.net_dem)):,}</td></tr>"
            )
        st.markdown(
            '<div style="overflow-x:auto"><table class="styled-table" style="min-width:700px">'
            '<thead><tr><th>County</th><th>Posted</th><th>New Votes</th><th>Dem Share of New</th>'
            '<th>Dem Share Before</th><th>Dem Share After</th><th>Net</th></tr></thead>'
            f'<tbody>{brows}</tbody></table></div>'
            '<p style="font-size:0.72rem;color:#999;margin-top:0.4rem">'
            'Newest first. Dem share is two-party (D ÷ (D + R)). Dem Share of New is green when the batch is more '
            'Democratic than the county\'s count before it, red when less. Batches are recorded whenever the app '
            '(or the email check) fetches a changed count, so one row can cover more than one county report.</p>',
            unsafe_allow_html=True
        )

# ── Needed share over time ────────────────────────────────────────────────
# Replays election_night_history.csv: after each county update, the district's
# two-party Dem share and the Dem share needed of the remaining vote (remaining
# as in County Results — aged SOS figure from its as_of time on, model before).
if not _mock_active and not HIST_PREP.empty:
    _model_tot = {cn: cd.get("registration", 0) * cd.get("turnout", 0) for cn, cd in COUNTIES.items()}
    _nh = REM.need_history(HIST_PREP, UNPROCESSED, SD4_FRACTION, _model_tot)
    try:
        _exp_late = float(_lm["expected_late_share"])
    except (NameError, KeyError, TypeError):
        _exp_late = np.nan

    import plotly.graph_objects as go
    _BLUE, _ORANGE, _INK, _MUTED, _GRID = "#2a78d6", "#eb6834", "#333", "#999", "#eee"
    _hov = _nh.assign(
        when=_nh["posted"].dt.strftime("%a %b %-d, %-I:%M %p"),
        upd=np.where(_nh["first"], "first saved count",
                     _nh["new_votes"].map(lambda v: f"+{int(v):,} votes")),
    )
    _cd = np.stack([_hov["when"], _hov["county"], _hov["upd"], _hov["remaining"]], axis=-1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_nh["posted"], y=_nh["needed"].clip(0, 1), mode="lines", line=dict(color=_ORANGE, width=2, shape="hv"),
        name="Dem share needed of remaining", customdata=_cd,
        hovertemplate="<b>Needed: %{y:.1%}</b> of ~%{customdata[3]:,.0f} remaining<br>"
                      "after %{customdata[1]} (%{customdata[2]})<br>%{customdata[0]}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=_nh["posted"], y=_nh["dem_2p"], mode="lines", line=dict(color=_BLUE, width=2, shape="hv"),
        name="Dem share, all counted", customdata=_cd,
        hovertemplate="<b>Counted: %{y:.1%}</b> Dem<br>after %{customdata[1]} (%{customdata[2]})"
                      "<br>%{customdata[0]}<extra></extra>"))
    _bd = _hov[~_hov["first"] & _hov["batch_dem_2p"].notna() & (_hov["new_votes"] > 0)]
    if not _bd.empty:
        fig.add_trace(go.Scatter(
            x=_bd["posted"], y=_bd["batch_dem_2p"], mode="markers", name="Dem share of each batch (size = votes)",
            marker=dict(color=_BLUE, opacity=0.4, line=dict(color="white", width=1.5),
                        size=8 + 22 * (_bd["new_votes"] / _bd["new_votes"].max()) ** 0.5),
            customdata=np.stack([_bd["when"], _bd["county"], _bd["new_votes"]], axis=-1),
            hovertemplate="<b>%{customdata[1]} batch: %{y:.1%}</b> Dem<br>%{customdata[2]:,} votes"
                          "<br>%{customdata[0]}<extra></extra>"))
    if np.isfinite(_exp_late):
        fig.add_hline(y=_exp_late, line=dict(color=_ORANGE, width=1, dash="dot"),
                      annotation_text=f"Expected now {_exp_late*100:.1f}%", annotation_position="top right",
                      annotation_font=dict(size=10, color=_INK))
    fig.add_hline(y=0.5, line=dict(color=_MUTED, width=1, dash="dash"))
    fig.update_layout(
        height=380, margin=dict(l=60, r=20, t=20, b=10), plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="IBM Plex Sans, sans-serif", size=11, color=_INK),
        hovermode="closest", hoverlabel=dict(bgcolor="white", font_size=11),
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, font=dict(size=11)),
        xaxis=dict(showgrid=False, linecolor="#ccc", tickformat="%a %-I %p"),
        yaxis=dict(title="Two-party Dem share", tickformat=".0%", gridcolor=_GRID, zeroline=False),
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    _last = _nh.iloc[-1]
    st.caption(
        f"Needed now: {_last['needed']*100:.1f}% of ~{_last['remaining']:,.0f} remaining "
        f"(the late-vote model's figure above uses the same remaining votes). Each step is one county "
        f"update, at its SOS posting time; each county's first saved count isn't drawn as a batch. "
        f"Remaining uses a county's SOS figure from its as_of time on and the model before. "
        f"Needed is clipped to 0–100% on the chart. Hover for details; drag to zoom, double-click to reset, "
        f"click a legend item to hide it."
    )

# ── Early vote vs reported results ────────────────────────────────────────
# Compare each county's reported shares with the party registration mix of
# its returned early ballots (last Optiq snapshot). If every registered Dem
# voted Dem and every registered Rep voted Rep, the Dem share would sit
# between its floor (D share of returns) and ceiling (D + Other share).
# Results outside that band need crossover voting; inside it, the implied
# Other→Dem split says how the unaffiliated/minor-party voters broke.
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Early Vote vs Reported Results</div>', unsafe_allow_html=True)

cp_path = DATA_DIR / "optiq_county_party.csv"

@st.cache_data(show_spinner=False)
def load_early_vote_party(p, mtime=0.0):
    """Last snapshot of Optiq county × party returns → {county: {D,R,O}}."""
    try:
        cp = pd.read_csv(p)
    except Exception:
        return None, None
    if cp.empty or "snapshot_date" not in cp:
        return None, None
    snap = cp["snapshot_date"].max()
    cp = cp[(cp["snapshot_date"] == snap) & cp["party"].isin(["D", "R", "O"])]
    cp["returned"] = pd.to_numeric(cp["returned"], errors="coerce").fillna(0)
    piv = cp.pivot_table(index="county", columns="party", values="returned", aggfunc="sum").fillna(0)
    return {cn: {p: float(piv.loc[cn].get(p, 0)) for p in ("D", "R", "O")} for cn in piv.index}, snap

early_party, early_snap = load_early_vote_party(str(cp_path), _mtime(cp_path))

def _ev_row(label, ev, d, r, o, d_loy=1.0, r_loy=1.0, o_dem=0.5, bold=False):
    """One table row comparing reported D/R shares with the early-vote party mix.

    Expected Dem = D·(Dem loyalty) + R·(1 − Rep loyalty) + Other·(O/NPP Dem share).
    """
    n_ev = ev["D"] + ev["R"] + ev["O"]
    tv = d + r + o
    if n_ev <= 0:
        return ""
    fd, fr, fo = ev["D"] / n_ev, ev["R"] / n_ev, ev["O"] / n_ev
    exp_d = fd * d_loy + fr * (1 - r_loy) + fo * o_dem
    cells = [f"{fd*100:.1f}%", f"{fr*100:.1f}%", f"{fo*100:.1f}%"]
    if tv > 0:
        sd, sr = d / tv, r / tv
        vs_d, vs_r = sd - fd, sr - fr
        def _pp(v):
            c = "#1a6b3c" if v >= 0 else "#b91c1c"
            return f"<span style='color:{c}'>{'+' if v >= 0 else '−'}{abs(v)*100:.1f}</span>"
        # Share of Other-registered returns the Dem needs if there's no crossover
        x = (sd - fd) / fo if fo > 0 else np.nan
        if np.isnan(x):
            x_str = "—"
        elif x > 1:
            x_str = f"<span style='color:#1a6b3c;font-weight:600'>&gt;100% · D crossover ≥{(sd-fd-fo)*100:.1f}pt</span>"
        elif x < 0:
            x_str = f"<span style='color:#b91c1c;font-weight:600'>&lt;0% · R crossover ≥{(fd-sd)*100:.1f}pt</span>"
        else:
            x_str = f"{x*100:.0f}% D / {(1-x)*100:.0f}% R"
        ratio = tv / n_ev
        ratio_str = f"{ratio:.2f}×" + (" <span style='color:#d97706'>⚑</span>" if ratio > 1.05 else "")
        gap = sd - exp_d
        exp_str = (f"{exp_d*100:.1f}%<br><span style='font-size:0.68rem;color:"
                   f"{'#1a6b3c' if gap >= 0 else '#b91c1c'}'>Dem {'+' if gap >= 0 else '−'}{abs(gap)*100:.1f}pt</span>")
        cells += [exp_str, f"{sd*100:.1f}%", _pp(vs_d), f"{sr*100:.1f}%", _pp(vs_r), x_str, ratio_str]
    else:
        cells += [f"{exp_d*100:.1f}%"] + ["—"] * 6
    td = "".join(f"<td>{c}</td>" for c in cells)
    name = f"<strong>{label}</strong>" if bold else label
    style = " style='border-top:2px solid #ddd;font-weight:600'" if bold else ""
    return f"<tr{style}><td>{name}</td>{td}</tr>"

if not early_party:
    st.caption("optiq_county_party.csv not found — no early-vote party data to compare.")
else:
    st.markdown(
        "Reported shares set against the party registration of each county's returned early ballots "
        f"(last Optiq snapshot, **{early_snap}**). If every registered Democrat voted Dem and every "
        "Republican voted Rep, the Dem share would fall between its **floor** (D share of returns) and "
        "its ceiling (D + Other). **Implied Other split** is how unaffiliated and minor-party voters "
        "would have to break with no crossover; outside 0–100% means crossover voting is needed to "
        "explain the result, and the figure shown is the minimum net crossover. **Expected Dem** is the "
        "Dem share the sliders below imply; the line beneath it shows how far the reported Dem share is above or below it."
    )
    lc1, lc2, lc3 = st.columns(3)
    ev_d_loy = lc1.slider("Democratic loyalty", 60.0, 100.0, 100.0, 0.5, format="%.1f%%", key="ev_d_loy",
                          help="Share of registered Democrats voting for the Democrat") / 100
    ev_r_loy = lc2.slider("Republican loyalty", 60.0, 100.0, 100.0, 0.5, format="%.1f%%", key="ev_r_loy",
                          help="Share of registered Republicans voting for the Republican") / 100
    ev_o_dem = lc3.slider("O/NPP share for the Democrat", 20.0, 80.0, 50.0, 0.5, format="%.1f%%", key="ev_o_dem",
                          help="Share of Other / No Party Preference voters voting for the Democrat") / 100
    _ev_kw = dict(d_loy=ev_d_loy, r_loy=ev_r_loy, o_dem=ev_o_dem)
    ev_rows = ""; tot_ev = {"D": 0.0, "R": 0.0, "O": 0.0}; tot_d = tot_r = tot_o = 0
    for _, row in df_night.iterrows():
        cn = row.get("county", "")
        if cn not in early_party:
            continue
        d4 = row.get("dem_votes", 0);   d4 = 0 if pd.isna(d4) else int(d4)
        r4 = row.get("rep_votes", 0);   r4 = 0 if pd.isna(r4) else int(r4)
        o4 = row.get("other_votes", 0); o4 = 0 if pd.isna(o4) else int(o4)
        ev_rows += _ev_row(cn, early_party[cn], d4, r4, o4, **_ev_kw)
        for k in tot_ev: tot_ev[k] += early_party[cn][k]
        tot_d += d4; tot_r += r4; tot_o += o4
    ev_rows += _ev_row("SD4", tot_ev, tot_d, tot_r, tot_o, bold=True, **_ev_kw)
    st.markdown(
        '<div style="overflow-x:auto"><table class="styled-table" style="min-width:900px">'
        '<thead><tr><th>County</th><th>Early D</th><th>Early R</th><th>Early Other</th>'
        '<th>Expected Dem</th><th>Dem Share</th><th>Dem vs Floor (pt)</th><th>Rep Share</th><th>Rep vs Floor (pt)</th>'
        '<th>Implied Other Split</th><th>Reported ÷ Early</th></tr></thead>'
        f'<tbody>{ev_rows}</tbody></table></div>'
        '<p style="font-size:0.72rem;color:#999;margin-top:0.4rem">'
        'Early D/R/Other = party registration of returned VBM ballots (Optiq). Floor = that party\'s share of returns. '
        'Expected Dem = Early D × Dem loyalty + Early R × (1 − Rep loyalty) + Early Other × O/NPP Dem share; '
        'Expected Rep is 100% minus this. '
        'Reported ÷ Early = votes counted ÷ early ballots returned; ⚑ above ~1.05× means Election Day votes are '
        'in the count and the comparison weakens (they lean Republican). Most useful for the first drops, '
        'which are largely early ballots.</p>',
        unsafe_allow_html=True
    )

# ── Manual data entry ──────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("✏️ Enter / update county results"):
    st.caption("Edit cells and click **Save results** to write to election_night.csv. "
               "Turn **Auto-refresh** off first, or the next SOS fetch will overwrite manual entries.")
    df_edit = df_night[["county","precincts_reporting","precincts_total","dem_votes","rep_votes","other_votes"]].copy()
    for col in ["dem_votes","rep_votes","other_votes","precincts_reporting"]:
        df_edit[col] = pd.to_numeric(df_edit[col], errors="coerce").fillna(0).astype(int)

    edited_night = st.data_editor(
        df_edit,
        hide_index=True,
        width="stretch",
        disabled=["county"],
        column_config={
            "county":              st.column_config.TextColumn("County", width="medium"),
            "precincts_reporting": st.column_config.NumberColumn("Prec. Reporting", min_value=0, step=1),
            "precincts_total":     st.column_config.NumberColumn("Prec. Total",     min_value=1, step=1),
            "dem_votes":           st.column_config.NumberColumn("Dem Votes",   min_value=0, step=1),
            "rep_votes":           st.column_config.NumberColumn("Rep Votes",   min_value=0, step=1),
            "other_votes":         st.column_config.NumberColumn("Other Votes", min_value=0, step=1),
        },
        key="night_editor"
    )
    if st.button("💾 Save results", key="night_save", disabled=_mock_active,
                 help="Disabled while a mock scenario is loaded" if _mock_active else None):
        # Merge back into the full night csv preserving other columns
        df_out_night = df_night.copy()
        for col in ["precincts_reporting","precincts_total","dem_votes","rep_votes","other_votes"]:
            df_out_night[col] = edited_night[col].values
        df_out_night["last_updated"] = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d %H:%M")
        df_out_night.to_csv(night_path, index=False)
        st.success("Results saved to election_night.csv")
        st.cache_data.clear()
        st.rerun()

# ── Election Night Targets ─────────────────────────────────────────────────
# Show what Dem share to expect per county at each reporting wave,
# conditional on the district being won (filtering to winning simulations).
# Early waves are mail-heavy, so county shares start higher and converge
# toward the full-count final share as more election-day votes come in.
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Election Night Targets</div>', unsafe_allow_html=True)
st.markdown(
    "What to look for county-by-county at each reporting wave, "
    "based on simulations **where the district is won** (≥50%). "
    "Early waves show higher Dem shares because mail ballots (which lean ~+4pp "
    "toward Democrats) are counted first."
)

# ── Reporting speed table ─────────────────────────────────────────────────
# Wave display labels (matching index positions in _SPEED above)
_WAVE_LABELS = ["8 pm", "9 pm", "11 pm", "9 am\n+1 day", "4 pm\n+1 day",
                "4 pm\n+2 days", "4 pm\n+3 days", "4 pm\n+4 days",
                "4 pm\n+5 days", "4 pm\n+6 days", "4 pm\n+7 days"]
_WAVE_LABELS_SHORT = ["8pm", "9pm", "11pm", "D+1\n9am", "D+1\n4pm",
                      "D+2", "D+3", "D+4", "D+5", "D+6", "D+7"]
N_WAVES = len(_WAVE_LABELS)

# Filter to winning simulations
win_mask = district_share >= WIN_THRESHOLD
n_win    = int(win_mask.sum())

# ── Build targets data (shared by both expanders) ────────────────────────
tgt_rows = []
for cn in sorted(county_shares.keys()):
    final_arr = county_shares[cn][win_mask]
    p25 = float(np.percentile(final_arr, 25))
    p50 = float(np.percentile(final_arr, 50))
    p75 = float(np.percentile(final_arr, 75))
    wave_cells = [
        (_wave_share(p25, cn, wi), _wave_share(p50, cn, wi), _wave_share(p75, cn, wi))
        for wi in range(N_WAVES)
    ]
    tgt_rows.append({"county": cn, "p25": p25, "p50": p50, "p75": p75, "waves": wave_cells})

def _fmt_band(lo, mid, hi, f_in):
    """Render a percentile cell; grey out if < 20% in."""
    if f_in < 0.20:
        return f"<span style='color:#ccc;font-size:0.72rem'>n/a<br><span style='font-size:0.60rem'>{f_in*100:.0f}% in</span></span>"
    color = "#1a6b3c" if mid >= 0.50 else ("#d97706" if mid >= 0.48 else "#b91c1c")
    weight = "700" if mid >= 0.50 else "400"
    return (f"<span style='color:{color};font-weight:{weight}'>{mid*100:.1f}%</span>"
            f"<br><span style='font-size:0.62rem;color:#777'>{lo*100:.1f}–{hi*100:.1f}%</span>")

# ── County targets table expander ────────────────────────────────────────
with st.expander(f"📊 County targets table ({n_win:,} winning simulations of {int(n_sims):,})", expanded=True):
    if n_win < 100:
        st.warning(f"Only {n_win} winning simulations — model gives low win probability. "
                   "Targets may be unreliable; consider adjusting the forecast environment.")
    else:
        st.markdown(
            "**How to read this table:** Each cell shows the Dem vote share you'd expect "
            "to see *on the results page* at that checkpoint — not the final share. "
            "Early numbers are inflated because pre-processed mail ballots (which lean "
            "~4 points more Democratic) are counted first. As election-day ballots are "
            "added over the following days, the share drifts down toward the **Final** column. "
            "Numbers are from simulations where the district is won; the range (small–**median**–large) "
            "covers the middle 50% of those outcomes."
        )
        st.caption(
            "Color guide: 🟢 **bold green** = median above 50% (county-level lead) · "
            "🟡 amber = 48–50%, marginal · 🔴 red = below 48% "
            "(trailing in this county but still winning the district overall in these sims). "
            "Grey 'n/a' = fewer than 20% of votes in — number not yet meaningful. "
            "Small grey '% in' beneath each cell shows the expected share of votes reported at that checkpoint."
        )
        st.markdown("")

        hdr = (
            "<thead>"
            "<tr>"
            "<th rowspan='2' style='vertical-align:bottom'>County</th>"
            "<th rowspan='2' style='vertical-align:bottom'>Final<br><span style='font-size:0.62rem;font-weight:400'>(true target)</span></th>"
            "<th colspan='3' style='text-align:center;border-bottom:1px solid #ddd'>Election night</th>"
            "<th colspan='2' style='text-align:center;border-bottom:1px solid #ddd'>Next day</th>"
            "<th colspan='6' style='text-align:center;border-bottom:1px solid #ddd'>Days 2–7 (4 pm each day)</th>"
            "</tr>"
            "<tr>"
            + "".join(f"<th>{lbl}</th>" for lbl in _WAVE_LABELS_SHORT)
            + "</tr></thead>"
        )
        def _fmt_band_with_speed(lo, mid, hi, f_in):
            """Like _fmt_band but appends a '% in' sub-line."""
            if f_in < 0.20:
                return f"<span style='color:#ccc;font-size:0.72rem'>n/a<br><span style='font-size:0.60rem'>{f_in*100:.0f}% in</span></span>"
            color = "#1a6b3c" if mid >= 0.50 else ("#d97706" if mid >= 0.48 else "#b91c1c")
            weight = "700" if mid >= 0.50 else "400"
            return (f"<span style='color:{color};font-weight:{weight}'>{mid*100:.1f}%</span>"
                    f"<br><span style='font-size:0.62rem;color:#777'>{lo*100:.1f}–{hi*100:.1f}%</span>"
                    f"<br><span style='font-size:0.60rem;color:#aaa'>{f_in*100:.0f}% in</span>")

        tbody = "<tbody>"
        for r in tgt_rows:
            cn         = r["county"]
            speed_list = _SPEED.get(cn, [0.5]*N_WAVES)
            tbody += f"<tr><td><strong>{cn}</strong></td>"
            tbody += f"<td>{_fmt_band(r['p25'], r['p50'], r['p75'], 1.0)}</td>"
            for wi, (w25, w50, w75) in enumerate(r["waves"]):
                tbody += f"<td>{_fmt_band_with_speed(w25, w50, w75, speed_list[wi])}</td>"
            tbody += "</tr>"
        tbody += "</tbody>"

        st.markdown(
            f'<div style="overflow-x:auto"><table class="styled-table" style="font-size:0.78rem;min-width:900px">'
            f'{hdr}{tbody}</table></div>',
            unsafe_allow_html=True
        )
        st.caption(
            "Reporting speeds: empirically derived from CalVoter Foundation 2024 General Election data "
            "for Alpine, El Dorado, Inyo, Madera, Merced, Mono, Nevada, Placer, Stanislaus. "
            "Amador, Calaveras, Mariposa, Tuolumne are interpolated from similar counties."
        )

# ── Key counties to watch expander ───────────────────────────────────────
with st.expander("🔍 Key counties to watch", expanded=True):
    st.markdown(
        "A county is worth watching closely if it reports a large share of its votes "
        "early *and* its result is competitive enough to matter. "
        "Fast counties that lean heavily one way tell you less — you already know what they'll do."
    )
    watch_notes = []
    for r in tgt_rows:
        cn        = r["county"]
        p50_final = r["p50"]
        speed0    = _SPEED.get(cn, [0.5]*N_WAVES)[0]
        if speed0 >= 0.35 and 0.40 <= p50_final <= 0.62:
            w8_lo, w8_mid, w8_hi = r["waves"][0]
            watch_notes.append((cn, speed0, p50_final, w8_mid, w8_lo, w8_hi))

    if watch_notes:
        # Build live dem-share lookup from df_night
        live_share = {}
        for _, _nr in df_night.iterrows():
            _d = _nr.get("dem_votes", 0);  _d = 0 if pd.isna(_d) else int(_d)
            _r = _nr.get("rep_votes", 0);  _r = 0 if pd.isna(_r) else int(_r)
            _o = _nr.get("other_votes", 0); _o = 0 if pd.isna(_o) else int(_o)
            _tv = _d + _r + _o
            if _tv > 0:
                live_share[_nr.get("county", "")] = int(_d) / _tv

        for cn, speed0, p50_final, w8_mid, w8_lo, w8_hi in watch_notes:
            actual = live_share.get(cn)
            if actual is None:
                bg, border = "#fafafa", "#e2e8f0"
            elif actual >= w8_hi:
                bg, border = "#f0fdf4", "#86efac"   # light green
            elif actual >= w8_lo:
                bg, border = "#fefce8", "#fde047"   # light yellow
            else:
                bg, border = "#fff1f2", "#fca5a5"   # light red

            st.markdown(
                f'<div style="border:1px solid {border};border-radius:8px;padding:0.9rem 1.1rem;margin-bottom:0.75rem;background:{bg}">'
                f'<div style="font-size:1rem;font-weight:700;margin-bottom:0.35rem">'
                f'{cn} <span style="font-weight:400;color:#555">— approx {speed0*100:.0f}% of votes in by 8pm</span></div>'
                f'<div style="margin-bottom:0.4rem">In winning simulations, the 8pm tally generally falls between '
                f'<strong>{w8_lo*100:.1f}%</strong> and <strong>{w8_hi*100:.1f}%</strong> (typical: {w8_mid*100:.1f}%).</div>'
                f'<div style="font-size:0.85rem;color:#444">'
                f'🟢 Above {w8_hi*100:.0f}% — strong positive sign &nbsp;·&nbsp; '
                f'🟡 {w8_lo*100:.0f}%–{w8_hi*100:.0f}% — within expected range &nbsp;·&nbsp; '
                f'🔴 Below {w8_lo*100:.0f}% — warning sign</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.caption("No county provides a strong early signal at the current forecast — check the district total.")

# ── Governor race panel ───────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Governor Race — Statewide Environment Check</div>', unsafe_allow_html=True)
st.markdown(
    "Statewide governor results provide a real-time read on the political environment. "
    "A governor Dem share above or below the model's predicted environment suggests "
    "conditions may differ from forecast."
)

# Statewide governor totals: manual entry (session) wins; otherwise sum the
# county rows the SOS fetcher writes to governor_night.csv.
if "gov_dem_state" in st.session_state:
    gov_dem_state   = int(st.session_state.get("gov_dem_state",   0))
    gov_rep_state   = int(st.session_state.get("gov_rep_state",   0))
    gov_other_state = int(st.session_state.get("gov_other_state", 0))
    gov_updated     = st.session_state.get("gov_updated", "—") + " (manual)"
else:
    _gov = None if _mock_active else load_night_csv(str(gov_path), _mtime(gov_path))
    if _gov is not None and not _gov.empty:
        _gs = _gov[["dem_votes", "rep_votes", "other_votes"]].apply(pd.to_numeric, errors="coerce").fillna(0).sum()
        gov_dem_state, gov_rep_state, gov_other_state = int(_gs["dem_votes"]), int(_gs["rep_votes"]), int(_gs["other_votes"])
        _gl = _gov["last_updated"].dropna().astype(str) if "last_updated" in _gov else pd.Series(dtype=str)
        gov_updated = (_gl.max()[:16].replace("T", " ") + " (SOS)") if not _gl.empty else "—"
    else:
        gov_dem_state = gov_rep_state = gov_other_state = 0
        gov_updated = "—"

gov_total = gov_dem_state + gov_rep_state + gov_other_state

gv1, gv2, gv3, gv4 = st.columns(4)
with gv1:
    gs = f"{gov_dem_state/gov_total*100:.1f}%" if gov_total > 0 else "—"
    color_gs = ("#1a6b3c" if gov_total > 0 and gov_dem_state/gov_total >= 0.50 else "#b91c1c") if gov_total > 0 else "#888"
    st.markdown(
        f'<div class="stat-card" style="border-color:{color_gs}">'
        f'<div class="label">Gov Dem Share</div>'
        f'<div class="value" style="color:{color_gs}">{gs}</div>'
        f'<div class="sub">Statewide · {gov_dem_state:,} D / {gov_rep_state:,} R</div>'
        f'</div>', unsafe_allow_html=True
    )
with gv2:
    if gov_total > 0:
        g_share   = gov_dem_state / gov_total
        blend_wt  = pct_precincts
        gov_blend = blend_wt * g_share + (1 - blend_wt) * fp_env
        st.markdown(
            f'<div class="stat-card"><div class="label">Gov → Adjusted Env</div>'
            f'<div class="value">{gov_blend*100:.1f}%</div>'
            f'<div class="sub">{blend_wt*100:.0f}% weight on live results</div>'
            f'</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="stat-card"><div class="label">Gov → Adjusted Env</div>'
            f'<div class="value">—</div><div class="sub">No results yet</div></div>',
            unsafe_allow_html=True
        )
with gv3:
    st.markdown(
        f'<div class="stat-card"><div class="label">Model Forecast Env</div>'
        f'<div class="value">{fp_env*100:.1f}%</div></div>',
        unsafe_allow_html=True
    )
with gv4:
    if gov_total > 0:
        g_share = gov_dem_state / gov_total
        diff_g  = g_share - fp_env
        c_g     = "#1a6b3c" if diff_g >= 0 else "#b91c1c"
        st.markdown(
            f'<div class="stat-card"><div class="label">Gov vs Forecast</div>'
            f'<div class="value" style="color:{c_g}">{"▲" if diff_g>=0 else "▼"} {abs(diff_g)*100:.1f}pt</div>'
            f'<div class="sub">Updated: {gov_updated}</div></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="stat-card"><div class="label">Gov vs Forecast</div>'
            f'<div class="value">—</div></div>',
            unsafe_allow_html=True
        )

with st.expander("✏️ Enter statewide governor totals"):
    st.caption(
        "Enter the statewide running totals for the governor race "
        "(Dem and Rep vote counts). These are used to estimate the live environment."
    )
    ge1, ge2, ge3 = st.columns(3)
    with ge1:
        new_gov_dem   = st.number_input("Dem votes (statewide)", min_value=0, value=gov_dem_state,   step=1000, key="gov_dem_input")
    with ge2:
        new_gov_rep   = st.number_input("Rep votes (statewide)", min_value=0, value=gov_rep_state,   step=1000, key="gov_rep_input")
    with ge3:
        new_gov_other = st.number_input("Other votes",           min_value=0, value=gov_other_state, step=1000, key="gov_other_input")

    if st.button("💾 Save governor totals", key="gov_save"):
        st.session_state["gov_dem_state"]   = new_gov_dem
        st.session_state["gov_rep_state"]   = new_gov_rep
        st.session_state["gov_other_state"] = new_gov_other
        st.session_state["gov_updated"]     = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%H:%M")
        st.success("Governor totals updated.")
        st.rerun()
    if "gov_dem_state" in st.session_state and st.button("↩ Use SOS totals instead", key="gov_reset"):
        for _k in ("gov_dem_state", "gov_rep_state", "gov_other_state", "gov_updated"):
            st.session_state.pop(_k, None)
        st.rerun()

# ── Auto-refresh logic ────────────────────────────────────────────────────
# A fragment ticks every minute without blocking the page (the forecast app
# slept for 10 minutes at the end of the script). When the last fetch — by any
# viewer — is 10+ minutes old it pulls fresh SOS results and reruns the page.
if auto_refresh and not _mock_active:
    st.caption("⏱ Auto-refreshing every 10 minutes — fetching live SOS results. Toggle off to stop.")

    @st.fragment(run_every=60)
    def _auto_refresh_tick():
        if _seconds_since_fetch() >= _FETCH_EVERY_S:
            _run_fetcher()
            st.rerun()
        elif _mtime(night_path) > st.session_state.get("_night_seen_mtime", 0.0):
            st.rerun()   # another viewer's fetch updated the CSV

    st.session_state["_night_seen_mtime"] = _mtime(night_path)
    _auto_refresh_tick()
elif auto_refresh and _mock_active:
    st.caption("⏱ Auto-refresh paused while a mock scenario is selected.")
