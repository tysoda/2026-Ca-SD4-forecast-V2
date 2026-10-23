"""
SD4 Ballot Return Tracker — standalone Streamlit app.

Run locally:   streamlit run ballot_tracker/app.py
Streamlit Cloud: same repo, main file path = ballot_tracker/app.py

Reads the CSVs the GitHub Actions already refresh at the repo root; no
dependency on the forecast model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import importlib

import analysis as A
import data as D

# Streamlit keeps imported helper modules between reruns, so after a git pull the
# app can run new app.py code against old data.py / analysis.py. Reload them.
D = importlib.reload(D)
A = importlib.reload(A)

sys.path.insert(0, str(D.REPO_DIR))
from vbm_turnout import COUNTY_PARAMS  # noqa: E402

st.set_page_config(page_title="SD4 Ballot Returns", page_icon="📬", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  h1, h2, h3, h4 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.02em; }
  .block-container { padding-top: 2rem; padding-bottom: 3rem; }
  .section-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase; color: #888;
    border-bottom: 1px solid #e0e0e0; padding-bottom: 0.4rem; margin: 1.4rem 0 0.8rem 0;
  }
  .stat-card { background: #f7f7f5; border-left: 3px solid #1a6b3c; padding: 0.85rem 1rem; border-radius: 2px; margin-bottom: 0.5rem; }
  .stat-card .label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #666; margin-bottom: 0.2rem; }
  .stat-card .value { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 600; color: #111; line-height: 1; }
  .stat-card .sub { font-size: 0.72rem; color: #888; margin-top: 0.25rem; }
  .src-badge { display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
    padding: 0.15rem 0.45rem; border-radius: 2px; letter-spacing: 0.05em; margin-right: 0.3rem;
    background: #e8e8e4; color: #444; }
  .src-badge.live { background: #1a6b3c; color: white; }
</style>
""", unsafe_allow_html=True)

SOURCE_GAP_NOTE = (
    "They're measured differently: (1) they update at different times, so one is often a day ahead; "
    "(2) for the four split counties (Madera, Merced, Nevada, Placer) the SOS figures are estimates — the "
    "county total scaled by SD4's share of registration — while Optiq places each voter in or out of the "
    "district; (3) Optiq only counts ballots it can match to a voter-file record (~98%). Compare each "
    "source with itself over time rather than one against the other."
)

def DROP_COLS(unit: str) -> dict:
    return {
        "last_drop": st.column_config.NumberColumn(
            "Last drop", format="localized",
            help=f"Ballots added in the county's most recent {unit} that showed an increase."),
        "last_drop_date": st.column_config.DateColumn(
            "Drop date", format="MMM D",
            help="Optiq: the county's most recent ballot return date. SOS: the report the last drop appeared in."),
    }


PLOT_LAYOUT = dict(
    plot_bgcolor="#f7f7f5", paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis=dict(gridcolor="#e0e0e0", title=None), yaxis=dict(gridcolor="#e0e0e0", title=None),
)
SEQ_COLORS = ["#1a6b3c", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#65a30d", "#475569"]
# One fixed colour per county so it's the same on every chart
from plotly.colors import qualitative as _q
COUNTY_COLORS = dict(zip(D.SD4_COUNTIES, (_q.Dark24[:6] + _q.Safe)[:13])) | {"SD4 total": "#111111"}


# ── Small UI helpers ─────────────────────────────────────────────────────────
def section(title: str, caption: str | None = None):
    st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def card(label, value, sub="", color=None):
    border = f' style="border-left-color:{color}"' if color else ""
    lab = f' style="color:{color}"' if color else ""
    st.markdown(
        f'<div class="stat-card"{border}><div class="label"{lab}>{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def pct(v, d=1):
    return "—" if v is None or not np.isfinite(v) else f"{v * 100:.{d}f}%"


def pp(v, d=1):
    return "—" if v is None or not np.isfinite(v) else f"{v * 100:+.{d}f}pp"


def pct_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Scale rate columns to 0–100 for display with a '%' format."""
    df = df.copy()
    for c in cols:
        df[c] = df[c] * 100
    return df


def pct_col(label, signed=False):
    return st.column_config.NumberColumn(label, format=("%+.1f pp" if signed else "%.1f%%"))


INT = lambda label: st.column_config.NumberColumn(label, format="localized")


# ── Table styling ────────────────────────────────────────────────────────────
ZEBRA_BG = "#f3f3ef"
TOTAL_BG = "#e3e9f2"
PACE_COLORS = {"Above": "#1a6b3c", "Near": "#666666", "Below": "#b26a00"}


def _fmt_value(v, fmt: str | None, kind: str) -> str:
    """Format one value the way the column's NumberColumn/DateColumn format would."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return "—"
    if kind == "date":
        return f"{pd.Timestamp(v):%b %-d}"
    if fmt in (None, "localized"):
        return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.1f}"
    try:
        return fmt % v
    except (TypeError, ValueError):
        return str(v)


def show_df(df: pd.DataFrame, column_config: dict | None = None, signed: list[str] | None = None,
            pace: str | None = None, **kwargs):
    """st.dataframe with the app's table styling.

    - alternating row shading; the "SD4 total" / "Total" row bold on a blue-grey band
    - `signed` columns (D−R margins and shifts only) tinted blue (+, Democratic) / red (−,
      Republican), stronger for bigger values (scaled within the column, total row excluded)
    - optional pace column coloured Above / Near / Below
    - first column pinned so it stays visible when the table scrolls sideways
    - empty cells show "—" (Streamlit would print "None"): columns with gaps are shown as
      formatted text, the rest stay numeric so sorting works
    """
    df = df.reset_index(drop=True).copy()
    cfg = dict(column_config or {})
    raw = df.copy()
    first = df.columns[0]
    is_total = df[first].astype(str).str.rstrip("*").isin(["SD4 total", "Total", "All"])

    def _conf(col):
        c = cfg.get(col)
        return c if isinstance(c, dict) else None

    signed_cols = set(signed or [])
    for col in df.columns:
        c = _conf(col)
        fmt = (c or {}).get("type_config", {}).get("format") if c else None
        kind = (c or {}).get("type_config", {}).get("type") if c else None
        numeric = pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col])
        if numeric and df[col].isna().any() and col != first:
            df[col] = [_fmt_value(v, fmt, "date" if kind == "date" or
                                  pd.api.types.is_datetime64_any_dtype(raw[col]) else "number")
                       for v in raw[col]]
            label = (c or {}).get("label") or (cfg.get(col) if isinstance(cfg.get(col), str) else col)
            cfg[col] = st.column_config.Column(label, help=(c or {}).get("help"), alignment="right")
        elif (c or {}).get("type_config", {}).get("type") == "text" and df[col].isna().any():
            df[col] = df[col].fillna("—")
    if first in cfg and isinstance(cfg[first], dict):
        cfg[first] = {**cfg[first], "pinned": True}
    else:
        cfg[first] = st.column_config.Column(cfg.get(first, first) if isinstance(cfg.get(first), str)
                                             else first, pinned=True)

    def _row_styles(row):
        i = row.name
        if is_total.iloc[i]:
            return [f"background-color:{TOTAL_BG};font-weight:700"] * len(row)
        return [f"background-color:{ZEBRA_BG}" if i % 2 else ""] * len(row)

    styler = df.style.apply(_row_styles, axis=1)

    for col in signed_cols & set(df.columns):
        vals = pd.to_numeric(raw[col], errors="coerce")
        scale = vals[~is_total].abs().max()
        scale = scale if scale and np.isfinite(scale) else 1.0

        def _tint(_, vals=vals, scale=scale):
            out = []
            for i, v in enumerate(vals):
                if pd.isna(v) or v == 0:
                    out.append("")
                    continue
                a = 0.06 + 0.30 * min(abs(v) / scale, 1.0)
                rgb, txt = ((21, 101, 192), "#0d47a1") if v > 0 else ((183, 28, 28), "#8e1b1b")
                w = ";font-weight:700" if is_total.iloc[i] else ""
                out.append(f"background-color:rgba({rgb[0]},{rgb[1]},{rgb[2]},{a:.2f});color:{txt}{w}")
            return out
        styler = styler.apply(_tint, subset=[col], axis=0)

    if pace and pace in df.columns:
        styler = styler.map(lambda v: f"color:{PACE_COLORS[v]};font-weight:600" if v in PACE_COLORS else "",
                            subset=[pace])
    kwargs.setdefault("hide_index", True)
    kwargs.setdefault("width", "stretch")
    return st.dataframe(styler, column_config=cfg, **kwargs)


def dim_color(dim: str, code: str, i: int) -> str:
    if dim == "party":
        return D.PARTY_COLORS.get(code, "#666")
    return SEQ_COLORS[i % len(SEQ_COLORS)]


# ── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def load_all():
    return {
        "sos": D.load_sos_returns(drop_stale=False),
        "base": D.load_2022_baseline(),
        "daily22": D.load_2022_daily(),
        "pdi22": D.load_pdi_2022(),
        "reg22": D.load_registration_2022(),
        "reg": D.load_party_registration(),
        "ct": D.load_optiq_crosstabs(),
        "cp": D.load_optiq_county_party(),
        "pre": D.load_optiq_precincts(),
        "saved_at": D.load_last_saved(),
        "drops": D.load_optiq_drops(),
        "pre_cov": D.load_precinct_coverage(),
    }


@st.cache_data(ttl=900, show_spinner=False)
def load_precinct_geo():
    return D.load_precinct_shapes(), D.load_sd4_boundary()


@st.cache_data(ttl=3600, show_spinner="Fetching live Optiq data…")
def load_live():
    fetched = pd.Timestamp.now(tz="America/Los_Angeles").tz_localize(None)
    cells, optiq_loaded = D.fetch_optiq_live()
    return cells, D.fetch_optiq_county_party_live(), fetched, optiq_loaded


@st.cache_data(ttl=1800, show_spinner="Checking SOS for a new report…")
def load_sos_live():
    checked = pd.Timestamp.now(tz="America/Los_Angeles").tz_localize(None)
    return D.fetch_sos_live(), checked


@st.cache_data(ttl=86400, show_spinner=False)
def load_county_geojson():
    import json
    import urllib.request
    urls = [
        "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/california-counties.geojson",
        "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/california-counties.geojson",
    ]
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                gj = json.loads(r.read())
        except Exception:
            continue
        feats = []
        for f in gj.get("features", []):
            name = (f.get("properties", {}).get("name") or f.get("properties", {}).get("NAME") or "")
            name = name.replace(" County", "").strip()
            if name in D.SD4_COUNTIES:
                f["properties"] = {"name": name}
                feats.append(f)
        if len(feats) >= 10:
            return {"type": "FeatureCollection", "features": feats}
    return None


def county_map(values: dict, title: str, diverging: bool, fmt: str, colorscale=None):
    gj = load_county_geojson()
    if gj is None:
        st.info("County boundaries couldn't be loaded — showing the table only.")
        return
    names = list(values.keys())
    z = [values[n] for n in names]
    finite = [v for v in z if np.isfinite(v)]
    if not finite:
        st.info("No values to map yet.")
        return
    if diverging:
        m = max(abs(min(finite)), abs(max(finite))) or 1
        zmin, zmax, cs = -m, m, colorscale or "RdBu"
    else:
        zmin, zmax, cs = min(finite), max(finite), colorscale or "Greens"
    text = [("n/a" if not np.isfinite(v) else fmt.format(v)) for v in z]
    # Drawn as filled shapes on plain axes: Plotly's geo layer downloads base-map
    # files from a CDN at render time, which fails behind some networks.
    from plotly.colors import sample_colorscale
    geoms = {f["properties"]["name"]: f["geometry"] for f in gj["features"]}
    fig = go.Figure()
    lat_mid = 37.5
    for n, v, t in zip(names, z, text):
        g = geoms.get(n)
        if g is None:
            continue
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        frac = 0.5 if not np.isfinite(v) or zmax == zmin else (v - zmin) / (zmax - zmin)
        color = "#dddddd" if not np.isfinite(v) else sample_colorscale(cs, [float(np.clip(frac, 0, 1))])[0]
        xs, ys = [], []
        for poly in polys:
            ring = poly[0]
            xs += [p[0] for p in ring] + [None]
            ys += [p[1] for p in ring] + [None]
        fig.add_scatter(x=xs, y=ys, fill="toself", fillcolor=color, mode="lines",
                        line=dict(color="#222", width=0.7), name=n, showlegend=False,
                        hoveron="fills", text=f"<b>{n}</b><br>{t}", hoverinfo="text")
        # Label at the ring's mean point (largest ring)
        big = max((p[0] for p in polys), key=len)
        cx, cy = np.mean([p[0] for p in big]), np.mean([p[1] for p in big])
        fig.add_annotation(x=cx, y=cy, text=f"{n}<br>{t}", showarrow=False,
                           font=dict(size=9, color="#111"), bgcolor="rgba(255,255,255,0.7)")
    # Invisible trace that only carries the colorbar
    fig.add_scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                    marker=dict(colorscale=cs, cmin=zmin, cmax=zmax, color=[zmin],
                                colorbar=dict(thickness=10, len=0.7,
                                              tickformat=(("+" if diverging else "") + ".0%") if "%" in fmt else ",.0f")))
    fig.update_xaxes(visible=False)
    # Rough equal-area aspect at SD4's latitude
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1 / np.cos(np.radians(lat_mid)))
    fig.update_layout(title=dict(text=title, font=dict(size=12)), height=520,
                      margin=dict(l=0, r=0, t=36, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", hovermode="closest")
    st.plotly_chart(fig, width="stretch")


# ── Load data ────────────────────────────────────────────────────────────────
data = load_all()
sos, base, daily22, reg, ct_saved, cp_saved = (
    data["sos"], data["base"], data["daily22"], data["reg"], data["ct"], data["cp"])
pdi22, reg22 = data["pdi22"], data["reg22"]
curve22 = A.pdi_curve(pdi22)

with st.sidebar:
    st.markdown("### 📬 SD4 Ballot Returns")
    use_live = st.toggle("Use live Optiq data", value=True,
                         help="Pull the Optiq API directly (cached 1 h). Falls back to the saved "
                              "snapshots if the API can't be reached.")
    st.markdown("**2022 pace curve**")
    curve_opts = (["2022 daily returns (PDI)"] if curve22 is not None else []) + ["Assumed shape (k)"]
    curve_mode = st.radio("Path to 2022's end-point", curve_opts, disabled=daily22 is not None,
                          help="2022 daily returns: the real 2022 SD4 return curve, read from PDI's tracker. "
                               "Assumed shape: (days elapsed / window) ^ k, for sensitivity checks.")
    use_k = curve_mode == "Assumed shape (k)"
    curve_k = st.slider(
        "Shape (k)", 1.0, 3.0, 1.0, 0.1,
        disabled=(daily22 is not None) or not use_k,
        help="1 = linear. Higher = more back-loaded: fewer ballots early, more late. "
             "The 2022 PDI curve corresponds to roughly k ≈ 1.5–2.",
    )
    if daily22 is not None:
        st.caption("Using vbm_2022_daily.csv (county-level) — curve choice not needed.")
    curve = None if use_k else curve22
    if st.button("↻ Refresh data"):
        st.cache_data.clear()
        st.rerun()

live_cells, live_cp, optiq_fetched, optiq_loaded = (pd.DataFrame(), pd.DataFrame(), None, None)
if use_live:
    live_cells, live_cp, optiq_fetched, optiq_loaded = load_live()

# California date (the server runs on UTC, which would roll over at 5pm Pacific)
today = pd.Timestamp.now(tz="America/Los_Angeles").tz_localize(None).normalize()
days_to_eday = int((D.EDAY_2026 - today).days)
if not live_cells.empty:
    ct_latest = D.live_to_snapshot(live_cells, today)
    ct_source = "live"
else:
    ct_latest = A.latest(ct_saved)
    ct_source = "saved"
# Time series always use saved snapshots, plus today's live pull if it's newer
# A live pull replaces any saved snapshot from the same day (it's at least as fresh)
ct_hist = ct_saved
if ct_source == "live":
    ct_hist = pd.concat([ct_saved[ct_saved["snapshot_date"] != today] if not ct_saved.empty else ct_saved,
                         ct_latest], ignore_index=True)

if not live_cp.empty:
    cp_latest, cp_source = live_cp.assign(snapshot_date=today), "live"
else:
    cp_latest, cp_source = A.latest(cp_saved), "saved"
cp_hist = cp_saved
if cp_source == "live":
    cp_hist = pd.concat([cp_saved[cp_saved["snapshot_date"] != today] if not cp_saved.empty else cp_saved,
                         cp_latest], ignore_index=True)

reg_counts = reg.get("counts", {})
reg_total = sum(reg_counts.values())

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 📬 SD4 Ballot Return Tracker")
# SOS: saved reports plus, if SOS has posted a newer one than the Action has saved, that report
sos_source = "saved"
sos_checked = None
if use_live and not sos.empty:
    _sl_live, sos_checked = load_sos_live()
    if not _sl_live.empty and _sl_live["snapshot_date"].max() > sos["snapshot_date"].max():
        sos = pd.concat([sos, _sl_live[[c for c in sos.columns if c in _sl_live]]], ignore_index=True)
        sos_source = "live"
sos_latest_report = sos["snapshot_date"].max() if not sos.empty else None
# Repeated reports (weekends: counties don't report, so SOS re-posts the same numbers) are dropped
sos = D._drop_stale_snapshots(sos, "snapshot_date", ["vbm_issued", "vbm_returned"], ["county"]) \
    .sort_values(["snapshot_date", "county"]).reset_index(drop=True) if not sos.empty else sos
if sos.empty:
    st.warning("No SOS ballot return data found (ballot_returns.csv).")
    st.stop()

# All 2022 comparisons are made at today's E-number
county_t, meta = A.county_table(sos, base, daily22, curve_k, curve, compare_date=today)
# Optiq county feed is the primary source for district figures; SOS is the fallback
use_optiq = not cp_hist.empty
if use_optiq:
    oct_, ometa = A.optiq_county_table(cp_hist, base, daily22, curve_k, curve, sos=sos, compare_date=today)
else:
    oct_, ometa = None, None
# Label every pace comparison with the assumption behind it
if daily22 is not None:
    pace_name, pace_short = "2022 county curve", "county curve"
elif curve is not None:
    pace_name, pace_short = "2022 curve", "2022 curve"
elif curve_k == 1:
    pace_name, pace_short = "linear path", "linear"
else:
    pace_name, pace_short = f"curve k={curve_k:g}", f"k={curve_k:g}"
ct_date = (ct_latest["snapshot_date"].max() if not ct_latest.empty else None)
badges = (
    f'<span class="src-badge">E−{days_to_eday} · {days_to_eday} days to Nov 3</span>'
    + (f'<span class="src-badge {"live" if cp_source == "live" else ""}">'
       f'Optiq · {"live, " if cp_source == "live" else ""}fetched {ometa["snapshot"]:%b %-d}</span>'
       if use_optiq else "")
    + (f'<span class="src-badge {"live" if sos_source == "live" else ""}">SOS report · '
       f'{sos_latest_report:%b %-d}{" · live" if sos_source == "live" else ""}'
       + (f' · no change since {meta["snapshot"]:%b %-d}' if sos_latest_report > meta["snapshot"] else "")
       + '</span>')
)
st.markdown(badges, unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Counties", "Precincts", "Crosstabs", "Trends", "Outstanding",
                "2022 comparison", "Method & data"])

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    # ── When the app last checked each source ────────────────────────────────
    def _when(ts):
        return (f"{pd.Timestamp(ts):%a %b %-d, %-I:%M %p}".replace("AM", "am").replace("PM", "pm")
                if ts is not None else "—")

    # Latest check of each source, whether by the app's live pull or a GitHub Action save
    _sv = data.get("saved_at", {})
    def _latest(*ts):
        ts = [t for t in ts if t is not None and pd.notna(t)]
        return max(ts) if ts else None
    _sos_chk = _latest(sos_checked if use_live else None, _sv.get("sos"))
    _opt_chk = _latest(optiq_fetched if (use_live and cp_source == "live") else None, _sv.get("optiq"))
    st.markdown(
        f'<div style="font-size:0.85rem;color:#555;margin:0.2rem 0 0.4rem 0">'
        f'<b>Last checked</b> &nbsp;·&nbsp; SOS {_when(_sos_chk)} &nbsp;·&nbsp; Optiq {_when(_opt_chk)}</div>',
        unsafe_allow_html=True)

    if use_optiq:
        dm = ometa
        section("District",
                f"Ballots returned is the official SOS count ({meta['snapshot']:%b %-d} report). Rates and "
                f"comparisons use Optiq (fetched {dm['snapshot']:%b %-d}). 2022 is taken at today's point, "
                f"E−{dm['days_left']}.")
        c = st.columns(4)
        with c[0]:
            card("Ballots returned · SOS", f"{meta['returned']:,}",
                 f"of {meta['issued']:,} issued ({meta['snapshot']:%b %-d}) · Optiq: {dm['returned']:,}")
        with c[1]:
            card("Return rate · Optiq", pct(dm["rate"]),
                 f"2022 last pre-election report: {pct(meta['rate_2022_baseline'])}")
        with c[2]:
            lrd = oct_["latest_return_date"].max() if "latest_return_date" in oct_ else pd.NaT
            card("Latest return logged", f"{lrd:%b %-d}" if pd.notna(lrd) else "—",
                 f"{oct_['latest_return_date'].eq(lrd).sum()} of 13 counties reported that day" if pd.notna(lrd) else "")
    else:
        dm = meta
        section("District — SOS ballot status report (Optiq unavailable)")
        c = st.columns(4)
        with c[0]:
            card("Ballots returned", f"{meta['returned']:,}", f"of {meta['issued']:,} issued")
        with c[1]:
            card("Return rate", pct(meta["rate"]), f"2022 last pre-election report: {pct(meta['rate_2022_baseline'])}")
        with c[2]:
            card("Returns / day", f"{meta['per_day']:,.0f}" if np.isfinite(meta["per_day"]) else "—",
                 "over the latest reporting interval")
    with c[3]:
        r = dm["pace_ratio"]
        col = "#2e7d32" if r >= 1.10 else ("#d97706" if r >= 0.85 else "#b91c1c")
        card(f"vs 2022 · {pace_name}" + (" · Optiq" if use_optiq else " · SOS"), f"{r:.0%}" if np.isfinite(r) else "—",
             f"2022 at E−{dm['days_left']}: {pct(dm['rate_2022_equiv'])} · 2026 now: {pct(dm['rate'])}",
             color=col)
    if daily22 is None:
        if use_optiq:
            parts = " · ".join(
                f"k={k_:g}: {A.optiq_county_table(cp_hist, base, None, k_, None, sos=sos, compare_date=today)[1]['pace_ratio']:.0%}"
                for k_ in (1.0, 1.5, 2.0, 2.5))
        else:
            sens = A.pace_sensitivity(sos, base, compare_date=today)
            parts = " · ".join(f"k={r.k:g}: {r.pace_ratio:.0%}" for r in sens.itertuples())
        if curve is not None:
            st.caption(f"The 2022 path uses SD4's real 2022 daily returns (PDI, read from their chart), aligned by "
                       f"days before Election Day — both elections fall on a Tuesday, so weekdays line up. One "
                       f"district curve is applied to every county. For comparison, assumed shapes give: {parts}.")
        else:
            st.caption(f"⚠️ The 2022 path to its pre-election end-point is assumed here, not observed. "
                       f"District pace under different shapes — {parts}. Switch the sidebar to "
                       "\"2022 daily returns\" to use the real 2022 curve.")

    if use_optiq or not ct_latest.empty:
        section("By party — Optiq",
                "Share of returns vs registration share is the like-for-like comparison: "
                "a party ahead of its registration share is banking a larger slice of the early vote.")
        if use_optiq:
            # County feed: same moment as the district figures above; registration from the same feed
            _pc = cp_latest.groupby("party", as_index=False)[["registered", "returned"]].sum()
            ps = A.group_summary(_pc.rename(columns={"registered": "issued"}), ["party"]).set_index("party")
            party_reg_share = {p_: ps.loc[p_, "share_issued"] for p_ in ps.index}
            party_src = ometa
        else:
            ps = A.group_summary(ct_latest, ["party"]).set_index("party")
            party_reg_share = {p_: reg_counts.get(p_, 0) / reg_total if reg_total else np.nan for p_ in ps.index}
            party_src = None
        cols = st.columns(3)
        for i, p in enumerate(D.PARTIES):
            if p not in ps.index:
                continue
            r = ps.loc[p]
            reg_share = party_reg_share.get(p, np.nan)
            with cols[i]:
                card(D.PARTY_LABELS[p], f"{int(r['returned']):,}",
                     f"{pct(r['rate'])} returned · {pct(r['share_returned'])} of returns "
                     f"vs {pct(reg_share)} of reg ({pp(r['share_returned'] - reg_share)})",
                     color=D.PARTY_COLORS[p])
        d_ret, r_ret = ps.loc["D", "returned"], ps.loc["R", "returned"]
        tot = ps["returned"].sum()
        reg_margin = party_reg_share.get("D", np.nan) - party_reg_share.get("R", np.nan)
        st.caption(f"D−R margin among returned ballots: **{pp((d_ret - r_ret) / tot)}** "
                   f"(registration: {pp(reg_margin)}) · raw D−R: **{int(d_ret - r_ret):+,}**")
        st.caption(
            f"ℹ️ **SOS and Optiq totals don't match exactly** — SOS: {meta['returned']:,} returned of "
            f"{meta['issued']:,} issued ({meta['snapshot']:%b %-d}); Optiq: {int(ps['returned'].sum()):,} "
            f"of {int(ps['issued'].sum()):,} "
            + (f"(fetched {party_src['snapshot']:%b %-d}). " if party_src else
               f"({'live' if ct_source == 'live' else f'{ct_date:%b %-d}'}). ") + SOURCE_GAP_NOTE)

    # ── 2026 vs 2022 at the same days out, both against their own registration
    if not ct_latest.empty and pdi22 is not None and reg22 is not None:
        days_out = int((D.EDAY_2026 - pd.Timestamp(ct_date)).days)
        cum22 = A.pdi_cumulative(pdi22)
        r22 = A.at_days_out(cum22, days_out)
        reg_m22 = A.registration_margin(reg22)
        if not cp_latest.empty:
            g26 = cp_latest.groupby("party")["registered"].sum()
            reg_m26 = {"D": g26["D"] / g26.sum(), "R": g26["R"] / g26.sum(), "O": g26["O"] / g26.sum(),
                       "dr": (g26["D"] - g26["R"]) / g26.sum(), "src": "Optiq"}
        else:
            rc = reg_counts
            reg_m26 = {"D": rc.get("D", 0) / reg_total, "R": rc.get("R", 0) / reg_total,
                       "O": rc.get("O", 0) / reg_total, "dr": (rc.get("D", 0) - rc.get("R", 0)) / reg_total,
                       "src": "SOS 60-day report"}
        # Returns from the same Optiq feed as registration when possible, so both are the same moment
        if not cp_latest.empty:
            ret26 = cp_latest.groupby("party")[["returned"]].sum()
            ret_src = "Optiq county feed"
        else:
            ret26 = A.group_summary(ct_latest, ["party"]).set_index("party")
            ret_src = "Optiq crosstabs"
        ret_date = today   # compare with 2022 at today's E-number
        days_out = int((D.EDAY_2026 - ret_date).days)
        r22 = A.at_days_out(cum22, days_out)
        tot26 = ret26["returned"].sum()
        m26 = (ret26.loc["D", "returned"] - ret26.loc["R", "returned"]) / tot26
        section(f"2026 vs 2022 at {days_out} days out — against each year's registration",
                "Registration has moved toward Republicans since 2022, so raw D−R comparisons overstate the "
                "change. The fair test is how far each year's returns sit from that year's registration.")
        if r22 is None or r22["cum_total"] == 0:
            st.caption("No 2022 returns yet at this many days out.")
        else:
            gap26, gap22 = m26 - reg_m26["dr"], r22["dr_margin"] - reg_m22["dr"]
            k1, k2, k3 = st.columns(3)
            with k1:
                card("2026: returns vs registration", pp(gap26),
                     f"D−R returns {pp(m26)} · registration {pp(reg_m26['dr'])}")
            with k2:
                card("2022 at same point", pp(gap22),
                     f"D−R returns {pp(r22['dr_margin'])} · registration {pp(reg_m22['dr'])}")
            with k3:
                diff = gap26 - gap22
                card("Difference (2026 − 2022)", pp(diff),
                     "positive = Democrats doing better than 2022, net of registration",
                     color=D.PARTY_COLORS["D"] if diff >= 0 else D.PARTY_COLORS["R"])
            cmp = pd.DataFrame([
                {"": f"2026 returns ({ret_src})", "D": ret26.loc["D", "returned"] / tot26, "R": ret26.loc["R", "returned"] / tot26,
                 "O": ret26.loc["O", "returned"] / tot26, "dr": m26},
                {"": f"2026 registration ({reg_m26['src']})", "D": reg_m26["D"], "R": reg_m26["R"],
                 "O": reg_m26["O"], "dr": reg_m26["dr"]},
                {"": f"2022 returns at {days_out} days out", "D": r22["share_D"], "R": r22["share_R"],
                 "O": r22["share_O"], "dr": r22["dr_margin"]},
                {"": "2022 registration (Oct 24)", "D": reg_m22["D"], "R": reg_m22["R"], "O": reg_m22["O"],
                 "dr": reg_m22["dr"]},
            ])
            show_df(pct_cols(cmp, ["D", "R", "O", "dr"]), hide_index=True, width="stretch",
                         signed=['dr'], column_config={"D": pct_col("Dem"), "R": pct_col("Rep"), "O": pct_col("O/NPP"),
                                        "dr": pct_col("D−R", True)})
            st.caption(f"2022 returns are estimates read from PDI's chart (about ±0.5pp on these shares); "
                       f"{r22['frac_of_final']:.0%} of 2022's early vote was in by this point. "
                       "See the 2022 comparison tab for the full picture.")

    section("Cumulative returns")
    ss = A.sos_series(sos)
    dist = ss[ss["county"] == "SD4 total"]
    # Cumulative returns by party (Optiq): exact by return date when the live feed is up,
    # otherwise from the saved daily snapshots
    party_src = None
    if not live_cells.empty and live_cells["return_date"].notna().any():
        _rp = (live_cells.dropna(subset=["return_date"]).groupby(["return_date", "party"])["voters"].sum()
               .unstack("party", fill_value=0))
        _rp = _rp.reindex(pd.date_range(_rp.index.min(), _rp.index.max(), freq="D"), fill_value=0).cumsum()
        party_src = "Optiq, by each ballot's return date"
    elif not ct_hist.empty:
        _rp = ct_hist.groupby(["snapshot_date", "party"])["returned"].sum().unstack("party", fill_value=0)
        party_src = "Optiq daily snapshots, dated by fetch day"

    DAY_MS = 86_400_000
    fig = go.Figure()
    # One shared axis; bars first so the lines draw in front.
    if party_src:
        # Wide bar = Optiq's total returns per day; thin bars inside = its D / R / O-NPP parts,
        # so the three always add up to the wide bar
        _pd = _rp.diff()
        _pd.iloc[0] = _rp.iloc[0]
        _gap = _rp.index.to_series().diff().dt.days.fillna(1).clip(lower=1)
        _pd = _pd.div(_gap, axis=0).clip(lower=0)
        _pd = _pd[_pd.sum(axis=1) > 0]
        fig.add_bar(x=_pd.index, y=_pd.sum(axis=1), name="Returns / day (Optiq)",
                    marker_color="#c3dece", hovertemplate="%{y:,.0f}/day",
                    width=0.84 * DAY_MS, offset=-0.42 * DAY_MS)
        for i_, p_ in enumerate(D.PARTIES):
            if p_ in _pd:
                fig.add_bar(x=_pd.index, y=_pd[p_], name=f"{D.PARTY_LABELS[p_]} / day", legendgroup=p_,
                            marker_color=D.PARTY_COLORS[p_], opacity=0.85,
                            width=0.24 * DAY_MS, offset=(-0.37 + 0.25 * i_) * DAY_MS,
                            hovertemplate="%{y:,.0f}/day")
    else:
        fig.add_bar(x=dist["snapshot_date"], y=dist["per_day"], name="Returns / day (SOS)",
                    marker_color="#c3dece", hovertemplate="%{y:,.0f}/day")
    fig.add_scatter(x=dist["snapshot_date"], y=dist["vbm_returned"], name="Cumulative returned (SOS)",
                    mode="lines+markers", line=dict(color="#1a6b3c", width=2.5),
                    hovertemplate="%{y:,}")
    if party_src:
        for p_ in D.PARTIES:
            if p_ in _rp:
                fig.add_scatter(x=_rp.index, y=_rp[p_], name=f"{D.PARTY_LABELS[p_]} (Optiq)", mode="lines",
                                line=dict(color=D.PARTY_COLORS[p_], width=2), legendgroup=p_,
                                hovertemplate="%{y:,}")
    fig.update_layout(**PLOT_LAYOUT, height=360, barmode="overlay")
    fig.update_yaxes(tickformat=",", rangemode="tozero")
    st.plotly_chart(fig, width="stretch")
    st.caption(("Bars: Optiq returns per day — the wide bar is the total, the thin bars inside are Democrat, "
                "Republican and Other/NPP. " if party_src else "Bars: SOS returns per day. ")
               + "Green line: SOS cumulative returns (snapshots where the figures didn't change are dropped)."
               + (f" Party data: {party_src}; Optiq's totals run a little below SOS." if party_src else ""))

# ══════════════════════════════════════════════════════════════════════════════
# COUNTIES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    cty_more = st.toggle("Show more columns", key="cty_more",
                         help="Adds the 2022 same-day rate, shares of returns and registration, ballots "
                              "not yet returned, and party ballot counts.")
    if use_optiq:
        cp_tab = A.county_party_table(cp_latest)
        src_txt = ("live API" if cp_source == "live" else f"saved snapshot {ometa['snapshot']:%b %-d}")
        section("County returns: totals — Optiq",
                f"Source: Optiq ({src_txt}). Optiq assigns each voter to SD4 directly, so split-county "
                "figures are exact rather than scaled, and every row carries the D/R/O split. "
                "Rates use Optiq's registered count for the SD4 part of each county. Optiq can run "
                "about a day behind SOS.")
        tab = oct_.merge(A.last_drop(A.optiq_county_series(cp_hist), "tot_ret"), on="county", how="left")
        # Optiq reports each county's latest return date directly — prefer it
        tab["last_drop_date"] = tab["latest_return_date"].fillna(tab["last_drop_date"])
        tab["pace"] = tab["pace_ratio"].map(A.pace_label)
        show = pct_cols(A.with_total_row(tab, base, "tot_reg", "tot_ret",
                                         ["share_of_district_returns", "share_of_district_reg"]),
                        ["rate", "rate_2022_equiv", "share_of_district_returns", "share_of_district_reg"])
        # Last update: time of the Optiq data load that brought each county's latest ballots
        # (logged by the Optiq Action from Optiq's latestUpdate; the live pull covers any change
        # since the last logged one). Falls back to the latest return date where no time is known.
        _drops = data["drops"]
        _now_ret = cp_latest.groupby("county")["returned"].sum() if not cp_latest.empty else pd.Series(dtype=int)
        _upd = {}
        for c_ in tab["county"]:
            dl = _drops[_drops["county"] == c_]
            ts_ = None
            if len(dl):
                last_ret, last_ts = dl["returned"].iloc[-1], dl["loaded_at"].iloc[-1]
                if (cp_source == "live" and optiq_loaded is not None and c_ in _now_ret
                        and _now_ret[c_] != last_ret):
                    ts_ = optiq_loaded
                elif last_ts is not None and pd.notna(last_ts):
                    ts_ = last_ts
            _upd[c_] = ts_
        def _upd_txt(c_, fallback):
            ts_ = _upd.get(c_)
            if ts_ is not None:
                return f"{ts_:%b %-d, %-I:%M %p}".replace("AM", "am").replace("PM", "pm")
            return f"{fallback:%b %-d}" if fallback is not None and pd.notna(fallback) else "—"
        _times = [t for t in _upd.values() if t is not None]
        _upd["SD4 total"] = max(_times) if _times else None   # most recent update in any county
        show["last_update"] = [_upd_txt(c_, d_) for c_, d_ in zip(show["county"], show["last_drop_date"])]
        _filled = show["county"].isin(tab.loc[tab["sos_filled"], "county"])
        show.loc[_filled, "county"] = show.loc[_filled, "county"] + "*"
        show["pace_ratio"] = show["pace_ratio"] * 100
        _cols1 = ["county", "tot_reg", "tot_ret", "rate", "last_drop", "last_update", "pace_ratio", "pace"]
        if cty_more:
            _cols1 += ["rate_2022_equiv", "share_of_district_returns", "share_of_district_reg", "outstanding"]
        show_df(
            show[_cols1], pace="pace", height=38 + 35 * len(show),
            column_config={
                "county": "County", "tot_reg": INT("Registered"), "tot_ret": INT("Returned"),
                "rate": pct_col("Rate"),
                **DROP_COLS("Optiq snapshot"),
                "last_update": st.column_config.TextColumn(
                    "Last update",
                    help="When Optiq loaded the county's latest new ballots (Pacific time). Where that "
                         "isn't known yet, the date of the county's most recent ballot return."),
                "rate_2022_equiv": st.column_config.NumberColumn(
                    "2022 same day", format="%.1f%%",
                    help=f"The county's estimated 2022 return rate at today's point in the window ({pace_name})."),
                "pace_ratio": st.column_config.NumberColumn(
                    "vs 2022", format="%.0f%%",
                    help=f"This year's rate as a share of the 2022 same-day rate ({pace_name}). 100% = level."),
                "pace": st.column_config.TextColumn("Pace", help=f"Compared with 2022 at the same point ({pace_name})."),
                "share_of_district_returns": pct_col("Share of returns"),
                "share_of_district_reg": pct_col("Share of reg."), "outstanding": INT("Not yet returned"),
            },
        )
        _cap = ("Last update is when Optiq loaded the county's latest new ballots, to within a few hours "
                "(dates without a time are the county's most recent ballot return date, shown until a "
                "change has been logged). "
                "Registered is Optiq's VBM universe (sums to the same total as the crosstab 'issued'). ")
        if tab["sos_filled"].any():
            _cap += ("* Optiq shows 0 returned for this county, so its returned count and date come from "
                     "the latest SOS report; its party split isn't available. ")
        if ometa["n_snapshots"] < 2:
            _cap += ("Last drop (ballots added in the latest update) needs Optiq snapshots on two different "
                     "days; it fills in once the GitHub Action has saved a second day (after the PR is merged).")
        else:
            _cap += ("Last drop: ballots added in the county's most recent update. Counties that haven't moved "
                     "since keep their earlier drop. 2022 comparison uses 2022 SOS rates, so treat small gaps "
                     "as noise.")
        st.caption(_cap)

        section("County returns: partisan split — Optiq",
                "Ballots returned by party, with each party's share of the county's returns. "
                "D−R returns = (D − R returned) ÷ all returned; D−R reg = the same margin in registration; "
                "shift vs reg = the difference (positive = Democrats banking a bigger share than their "
                "registration).")
        v = A.county_party_with_total(cp_latest)
        for p_ in D.PARTIES:
            v[f"{p_}_share_ret"] = v[f"{p_}_share_ret"] * 100
            v[f"{p_}_share_reg"] = v[f"{p_}_share_reg"] * 100
        v["dr_margin_ret"] = v["dr_margin_ret"] * 100
        v["dr_margin_reg"] = v["dr_margin_reg"] * 100
        v["margin_shift"] = v["margin_shift"] * 100
        cols2 = (["county"]
                 + ([f"{p_}_ret" for p_ in D.PARTIES] if cty_more else [])
                 + [f"{p_}_share_ret" for p_ in D.PARTIES]
                 + (["raw_dr"] if cty_more else [])
                 + ["dr_margin_ret", "dr_margin_reg", "margin_shift"])
        cfg2 = {"county": "County",
                "raw_dr": st.column_config.NumberColumn("D−R returned", format="%+d",
                                                        help="Democratic minus Republican ballots returned."),
                "dr_margin_ret": pct_col("D−R returns", True),
                "dr_margin_reg": pct_col("D−R reg", True),
                "margin_shift": pct_col("Shift vs reg", True)}
        short = {"D": "D", "R": "R", "O": "O/NPP"}
        for p_ in D.PARTIES:
            cfg2[f"{p_}_ret"] = INT(f"{short[p_]} returned")
            cfg2[f"{p_}_share_ret"] = st.column_config.NumberColumn(
                f"{short[p_]} %", format="%.1f%%", help=f"{D.PARTY_LABELS[p_]} share of the county's returned ballots.")
        show_df(v[cols2], height=38 + 35 * len(v), column_config=cfg2,
                signed=["raw_dr", "dr_margin_ret", "dr_margin_reg", "margin_shift"])

        section("Returns margin vs registration margin — counties",
                "Each county, sized by ballots returned. Above the diagonal, the county's returns so far are "
                "more Democratic than its registration; below, more Republican. The black diamond is SD4.")
        cs_ = v[v["county"] != "SD4 total"].copy()
        cs_["ret_all"] = cs_[[f"{p_}_ret" for p_ in D.PARTIES]].sum(axis=1)
        cs_ = cs_[(cs_["ret_all"] > 0) & np.isfinite(cs_["dr_margin_ret"])]
        few_ = cs_[cs_["ret_all"] < 25]
        cs_ = cs_[cs_["ret_all"] >= 25]
        if len(few_):
            st.caption("Not shown (fewer than 25 ballots back, so the margin is noise): "
                       + ", ".join(f"{r_.county} ({int(r_.ret_all)})" for r_ in few_.itertuples()) + ".")
        if len(cs_):
            fig = go.Figure()
            smax = float(cs_["ret_all"].max())
            for r_ in cs_.itertuples():
                fig.add_scatter(
                    x=[r_.dr_margin_reg], y=[r_.dr_margin_ret], mode="markers+text", name=r_.county,
                    text=[r_.county], textposition="top center", textfont=dict(size=10),
                    marker=dict(size=float(np.clip(np.sqrt(r_.ret_all / smax) * 46, 8, 46)),
                                color=COUNTY_COLORS.get(r_.county, "#666"), opacity=0.75,
                                line=dict(width=0.5, color="white")),
                    showlegend=False,
                    hovertemplate=(f"<b>{r_.county}</b><br>{int(r_.ret_all):,} returned<br>"
                                   "Reg %{x:+.1f}% · Returns %{y:+.1f}%<extra></extra>"))
            tot_ = v[v["county"] == "SD4 total"]
            if len(tot_):
                fig.add_scatter(x=tot_["dr_margin_reg"], y=tot_["dr_margin_ret"], mode="markers+text",
                                text=["SD4"], textposition="bottom center", showlegend=False,
                                marker=dict(symbol="diamond", size=14, color="#111"),
                                hovertemplate="<b>SD4</b><br>Reg %{x:+.1f}% · Returns %{y:+.1f}%<extra></extra>")
            lo = float(min(cs_["dr_margin_reg"].min(), cs_["dr_margin_ret"].min())) - 5
            hi = float(max(cs_["dr_margin_reg"].max(), cs_["dr_margin_ret"].max())) + 5
            fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#1f1f1f", width=2.5, dash="dash"),
                            showlegend=False, hoverinfo="skip")
            fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"}, height=480)
            fig.update_xaxes(title="D−R margin of registration", ticksuffix="%", tickformat="+.0f")
            fig.update_yaxes(title="D−R margin of returns", ticksuffix="%", tickformat="+.0f")
            st.plotly_chart(fig, width="stretch")
    else:
        cp_tab = pd.DataFrame()
        section("County returns — SOS (Optiq unavailable)",
                "The Optiq API couldn't be reached and no optiq_county_party.csv snapshots exist yet, so this "
                "falls back to SOS. Split counties (Madera, Merced, Nevada, Placer) are SOS county totals "
                "scaled by SD4's registration share.")
        _ss = A.sos_series(sos)
        ctab = county_t.merge(A.last_drop(_ss, "vbm_returned"), on="county", how="left")
        ctab["pace"] = ctab["pace_ratio"].map(A.pace_label)
        show = pct_cols(A.with_total_row(ctab, base, "issued", "returned",
                                         ["share_of_district_returns", "share_of_district_issued"]),
                        ["rate", "rate_2022_equiv", "share_of_district_returns", "share_of_district_issued"])
        show["pace_ratio"] = show["pace_ratio"] * 100
        _cols1 = ["county", "issued", "returned", "rate", "last_drop", "last_drop_date", "pace_ratio", "pace"]
        if cty_more:
            _cols1 += ["rate_2022_equiv", "share_of_district_returns", "share_of_district_issued", "outstanding"]
        show_df(
            show[_cols1], pace="pace", height=38 + 35 * len(show),
            column_config={
                "county": "County", "issued": INT("Issued"), "returned": INT("Returned"),
                "rate": pct_col("Rate"),
                **DROP_COLS("SOS report"),
                "rate_2022_equiv": st.column_config.NumberColumn(
                    "2022 same day", format="%.1f%%",
                    help=f"The county's estimated 2022 return rate at today's point in the window ({pace_name})."),
                "pace_ratio": st.column_config.NumberColumn(
                    "vs 2022", format="%.0f%%",
                    help=f"This year's rate as a share of the 2022 same-day rate ({pace_name}). 100% = level."),
                "pace": st.column_config.TextColumn("Pace", help=f"Compared with 2022 at the same point ({pace_name})."),
                "share_of_district_returns": pct_col("Share of returns"),
                "share_of_district_issued": pct_col("Share of issued"), "outstanding": INT("Outstanding"),
            },
        )

    if use_optiq and cp_latest["returned"].sum() > 0:
        dec, dec_c = A.margin_decomposition(cp_latest)
        section("Why the district shift differs from the counties",
                "The district's D−R shift vs registration splits into two parts that add up exactly. "
                "Within counties: Democrats and Republicans returning at different rates inside each county. "
                "County mix: some counties making up more of the returns than of registration. "
                "Mix fades as every county's share of returns approaches its share of registration.")
        k1, k2, k3 = st.columns(3)
        with k1:
            card("Within counties", pp(dec["within"]), "party return rates inside each county",
                 color=D.PARTY_COLORS["D"] if dec["within"] >= 0 else D.PARTY_COLORS["R"])
        with k2:
            card("County mix", pp(dec["mix"]), "which counties have returned most",
                 color=D.PARTY_COLORS["D"] if dec["mix"] >= 0 else D.PARTY_COLORS["R"])
        with k3:
            card("District shift", pp(dec["shift"]),
                 f"D−R returns {pp(dec['ret'])} vs registration {pp(dec['reg'])}")

        dc1, dc2 = st.columns(2)
        with dc1:
            fig = go.Figure(go.Waterfall(
                orientation="v", measure=["absolute", "relative", "relative", "total"],
                x=["D−R registration", "Within counties", "County mix", "D−R returns"],
                y=[dec["reg"] * 100, dec["within"] * 100, dec["mix"] * 100, 0],
                text=[f"{dec['reg']*100:+.1f}", f"{dec['within']*100:+.1f}", f"{dec['mix']*100:+.1f}",
                      f"{dec['ret']*100:+.1f}"],
                textposition="outside",
                increasing=dict(marker=dict(color=D.PARTY_COLORS["D"])),
                decreasing=dict(marker=dict(color=D.PARTY_COLORS["R"])),
                totals=dict(marker=dict(color="#555555")),
                connector=dict(line=dict(color="#bbb", width=1)),
                hovertemplate="%{x}: %{y:+.1f} pp<extra></extra>",
            ))
            fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"}, height=360, showlegend=False,
                              title=dict(text="From registration margin to returns margin (pp)",
                                         font=dict(size=12)))
            fig.update_yaxes(ticksuffix=" pp", zeroline=True, zerolinecolor="#999")
            st.plotly_chart(fig, width="stretch")
        with dc2:
            dcs = dec_c[(dec_c["within"].abs() + dec_c["mix"].abs()) > 0.0005].copy()
            dcs["total"] = dcs["within"] + dcs["mix"]
            dcs = dcs.sort_values("total")
            fig = go.Figure()
            fig.add_bar(y=dcs["county"], x=dcs["within"] * 100, name="Within county", orientation="h",
                        marker_color="#7c9cc4", hovertemplate="%{y}: %{x:+.2f} pp<extra>Within</extra>")
            fig.add_bar(y=dcs["county"], x=dcs["mix"] * 100, name="County mix", orientation="h",
                        marker_color="#d9a441", hovertemplate="%{y}: %{x:+.2f} pp<extra>Mix</extra>")
            fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest",
                                 "legend": dict(orientation="h", y=-0.12, x=0),
                                 "margin": dict(l=10, r=10, t=40, b=10)},
                              barmode="relative", height=380,
                              title=dict(text="Each county's contribution (pp)", font=dict(size=12)))
            fig.update_xaxes(ticksuffix=" pp", zeroline=True, zerolinecolor="#999")
            st.plotly_chart(fig, width="stretch")
        st.caption("Positive = toward Democrats. A county's mix piece is negative when it is over-represented "
                   "in returns and more Republican than the district, or under-represented and more Democratic.")

        dser = A.decomposition_series(cp_hist)
        if len(dser) >= 2:
            fig = go.Figure()
            for col, name, colr, dash in (("within", "Within counties", "#7c9cc4", None),
                                          ("mix", "County mix", "#d9a441", None),
                                          ("shift", "District shift", "#222222", "dot")):
                fig.add_scatter(x=dser["snapshot_date"], y=dser[col] * 100, name=name, mode="lines+markers",
                                line=dict(color=colr, width=2.5 if col == "shift" else 2, dash=dash),
                                hovertemplate="%{y:+.1f} pp")
            fig.update_layout(**PLOT_LAYOUT, height=320,
                              title=dict(text="Over time (pp)", font=dict(size=12)))
            fig.update_yaxes(ticksuffix=" pp", zeroline=True, zerolinecolor="#999")
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("The over-time chart appears once Optiq county snapshots exist on two different days.")

    # Map metrics: Optiq when available, else SOS
    if use_optiq:
        oi = oct_.set_index("county")
        cpi = cp_tab.set_index("county")
        metric_opts = {
            "Return rate": (oi["rate"].to_dict(), False, "{:.1%}", "Greens"),
            "vs 2022 pace": ((oi["pace_ratio"] - 1).to_dict(), True, "{:+.0%}", "RdYlGn"),
            "Share of district returns − share of reg.": (
                (oi["share_of_district_returns"] - oi["share_of_district_reg"]).to_dict(), True, "{:+.1%}", "PiYG"),
            "Dem return rate": (cpi["D_rate"].to_dict(), False, "{:.1%}", "Blues"),
            "Rep return rate": (cpi["R_rate"].to_dict(), False, "{:.1%}", "Reds"),
            "D−R margin among returns": (cpi["dr_margin_ret"].to_dict(), True, "{:+.1%}", "RdBu"),
            "Margin shift vs registration": (cpi["margin_shift"].to_dict(), True, "{:+.1%}", "RdBu"),
            "D − R return-rate gap": (cpi["turnout_gap"].to_dict(), True, "{:+.1%}", "RdBu"),
            "Raw D − R returned": (cpi["raw_dr"].astype(float).to_dict(), True, "{:+,.0f}", "RdBu"),
        }
        vel = tab[["county", "last_drop", "last_drop_date", "tot_ret"]].rename(columns={"tot_ret": "returned"})
    else:
        metric_opts = {
            "Return rate": (dict(zip(county_t["county"], county_t["rate"])), False, "{:.1%}", "Greens"),
            "vs 2022 pace": (dict(zip(county_t["county"], county_t["pace_ratio"] - 1)), True, "{:+.0%}", "RdYlGn"),
            "Share of district returns − share of reg.": (
                dict(zip(county_t["county"], county_t["share_of_district_returns"] - county_t["share_of_district_issued"])),
                True, "{:+.1%}", "PiYG"),
        }
        vel = ctab[["county", "last_drop", "last_drop_date", "returned"]]
    mcol, tcol = st.columns([3, 2])
    with mcol:
        _lbl = lambda k: f"vs 2022 {pace_name}" if k == "vs 2022 pace" else k
        sel = st.selectbox("Map metric", list(metric_opts), format_func=_lbl, key="map_metric")
        vals, div, fmt, cs = metric_opts[sel]
        vals = {cn: float(vals.get(cn, np.nan)) for cn in D.SD4_COUNTIES}
        county_map(vals, f"{_lbl(sel)} ({'Optiq' if use_optiq else 'SOS'})", div, fmt, cs)
    with tcol:
        st.markdown("**Biggest last drops**")
        top = vel.sort_values("last_drop", ascending=False).head(6)
        if top["last_drop"].notna().any():
            show_df(top, hide_index=True, width="stretch",
                         column_config={"county": "County", **DROP_COLS("snapshot"),
                                        "returned": INT("Total returned")})
        else:
            st.caption("Appears once there are snapshots on two different days.")
        st.caption("Stanislaus holds nearly half of SD4's ballots, so its return speed moves the "
                   "district number more than every small county combined.")

    # Trends: Optiq once there's history, else SOS
    oser = A.optiq_county_series(cp_hist) if use_optiq else pd.DataFrame()
    optiq_trend = not oser.empty and oser["snapshot_date"].nunique() >= 2
    section(f"County return rate over time — {'Optiq' if optiq_trend else 'SOS'}",
            None if optiq_trend else "Uses SOS until Optiq county snapshots exist on at least two days.")
    pick = st.multiselect("Counties", D.SD4_COUNTIES + ["SD4 total"],
                          default=["SD4 total", "Stanislaus", "El Dorado", "Madera", "Amador"])
    ss = oser if optiq_trend else A.sos_series(sos)
    fig = go.Figure()
    for cn in pick:
        s_ = ss[ss["county"] == cn]
        fig.add_scatter(x=s_["snapshot_date"], y=s_["rate"], name=cn, mode="lines+markers",
                        line=dict(width=3 if cn == "SD4 total" else 1.8, color=COUNTY_COLORS[cn]),
                        hovertemplate="%{y:.1%}")
    fig.update_layout(**PLOT_LAYOUT, height=340)
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    if optiq_trend:
        section("County D−R margin among returns over time — Optiq")
        fig = go.Figure()
        for cn in D.SD4_COUNTIES:
            s_ = oser[oser["county"] == cn]
            fig.add_scatter(x=s_["snapshot_date"], y=s_["dr_margin_ret"], name=cn, mode="lines",
                            line=dict(color=COUNTY_COLORS[cn]), hovertemplate="%{y:+.1%}")
        fig.update_layout(**PLOT_LAYOUT, height=340)
        fig.update_yaxes(tickformat="+.0%")
        st.plotly_chart(fig, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# PRECINCTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    ptab, pmeta = A.precinct_table(data["pre"], data["pre_cov"])
    if ptab.empty:
        st.info("No precinct data yet. It's saved by the *Update Optiq Crosstabs* Action "
                "(`fetch_optiq_precincts.py`) from Civiq's precinct map, and appears here after its next run.")
    else:
        shapes, sd4_ring = load_precinct_geo()
        new_lbl = f"New since {pmeta['prev']:%b %-d}" if pmeta["prev"] is not None else None
        section("Precincts",
                f"Optiq precinct figures from Civiq's map, saved {pmeta['snapshot']:%b %-d} "
                f"({int((~ptab['unmapped']).sum()):,} SD4 precincts; {pmeta['n_snapshots']} saved snapshot"
                f"{'s' if pmeta['n_snapshots'] != 1 else ''}). Split counties keep the precincts whose centre "
                f"is inside SD4. Grey on the map: precincts with fewer than {A.PRECINCT_MIN_REG} registered voters, "
                f"and for the margin views, fewer than {A.PRECINCT_MIN_RET} ballots back — too few to read.")
        if pmeta.get("unmapped_counties"):
            st.caption("Some precincts aren't on Civiq's map, so their ballots can't be placed. Each county's "
                       f"remainder is a “{A.UNMAPPED}” row in the table (party split from Optiq's county "
                       "figures), so totals match Optiq: " + ", ".join(
                           f"{c} {n:,} voters" for c, n in sorted(pmeta["unmapped_counties"].items(),
                                                                  key=lambda kv: -kv[1])) + ".")
        c1, c2 = st.columns([2, 1])
        metrics = {
            "Return rate": ("rate", False, "{:.1%}", "Greens"),
            "D−R margin of returns": ("dr_ret", True, "{:+.1%}", "RdBu"),
            "Shift vs registration (D−R returns − D−R reg)": ("shift", True, "{:+.1%}", "RdBu"),
            "Ballots returned": ("returned", False, "{:,.0f}", "Greens"),
        }
        if new_lbl:
            metrics[new_lbl] = ("new", False, "{:,.0f}", "Greens")
        mname = c1.selectbox("Colour precincts by", list(metrics), key="pre_metric")
        county_opts = ["All of SD4"] + sorted(ptab["county"].unique())
        csel = c2.selectbox("County", county_opts, key="pre_county")
        view = ptab if csel == "All of SD4" else ptab[ptab["county"] == csel]
        col, diverging, fmt, cs = metrics[mname]

        # Summary cards for the selection
        k = st.columns(4)
        tot_reg, tot_ret = int(view["registered"].sum()), int(view["returned"].sum())
        card_ret = view[["D", "R", "O"]].sum()
        with k[0]:
            card("Ballots returned", f"{tot_ret:,}", f"{tot_reg:,} registered")
        with k[1]:
            card("Return rate", pct(tot_ret / tot_reg if tot_reg else np.nan),
                 f"Precincts {pct(view.loc[~view['small'] & ~view['unmapped'], 'rate'].min())} – "
                 f"{pct(view.loc[~view['small'] & ~view['unmapped'], 'rate'].max())}")
        with k[2]:
            dr_ret = (card_ret["D"] - card_ret["R"]) / tot_ret if tot_ret else np.nan
            card("D−R margin of returns", pct(dr_ret) if not np.isfinite(dr_ret) else f"{dr_ret:+.1%}")
        with k[3]:
            dr_reg = (view["D_reg"].sum() - view["R_reg"].sum()) / tot_reg if tot_reg else np.nan
            shift = dr_ret - dr_reg
            card("Shift vs registration", pp(shift), f"D−R registration {dr_reg:+.1%}" if np.isfinite(dr_reg) else "")

        # Map
        if shapes is None:
            st.info("Precinct shapes (sd4_precincts.geojson) aren't saved yet — showing the table only.")
        else:
            shown = view[view["id"].isin({f["id"] for f in shapes["features"]})]
            z = shown[col].where(~shown["small"]) if col not in ("returned", "new") else shown[col]
            if col in ("dr_ret", "shift"):
                z = z.where(shown["returned"] >= A.PRECINCT_MIN_RET)
            finite = z[np.isfinite(z.astype(float))]
            if diverging:
                m = float(np.nanpercentile(np.abs(finite), 95)) if len(finite) else 1.0
                zmin, zmax = -(m or 1), (m or 1)
            else:
                zmin = float(np.nanpercentile(finite, 2)) if len(finite) else 0
                zmax = float(np.nanpercentile(finite, 98)) if len(finite) else 1
            hover = (
                "<b>" + shown["county"] + " " + shown["precinct"] + "</b><br>"
                + shown["returned"].map("{:,}".format) + " of " + shown["registered"].map("{:,}".format)
                + " returned (" + shown["rate"].map(lambda v: pct(v)) + ")<br>"
                + "D " + shown["D"].map("{:,}".format) + " · R " + shown["R"].map("{:,}".format)
                + " · O/NPP " + shown["O"].map("{:,}".format) + "<br>"
                + "D−R returns " + shown["dr_ret"].map(lambda v: "—" if not np.isfinite(v) else f"{v:+.1%}")
                + " vs reg " + shown["dr_reg"].map(lambda v: "—" if not np.isfinite(v) else f"{v:+.1%}")
                + (("<br>" + new_lbl + ": " + shown["new"].map(lambda v: f"{int(v):,}")) if new_lbl else "")
            )
            fig = go.Figure()
            ok = np.isfinite(z.astype(float))
            # Each trace gets only its own precincts' shapes, so no shape is sent to the browser twice
            feat_by_id = {f["id"]: f for f in shapes["features"]}
            def sub_geo(ids):
                return {"type": "FeatureCollection", "features": [feat_by_id[i] for i in ids]}
            # Grey for small / no-data precincts
            if (~ok).any():
                fig.add_trace(go.Choroplethmap(
                    geojson=sub_geo(shown.loc[~ok, "id"]), locations=shown.loc[~ok, "id"], featureidkey="id",
                    z=[0] * int((~ok).sum()), colorscale=[[0, "#e4e4e0"], [1, "#e4e4e0"]], showscale=False,
                    marker=dict(line=dict(width=0.3, color="#999")), text=hover[~ok], hoverinfo="text"))
            fig.add_trace(go.Choroplethmap(
                geojson=sub_geo(shown.loc[ok, "id"]), locations=shown.loc[ok, "id"], featureidkey="id", z=z[ok],
                zmin=zmin, zmax=zmax, colorscale=cs, zmid=0 if diverging else None,
                marker=dict(line=dict(width=0.3, color="#666"), opacity=0.9),
                text=hover[ok], hoverinfo="text",
                colorbar=dict(thickness=10, len=0.7,
                              tickformat=("+.0%" if diverging else (".0%" if "%" in fmt else ",.0f")))))
            if sd4_ring and csel == "All of SD4":
                fig.add_trace(go.Scattermap(lon=[p[0] for p in sd4_ring], lat=[p[1] for p in sd4_ring],
                                            mode="lines", line=dict(color="#111", width=1.5),
                                            hoverinfo="skip", showlegend=False))
            lons = [pt[0] for f in shapes["features"] if f["id"] in set(shown["id"])
                    for poly in f["geometry"]["coordinates"] for pt in poly[0]]
            lats = [pt[1] for f in shapes["features"] if f["id"] in set(shown["id"])
                    for poly in f["geometry"]["coordinates"] for pt in poly[0]]
            if lons:
                span = max(max(lons) - min(lons), (max(lats) - min(lats)) * 1.3, 0.05)
                zoom = float(np.clip(np.log2(360 / span) + 0.1, 4, 12))
                center = dict(lon=(min(lons) + max(lons)) / 2, lat=(min(lats) + max(lats)) / 2)
            else:
                zoom, center = 6, dict(lon=-119.0, lat=37.6)
            # white-bg needs no base-map tiles, so it renders behind any network
            fig.update_layout(map=dict(style="white-bg", zoom=zoom, center=center), height=620,
                              margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
            # Framed, and the mouse wheel scrolls the page rather than zooming: zoom with the +/− buttons
            with st.container(border=True):
                st.plotly_chart(fig, width="stretch",
                                config={"scrollZoom": False, "displayModeBar": True,
                                        "modeBarButtonsToRemove": ["select2d", "lasso2d"]})
                st.caption("Zoom with the +/− buttons in the corner (or pinch / double-click); "
                           "the mouse wheel scrolls the page.")

        # Registration margin vs returns margin
        section("Returns margin vs registration margin",
                f"Each precinct with at least {A.PRECINCT_MIN_RET} ballots back, sized by ballots returned. Above the diagonal, the returns so far are "
                "more Democratic than the precinct's registration; below, more Republican.")
        sc = view[~view["small"] & ~view["unmapped"] & np.isfinite(view["dr_ret"])
                  & (view["returned"] >= A.PRECINCT_MIN_RET)]
        if len(sc):
            fig = go.Figure()
            for cty, g in sc.groupby("county"):
                fig.add_scatter(x=g["dr_reg"], y=g["dr_ret"], mode="markers", name=cty,
                                marker=dict(size=np.clip(np.sqrt(g["returned"]) * 0.9, 4, 24),
                                            color=COUNTY_COLORS.get(cty, "#666"), opacity=0.7,
                                            line=dict(width=0.5, color="white")),
                                text=g["precinct"] + " · " + g["returned"].map("{:,}".format) + " returned",
                                hovertemplate="<b>%{text}</b><br>Reg %{x:+.1%} · Returns %{y:+.1%}<extra>"
                                              + cty + "</extra>")
            lo = float(min(sc["dr_reg"].min(), sc["dr_ret"].min())) - 0.05
            hi = float(max(sc["dr_reg"].max(), sc["dr_ret"].max())) + 0.05
            fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#1f1f1f", width=2.5, dash="dash"),
                            showlegend=False, hoverinfo="skip")
            fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"}, height=480)
            fig.update_xaxes(title="D−R margin of registration", tickformat="+.0%")
            fig.update_yaxes(title="D−R margin of returns", tickformat="+.0%")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No precincts with returns to plot yet.")

        # Table
        section("Precinct table", "Sort by any column. Shares are of each precinct's returned ballots.")
        cols = ["county", "precinct", "registered", "returned", "rate"] + (["new"] if new_lbl else []) + \
               ["D", "R", "O", "d_share", "r_share", "o_share", "dr_ret", "dr_reg", "shift"]
        tbl = pct_cols(view[cols], ["rate", "d_share", "r_share", "o_share", "dr_ret", "dr_reg", "shift"])
        show_df(tbl, hide_index=True, width="stretch", height=560, signed=['dr_ret', 'dr_reg', 'shift'], column_config={
            "county": "County", "precinct": "Precinct",
            "registered": INT("Registered"), "returned": INT("Returned"), "rate": pct_col("Rate"),
            "new": INT(new_lbl or "New"),
            "D": INT("D returned"), "R": INT("R returned"), "O": INT("O/NPP returned"),
            "d_share": pct_col("D %"), "r_share": pct_col("R %"), "o_share": pct_col("O/NPP %"),
            "dr_ret": pct_col("D−R returns", signed=True), "dr_reg": pct_col("D−R reg", signed=True),
            "shift": pct_col("Shift vs reg", signed=True),
        })
        st.download_button("Download precinct table (CSV)", view[cols].to_csv(index=False).encode(),
                           f"sd4_precincts_{pmeta['snapshot']:%Y-%m-%d}.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# CROSSTABS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    if ct_latest.empty:
        st.info("No Optiq crosstab data available.")
    else:
        st.caption(f"Optiq voter-file match · {'live pull' if ct_source == 'live' else f'snapshot {ct_date:%b %-d}'} "
                   f"· {int(ct_latest['issued'].sum()):,} ballots issued, "
                   f"{int(ct_latest['returned'].sum()):,} returned.")

        # Filters apply to everything on this tab
        with st.expander("Filter the universe", expanded=False):
            fcols = st.columns(4)
            filt = {}
            for i, dim in enumerate(["party", "age", "gender", "ethnicity"]):
                opts = [k for k in D.DIM_LABELS[dim] if k in set(ct_latest[dim])]
                filt[dim] = fcols[i].multiselect(D.DIM_NAMES[dim], opts, default=opts,
                                                 format_func=lambda k, d=dim: D.DIM_LABELS[d][k],
                                                 key=f"f_{dim}")
        cx = ct_latest
        for dim, keep in filt.items():
            cx = cx[cx[dim].isin(keep)]
        if cx.empty:
            st.warning("Filters exclude every ballot — showing the full universe instead.")
            cx = ct_latest

        section("One-way breakdowns",
                "Representation = share of returns − share of issued. Positive means the group "
                "makes up more of the returned ballots than of the ballots mailed.")
        one_cols = st.columns(2)
        for i, dim in enumerate(["party", "age", "gender", "ethnicity"]):
            s = A.label_codes(A.group_summary(cx, [dim]), [dim])
            order = list(D.DIM_LABELS[dim].values())
            s = s.set_index(dim).reindex([o for o in order if o in set(s[dim])]).reset_index()
            s = pct_cols(s, ["rate", "share_issued", "share_returned", "representation"])
            with one_cols[i % 2]:
                st.markdown(f"**{D.DIM_NAMES[dim]}**")
                show_df(
                    s[[dim, "issued", "returned", "rate", "share_issued", "share_returned", "representation"]],
                    hide_index=True, width="stretch",
                    column_config={dim: D.DIM_NAMES[dim], "issued": INT("Issued"), "returned": INT("Ret."),
                                   "rate": pct_col("Rate"), "share_issued": pct_col("% iss."),
                                   "share_returned": pct_col("% ret."),
                                   "representation": pct_col("Repr.", True)},
                )

        section("Two-way crosstab")
        METRICS = {
            "Return rate": ("rate", "{:.1%}", False),
            "Returned (count)": ("returned", "{:,.0f}", False),
            "Outstanding (count)": ("outstanding", "{:,.0f}", False),
            "Share of all returns": ("share_returned", "{:.1%}", False),
            "Composition within column": ("col_share", "{:.1%}", False),
            "Representation (pp)": ("representation", "{:+.2%}", True),
        }
        tc = st.columns([1, 1, 2])
        dims = ["party", "age", "gender", "ethnicity"]
        rdim = tc[0].selectbox("Rows", dims, index=1, format_func=D.DIM_NAMES.get)
        cdim = tc[1].selectbox("Columns", [d for d in dims if d != rdim], index=0, format_func=D.DIM_NAMES.get)
        mname = tc[2].selectbox("Metric", list(METRICS), index=0)
        mkey, mfmt, mdiv = METRICS[mname]
        piv = A.two_way(cx, rdim, cdim, mkey)
        zt = [[("" if not np.isfinite(v) else mfmt.format(v)) for v in row] for row in piv.values]
        if mdiv:
            m = np.nanmax(np.abs(piv.values)) or 1
            cs, zmin, zmax = "RdBu", -m, m
        else:
            cs, zmin, zmax = "Greens", np.nanmin(piv.values), np.nanmax(piv.values)
        fig = go.Figure(go.Heatmap(z=piv.values, x=list(piv.columns), y=list(piv.index), text=zt,
                                   texttemplate="%{text}", colorscale=cs, zmin=zmin, zmax=zmax,
                                   showscale=False, hovertemplate="%{y} · %{x}: %{text}<extra></extra>"))
        fig.update_layout(height=60 + 55 * len(piv.index), margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
        if mkey == "col_share":
            st.caption(f"Each column sums to 100%: the {D.DIM_NAMES[rdim].lower()} mix of returned ballots "
                       f"within each {D.DIM_NAMES[cdim].lower()} group.")

        section("Party gap within each group",
                "D−R return-rate gap and D−R share of returns inside every age, gender and ethnicity "
                "group — where each party is over- or under-performing its own base.")
        gap_rows = []
        for dim in ["age", "gender", "ethnicity"]:
            s = A.group_summary(cx, [dim, "party"]).set_index([dim, "party"])
            for code in [k for k in D.DIM_LABELS[dim] if k in set(cx[dim])]:
                def g(p, col):
                    return s.loc[(code, p), col] if (code, p) in s.index else np.nan
                ret_dr = g("D", "returned") + g("R", "returned") + g("O", "returned")
                iss = g("D", "issued") + g("R", "issued") + g("O", "issued")
                gap_rows.append({
                    "Dimension": D.DIM_NAMES[dim], "Group": D.DIM_LABELS[dim][code],
                    "D rate": g("D", "rate"), "R rate": g("R", "rate"), "O rate": g("O", "rate"),
                    "rate_gap": g("D", "rate") - g("R", "rate"),
                    "margin_ret": (g("D", "returned") - g("R", "returned")) / ret_dr if ret_dr else np.nan,
                    "margin_iss": (g("D", "issued") - g("R", "issued")) / iss if iss else np.nan,
                })
        gdf = pd.DataFrame(gap_rows)
        gdf["shift"] = gdf["margin_ret"] - gdf["margin_iss"]
        gdf = pct_cols(gdf, ["D rate", "R rate", "O rate", "rate_gap", "margin_ret", "margin_iss", "shift"])
        show_df(gdf, hide_index=True, width="stretch", height=38 + 35 * len(gdf), signed=['rate_gap', 'margin_ret', 'margin_iss', 'shift'], column_config={
            "D rate": pct_col("D rate"), "R rate": pct_col("R rate"), "O rate": pct_col("O/NPP rate"),
            "rate_gap": pct_col("D−R rate gap", True), "margin_ret": pct_col("D−R (returns)", True),
            "margin_iss": pct_col("D−R (issued)", True), "shift": pct_col("Shift", True)})

        section("Cell explorer", "Every party × age × gender × ethnicity cell, sortable.")
        cells = A.label_codes(A.group_summary(cx, ["party", "age", "gender", "ethnicity"]),
                              ["party", "age", "gender", "ethnicity"])
        min_n = st.slider("Minimum ballots issued", 0, 5000, 500, 100)
        cells = cells[cells["issued"] >= min_n].sort_values("representation", ascending=False)
        cells = pct_cols(cells, ["rate", "share_issued", "share_returned", "representation"])
        show_df(cells[["party", "age", "gender", "ethnicity", "issued", "returned", "rate",
                            "share_returned", "representation"]],
                     hide_index=True, width="stretch", height=360,
                     column_config={"party": "Party", "age": "Age", "gender": "Gender",
                                    "ethnicity": "Ethnicity", "issued": INT("Issued"),
                                    "returned": INT("Returned"), "rate": pct_col("Rate"),
                                    "share_returned": pct_col("% of returns"),
                                    "representation": pct_col("Repr.", True)})
        st.caption("Age codes 1/3/4/6 = 18–34 / 35–49 / 50–64 / 65+. Ethnicity is modeled from the "
                   "voter file, not self-reported.")

# ══════════════════════════════════════════════════════════════════════════════
# TRENDS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    section("Daily returns — SOS", "New ballots per day between reports, by county (stacked).")
    ss = A.sos_series(sos)
    cty = ss[(ss["county"] != "SD4 total") & ss["per_day"].notna()]
    fig = go.Figure()
    order = county_t.sort_values("returned", ascending=False)["county"].tolist()
    for i, cn in enumerate(order):
        s = cty[cty["county"] == cn]
        fig.add_bar(x=s["snapshot_date"], y=s["per_day"], name=cn,
                    marker_color=COUNTY_COLORS[cn], hovertemplate="%{y:,.0f}/day")
    fig.update_layout(**PLOT_LAYOUT, barmode="stack", height=360)
    st.plotly_chart(fig, width="stretch")

    if ct_hist.empty or ct_hist["snapshot_date"].nunique() < 2:
        st.info("Optiq trend charts need at least two distinct snapshots.")
    else:
        tdim = st.radio("Break Optiq trends down by", ["party", "age", "gender", "ethnicity"],
                        horizontal=True, format_func=D.DIM_NAMES.get)
        os_ = A.optiq_series(ct_hist, tdim)
        codes = [k for k in D.DIM_LABELS[tdim] if k in set(os_[tdim])]

        tc1, tc2 = st.columns(2)
        with tc1:
            section("Return rate over time")
            fig = go.Figure()
            for i, k in enumerate(codes):
                s = os_[os_[tdim] == k]
                fig.add_scatter(x=s["snapshot_date"], y=s["rate"], name=D.DIM_LABELS[tdim][k],
                                mode="lines+markers", line=dict(color=dim_color(tdim, k, i), width=2),
                                hovertemplate="%{y:.2%}")
            fig.update_layout(**PLOT_LAYOUT, height=320)
            fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig, width="stretch")
        with tc2:
            section("Share of all returns over time")
            fig = go.Figure()
            for i, k in enumerate(codes):
                s = os_[os_[tdim] == k]
                fig.add_scatter(x=s["snapshot_date"], y=s["share_returned"], name=D.DIM_LABELS[tdim][k],
                                mode="lines+markers", line=dict(color=dim_color(tdim, k, i), width=2),
                                hovertemplate="%{y:.1%}")
                # Dashed reference: share of issued (constant-ish)
                fig.add_scatter(x=s["snapshot_date"], y=s["share_issued"], mode="lines", showlegend=False,
                                line=dict(color=dim_color(tdim, k, i), dash="dot", width=1),
                                hoverinfo="skip")
            fig.update_layout(**PLOT_LAYOUT, height=320)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
            st.caption("Dotted lines: each group's share of ballots issued.")

        section("Composition of each day's new returns",
                "Who returned ballots in each reporting interval — catches shifts that cumulative "
                "shares smooth over. Intervals with fewer than 100 new ballots are left blank.")
        fig = go.Figure()
        for i, k in enumerate(codes):
            s = os_[(os_[tdim] == k) & os_["share_of_new"].notna()]
            fig.add_bar(x=s["snapshot_date"], y=s["share_of_new"], name=D.DIM_LABELS[tdim][k],
                        marker_color=dim_color(tdim, k, i), hovertemplate="%{y:.1%}")
        fig.update_layout(**PLOT_LAYOUT, barmode="stack", height=300)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

        if not live_cells.empty:
            curve = A.live_daily_curve(live_cells, tdim)
            if not curve.empty:
                section("Daily returns by return date — live Optiq",
                        "Built from each ballot's recorded return date, so it isn't limited to the days "
                        "a snapshot happened to run.")
                fig = go.Figure()
                for i, k in enumerate(codes):
                    s = curve[curve[tdim] == k]
                    fig.add_bar(x=s["date"], y=s["new"], name=D.DIM_LABELS[tdim][k],
                                marker_color=dim_color(tdim, k, i), hovertemplate="%{y:,}")
                fig.update_layout(**PLOT_LAYOUT, barmode="stack", height=300)
                st.plotly_chart(fig, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# OUTSTANDING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    # ── O/NPP split needed to close the D−R gap among returned ballots ──────
    if use_optiq or not ct_latest.empty:
        _pt = (cp_latest.groupby("party")["returned"].sum() if use_optiq
               else ct_latest.groupby("party")["returned"].sum())
        _d, _r, _n = (int(_pt.get(p_, 0)) for p_ in ("D", "R", "O"))
        section("What O/NPP voters would need to do",
                "If every Democratic ballot votes Democratic and every Republican ballot votes Republican, "
                "this is the share of O/NPP ballots the trailing side needs just to draw level on the ballots "
                "returned so far.")
        nc = st.columns(3)
        with nc[0]:
            if _n > 0:
                need_d = (_n + _r - _d) / (2 * _n)   # D + x·N = R + (1 − x)·N
                if need_d > 1:
                    val, sub, colr = ">100%", "Democrats can't close the gap with O/NPP ballots alone", D.PARTY_COLORS["R"]
                elif need_d < 0:
                    val, sub, colr = "<0%", "Democrats lead even if every O/NPP ballot goes Republican", D.PARTY_COLORS["D"]
                else:
                    val, sub, colr = pct(need_d), f"Republicans would need {pct(1 - need_d)}", (
                        D.PARTY_COLORS["R"] if need_d > 0.5 else D.PARTY_COLORS["D"])
                card("O/NPP share Democrats need", val, sub, color=colr)
            else:
                card("O/NPP share Democrats need", "—", "no O/NPP ballots returned yet")
        with nc[1]:
            gap = _d - _r
            card("Democrats vs Republicans", f"{gap:+,}",
                 f"{_d:,} D · {_r:,} R returned", color=D.PARTY_COLORS["D"] if gap >= 0 else D.PARTY_COLORS["R"])
        with nc[2]:
            card("O/NPP ballots returned", f"{_n:,}", f"{_n / max(_d + _r + _n, 1):.1%} of returns")
        st.caption("Party registration of returned ballots, not candidate votes — crossover and O/NPP turnout "
                   "differences will move the real figure. Source: "
                   + ("Optiq county feed." if use_optiq else "Optiq crosstabs."))

        # ── Scenario grid: margin on returned ballots under crossover × O/NPP split ──
        section("Margin on returned ballots: crossover × O/NPP split",
                "What the D−R vote margin among ballots already returned would be, for different party loyalty "
                "and O/NPP splits. Loyalty = share of a party's voters backing its own candidate; the rest cross "
                "over. Blue = Democrat ahead, red = Republican ahead.")
        gc = st.columns(3)
        base_loy = gc[0].slider("Party loyalty", 70, 100, 90, 1, format="%d%%", key="grid_loy") / 100
        loy_gap = gc[1].slider("Loyalty gap for the 'higher loyalty' columns", 1, 15, 5, 1,
                               format="%d pp", key="grid_gap") / 100
        excl = gc[2].slider("Excluded (roll-off)", 3.0, 6.0, 5.0, 0.5, format="%.1f%%", key="grid_excl",
                            help="Share of returned ballots that won't record a vote in this race (skipped, "
                                 "blank or rejected). June 2026 primary: ~6.2% of SD4 ballots cast had no "
                                 "SD4 vote; a two-candidate general usually runs lower.") / 100
        lo_ = base_loy - loy_gap
        grid_cols = [
            (f"{base_loy:.0%} D / {base_loy:.0%} R", "Equal loyalty", base_loy, base_loy),
            (f"{base_loy:.0%} D / {lo_:.0%} R", "Higher Dem loyalty", base_loy, lo_),
            (f"{lo_:.0%} D / {base_loy:.0%} R", "Higher Rep loyalty", lo_, base_loy),
        ]
        grid_rows = [("Dem +10", 0.55), ("Dem +5", 0.525), ("Even split", 0.50), ("Rep +5", 0.475), ("Rep +10", 0.45)]

        def _margin(f_, l_, s_):
            # Dem votes − Rep votes: D·f + R·(1−l) + N·s − [R·l + D·(1−f) + N·(1−s)]
            # Excluded ballots are taken equally from every party, as in the Nevada Independent's table
            return (1 - excl) * (_d * (2 * f_ - 1) - _r * (2 * l_ - 1) + _n * (2 * s_ - 1))

        hdr_bg = ["#ececea", "#dbeafe", "#fde2e2"]
        html = ['<table style="width:100%;border-collapse:collapse;font-family:\'IBM Plex Sans\',sans-serif">',
                '<tr><th style="background:#14284b;color:white;text-align:left;padding:10px 12px">O/NPP split</th>']
        for (h, sub_, _, _), bg in zip(grid_cols, hdr_bg):
            html.append(f'<th style="background:{bg};text-align:left;padding:10px 12px">{h}'
                        f'<div style="font-weight:400;font-size:0.75rem;color:#666">{sub_}</div></th>')
        html.append("</tr>")
        for k_, (lab, s_) in enumerate(grid_rows):
            zebra = "#f7f7f5" if k_ % 2 else "white"
            html.append(f'<tr style="background:{zebra}"><td style="padding:8px 12px;border-top:1px solid #e0e0e0">'
                        f'<b>{lab}</b><div style="font-size:0.75rem;color:#666">{s_:.1%} / {1 - s_:.1%} (D / R)'
                        f'</div></td>')
            for _, _, f_, l_ in grid_cols:
                m_ = _margin(f_, l_, s_)
                colr = D.PARTY_COLORS["D"] if m_ >= 0 else D.PARTY_COLORS["R"]
                html.append(f'<td style="padding:8px 12px;border-top:1px solid #e0e0e0;text-align:center;'
                            f'font-family:\'IBM Plex Mono\',monospace;font-size:1.35rem;font-weight:600;'
                            f'color:{colr}">{m_:+,.0f}</td>')
            html.append("</tr>")
        html.append("</table>")
        st.markdown("".join(html), unsafe_allow_html=True)
        st.caption(f"Built from {_d:,} D, {_r:,} R and {_n:,} O/NPP ballots returned, less {excl:.1%} "
                   f"roll-off ({(1 - excl) * (_d + _r + _n):,.0f} modelled votes). Margin = (1 − excluded) × "
                   "[D·(2·D loyalty − 1) − R·(2·R loyalty − 1) + O/NPP·(2·Dem share − 1)]. "
                   "Each 5-point shift in the O/NPP split moves the margin by "
                   f"{(1 - excl) * 0.05 * _n:,.0f} votes. Roll-off: in the June 2026 primary about 6.2% of SD4 "
                   "ballots had no vote in this race (263,941 race votes of ~281,500 ballots); a two-candidate "
                   "general usually runs lower, hence 3–6%.")

    if ct_latest.empty:
        st.info("No Optiq crosstab data available.")
    else:
        section("Who hasn't returned yet",
                "Unreturned ballots by group, and how that pool compares to who has already voted. "
                "Gap = share of outstanding − share of returns: positive groups make up more of what's "
                "left than of what's in.")
        for dim in ["party", "age", "ethnicity"]:
            s = A.label_codes(A.group_summary(ct_latest, [dim]), [dim])
            s = s.set_index(dim).reindex([v for v in D.DIM_LABELS[dim].values() if v in set(s[dim])]).reset_index()
            s["gap"] = s["share_outstanding"] - s["share_returned"]
            s = pct_cols(s, ["rate", "share_returned", "share_outstanding", "gap"])
            st.markdown(f"**{D.DIM_NAMES[dim]}**")
            show_df(s[[dim, "issued", "returned", "rate", "outstanding",
                            "share_returned", "share_outstanding", "gap"]],
                         hide_index=True, width="stretch",
                         column_config={dim: D.DIM_NAMES[dim], "issued": INT("Issued"),
                                        "returned": INT("Returned"), "rate": pct_col("Return rate"),
                                        "outstanding": INT("Outstanding"),
                                        "share_returned": pct_col("Share of returns"),
                                        "share_outstanding": pct_col("Share of outstanding"),
                                        "gap": pct_col("Gap", True)})

        # Scenario defaults: SD4 forecast (default) or 2022 actuals
        issued_by_cty = county_t.set_index("county")["issued"]
        rates22 = A.baseline_2022_rates(base, issued_by_cty, {c: v[0] for c, v in COUNTY_PARAMS.items()})
        ratesfc = A.forecast_vbm_rates(D.load_forecast_params(), issued_by_cty,
                                       {c: v[0] for c, v in COUNTY_PARAMS.items()},
                                       {c: v[1] for c, v in COUNTY_PARAMS.items()})
        ENDPOINTS = {"Last pre-election report": ("pre", "early vote"),
                     "Final VBM count": ("final", "final mail vote")}
        SOURCES = {"SD4 forecast": ratesfc, "2022 actual": rates22} if ratesfc else {"2022 actual": rates22}

        section("End-of-window scenario",
                "Pick an end-point and a return rate, and see what the electorate looks like when it's "
                "reached. The default rate comes from the SD4 forecast or from 2022; the party split is "
                "your assumption.")
        sc = st.columns([1, 2])
        with sc[0]:
            target = st.radio("End-point", list(ENDPOINTS), horizontal=True,
                              help="Last pre-election report: the early vote, matching the 2022 pace "
                                   "baseline. Final VBM count: adds ballots that arrive or are processed "
                                   "after the last report.")
            source = st.radio("Default from", list(SOURCES), horizontal=True,
                              help="SD4 forecast: the forecast's county turnout × each county's VBM share "
                                   "of votes and pre-election share of VBM (vbm_turnout.py). "
                                   "2022 actual: 2022 SD4 VBM returns.")
            ekey, t_noun = ENDPOINTS[target]
            t_rate = SOURCES[source][ekey]
            uniform = st.slider("Return rate (all parties)", 0.10, 0.90, round(float(t_rate), 2), 0.01,
                                key=f"rate_{target}_{source}",
                                help=f"Default {t_rate:.1%} from {source}.")
            alt = " · ".join(f"{k}: {v[ekey]:.1%}" for k, v in SOURCES.items())
            st.caption(f"{target}: {alt}")
            differ = st.toggle("Set party rates separately", value=False,
                               help="With one rate for every party the end-point mix equals the mix of "
                                    "ballots issued; only different party rates move the margin.")
            if differ:
                rates = {p: st.slider(f"{D.PARTY_LABELS[p]}", 0.10, 0.95, uniform, 0.01,
                                      key=f"fr_{target}_{source}_{p}")
                         for p in D.PARTIES}
            else:
                rates = {p: uniform for p in D.PARTIES}
        proj_p = A.project_final(ct_latest, rates, "party").set_index("party")
        with sc[1]:
            tot_final = proj_p["proj_final"].sum()
            d_, r_ = proj_p.loc["D", "proj_final"], proj_p.loc["R", "proj_final"]
            k = st.columns(3)
            with k[0]:
                card(f"Projected {t_noun}", f"{tot_final:,.0f}",
                     f"{tot_final - proj_p['returned'].sum():,.0f} still to come")
            with k[1]:
                card(f"D−R margin, {t_noun}", pp((d_ - r_) / tot_final),
                     f"returned so far: {pp((proj_p.loc['D','returned'] - proj_p.loc['R','returned']) / proj_p['returned'].sum())}")
            with k[2]:
                card("D−R among ballots to come", pp(
                    (proj_p.loc["D", "still_to_come"] - proj_p.loc["R", "still_to_come"])
                    / proj_p["still_to_come"].sum()) if proj_p["still_to_come"].sum() else "—",
                     "registration-based, not vote share")
        with st.expander("Where the defaults come from"):
            bc = rates22["by_county"].rename(columns={"pre": "pre22", "final": "final22"})
            if ratesfc:
                bc = bc.join(ratesfc["by_county"][["pre", "final", "pre_vbm"]].rename(
                    columns={"pre": "prefc", "final": "finalfc", "pre_vbm": "prefc_n"}))
            bc = pct_cols(bc.reset_index(), [c for c in ("prefc", "finalfc", "pre22", "final22") if c in bc])
            show_df(bc, hide_index=True, width="stretch", column_config={
                "county": "County",
                "prefc_n": st.column_config.NumberColumn("Forecast early vote", format="localized"),
                "prefc": pct_col("Forecast pre-election"), "finalfc": pct_col("Forecast final"),
                "pre22": pct_col("2022 pre-election"), "final22": pct_col("2022 final (est.)")},
                column_order=["county", "prefc_n", "prefc", "finalfc", "pre22", "final22"])
            if ratesfc:
                st.caption(f"Forecast: {ratesfc['total']:,.0f} total ballots (forecast_params.json, structural "
                           f"model) → {ratesfc['final_vbm']:,.0f} final VBM → {ratesfc['pre_vbm']:,.0f} by "
                           "the last pre-election report, using each county's VBM share of votes and "
                           "pre-election share of VBM from vbm_turnout.py (avg of 2022 & 2024). Turnout "
                           "Explorer adjustments in the forecast app aren't reflected.")
            st.caption("2022: pre-election = VBM returned ÷ issued at the last report (vbm_2022_baseline.csv); "
                       "final = pre-election accepted ÷ the county's pre-election share of final VBM. "
                       "District figures weight counties by 2026 ballots issued. No party-level rates "
                       "exist for either source.")
        pcols = st.columns(2)
        for i, dim in enumerate(["party", "age"]):
            pr = A.project_final(ct_latest, rates, dim)
            pr = pr.set_index("label").reindex([v for v in D.DIM_LABELS[dim].values() if v in set(pr["label"])])
            fig = go.Figure()
            for name, col, colr in (("Returned so far", "share_returned", "#9ca3af"),
                                    ("Still to come", "share_still_to_come", "#d97706"),
                                    ("At end-point", "share_proj_final", "#1a6b3c")):
                fig.add_bar(x=pr.index, y=pr[col], name=name, marker_color=colr, hovertemplate="%{y:.1%}")
            fig.update_layout(**{**PLOT_LAYOUT, "legend": dict(orientation="h", y=-0.15, x=0), "margin": dict(l=10, r=10, t=40, b=10)},
                              barmode="group", height=320,
                              title=dict(text=f"Composition by {D.DIM_NAMES[dim].lower()}", font=dict(size=12)))
            fig.update_yaxes(tickformat=".0%")
            with pcols[i]:
                st.plotly_chart(fig, width="stretch")
        st.caption("Party rates are applied to every age/gender/ethnicity cell within the party, so the "
                   "age projection inherits today's party mix rather than modelling age-specific "
                   "propensity. A cell already above its assumed final rate is held at its current count.")

        section(f"Outstanding by county — {'Optiq' if use_optiq else 'SOS'}")
        if use_optiq:
            oc = oct_[["county", "tot_reg", "tot_ret", "outstanding", "rate"]].rename(
                columns={"tot_reg": "issued", "tot_ret": "returned"}).copy()
        else:
            oc = county_t[["county", "issued", "returned", "outstanding", "rate"]].copy()
        oc["share_outstanding"] = oc["outstanding"] / oc["outstanding"].sum()
        oc = pct_cols(oc, ["rate", "share_outstanding"]).sort_values("outstanding", ascending=False)
        show_df(oc, hide_index=True, width="stretch",
                     column_config={"county": "County",
                                    "issued": INT("Registered" if use_optiq else "Issued"),
                                    "returned": INT("Returned"),
                                    "outstanding": INT("Outstanding"), "rate": pct_col("Rate"),
                                    "share_outstanding": pct_col("% of outstanding")})

# ══════════════════════════════════════════════════════════════════════════════
# 2022 COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    if pdi22 is None:
        st.info("2022 data not found (pdi_2022_daily_sd4.csv).")
    else:
        cum22 = A.pdi_cumulative(pdi22)
        reg_m22 = A.registration_margin(reg22) if reg22 is not None else None
        s26 = A.series_2026(ct_hist, cp_hist)
        # Today's county-feed reading (live), if newer than anything saved
        if not cp_latest.empty:
            live_pt = A.party_by_days_out_cp(cp_latest.assign(snapshot_date=pd.Timestamp(cp_latest["snapshot_date"].max())))
            live_pt["source"] = "Optiq county feed"
            s26 = (pd.concat([live_pt, s26], ignore_index=True)
                     .drop_duplicates("days_before_eday").sort_values("days_before_eday", ascending=False))
        if not cp_latest.empty:
            g26 = cp_latest.groupby("party")["registered"].sum()
            reg_dr26 = (g26["D"] - g26["R"]) / g26.sum()
        else:
            reg_dr26 = (reg_counts.get("D", 0) - reg_counts.get("R", 0)) / reg_total if reg_total else np.nan
        now_out = int(s26["days_before_eday"].min()) if not s26.empty else None

        st.caption("2022 figures are SD4's 2022 general-election daily returns by party, read from PDI's tracker "
                   "chart (average of three readings; roughly ±300 per party per day, ±5% on the biggest days). "
                   "Days are counted back from Election Day; 2022 and 2026 elections are both on a Tuesday, so "
                   "weekdays line up. 2026 points come from Optiq.")

        fin = cum22.iloc[-1]
        k = st.columns(4)
        with k[0]:
            card("2022 early vote", f"{int(fin['cum_total']):,}", "returned by the day before Election Day")
        with k[1]:
            card("2022 final early D−R", pp(fin["dr_margin"]),
                 f"registration {pp(reg_m22['dr'])}" if reg_m22 else "")
        with k[2]:
            if now_out is not None:
                r_now = A.at_days_out(cum22, now_out)
                card(f"2022 share in by {now_out} days out", pct(r_now["frac_of_final"]) if r_now is not None else "—",
                     f"{int(r_now['cum_total']):,} ballots" if r_now is not None else "")
            else:
                card("2022 share in today", "—")
        with k[3]:
            if now_out is not None and reg_m22:
                latest26 = s26.iloc[-1]
                r_now = A.at_days_out(cum22, now_out)
                diff = (latest26["dr_margin"] - reg_dr26) - (r_now["dr_margin"] - reg_m22["dr"])
                card("2026 vs 2022, net of registration", pp(diff),
                     "positive = Democrats ahead of 2022's pace",
                     color=D.PARTY_COLORS["D"] if diff >= 0 else D.PARTY_COLORS["R"])

        # ── Daily returns ───────────────────────────────────────────────────
        section("Daily returns by days out", "Bars: 2022 by party.")
        fig = go.Figure()
        for p_ in ["D", "R", "O"]:
            fig.add_bar(x=pdi22["days_before_eday"], y=pdi22[p_], name=f"2022 {D.PARTY_LABELS[p_]}",
                        marker_color=D.PARTY_COLORS[p_], opacity=0.55, hovertemplate="%{y:,}")
        daily26_src = None
        if not live_cells.empty and live_cells["return_date"].notna().any():
            # Exact: Optiq's recorded return date for every ballot
            rd = (live_cells.dropna(subset=["return_date"]).groupby(["return_date", "party"])["voters"].sum()
                  .unstack("party", fill_value=0))
            rd = rd.reindex(pd.date_range(rd.index.min(), rd.index.max(), freq="D"), fill_value=0)
            rd["days_before_eday"] = (D.EDAY_2026 - rd.index).days
            for p_ in D.PARTIES:
                if p_ in rd:
                    fig.add_scatter(x=rd["days_before_eday"], y=rd[p_], name=f"2026 {D.PARTY_LABELS[p_]}",
                                    mode="lines+markers", line=dict(color=D.PARTY_COLORS[p_], width=2.5),
                                    hovertemplate="%{y:,}")
            daily26_src = "Lines: 2026 by party, from Optiq's recorded return date for each ballot."
        elif len(s26) >= 2:
            t26 = s26.sort_values("days_before_eday", ascending=False)
            gap = -t26["days_before_eday"].diff()
            per_day = t26["total"].diff() / gap
            fig.add_scatter(x=t26["days_before_eday"], y=per_day, name="2026 total / day", mode="lines+markers",
                            line=dict(color="#111", width=2.5), hovertemplate="%{y:,.0f}")
            daily26_src = ("Line: 2026 total per day from Optiq snapshots (live data unavailable) — snapshots are "
                           "dated when fetched and run about a day behind, so treat day-to-day wiggles loosely.")
        fig.update_layout(**PLOT_LAYOUT, barmode="stack", height=360)
        fig.update_xaxes(autorange="reversed", title="Days before Election Day", dtick=1)
        st.plotly_chart(fig, width="stretch")
        if daily26_src:
            st.caption(daily26_src)
        st.caption("In 2022, almost nothing was logged at weekends; Mondays and Wednesdays carried the big batches, "
                   "and the last Monday alone was 16% of the early vote.")

        # ── Cumulative ──────────────────────────────────────────────────────
        cc1, cc2 = st.columns(2)
        with cc1:
            section("Cumulative returns")
            fig = go.Figure()
            fig.add_scatter(x=cum22["days_before_eday"], y=cum22["cum_total"], name="2022", mode="lines",
                            line=dict(color="#9ca3af", width=3), hovertemplate="%{y:,}")
            if not s26.empty:
                fig.add_scatter(x=s26["days_before_eday"], y=s26["total"], name="2026", mode="lines+markers",
                                line=dict(color="#1a6b3c", width=3), hovertemplate="%{y:,}")
            fig.update_layout(**PLOT_LAYOUT, height=330)
            fig.update_xaxes(autorange="reversed", title="Days before Election Day")
            st.plotly_chart(fig, width="stretch")
            st.caption("Raw counts. 2026 has ~4% more ballots issued than 2022's estimate, so equal counts mean a "
                       "slightly lower rate.")
        with cc2:
            section("D−R among returns, minus registration")
            fig = go.Figure()
            if reg_m22:
                fig.add_scatter(x=cum22["days_before_eday"], y=(cum22["dr_margin"] - reg_m22["dr"]) * 100,
                                name="2022", mode="lines", line=dict(color="#9ca3af", width=3),
                                hovertemplate="%{y:+.1f} pp")
            if not s26.empty and np.isfinite(reg_dr26):
                fig.add_scatter(x=s26["days_before_eday"], y=(s26["dr_margin"] - reg_dr26) * 100, name="2026",
                                mode="lines+markers", line=dict(color="#1a6b3c", width=3),
                                customdata=s26["source"], hovertemplate="%{y:+.1f} pp · %{customdata}")
            fig.update_layout(**PLOT_LAYOUT, height=330)
            fig.update_xaxes(autorange="reversed", title="Days before Election Day")
            fig.update_yaxes(ticksuffix=" pp", zeroline=True, zerolinecolor="#999")
            fig.update_xaxes(range=[25, 0])
            st.plotly_chart(fig, width="stretch")
            st.caption("Above zero = returns more Democratic than that year's registration. Starts at 25 days out: "
                       "earlier points rest on a few hundred ballots. Points before ~22 days out are still noisy.")

        # ── Party shares over the window ────────────────────────────────────
        section("Party share of returns by days out",
                "Solid: 2022 running share. Markers: 2026. Dotted: each year's registration share.")
        fig = go.Figure()
        reg26_sh = ({p_: g26[p_] / g26.sum() for p_ in D.PARTIES} if not cp_latest.empty else
                    {p_: reg_counts.get(p_, 0) / reg_total for p_ in D.PARTIES})
        for p_ in D.PARTIES:
            colr = D.PARTY_COLORS[p_]
            fig.add_scatter(x=cum22["days_before_eday"], y=cum22[f"share_{p_}"], name=f"2022 {D.PARTY_LABELS[p_]}",
                            mode="lines", line=dict(color=colr, width=2), opacity=0.6, hovertemplate="%{y:.1%}")
            if not s26.empty:
                fig.add_scatter(x=s26["days_before_eday"], y=s26[f"share_{p_}"], name=f"2026 {D.PARTY_LABELS[p_]}",
                                mode="markers", marker=dict(color=colr, size=9, symbol="diamond"),
                                hovertemplate="%{y:.1%}")
            if reg_m22:
                fig.add_hline(y=reg_m22[p_], line=dict(color=colr, dash="dot", width=1), opacity=0.5)
        fig.update_layout(**PLOT_LAYOUT, height=360)
        fig.update_xaxes(autorange="reversed", title="Days before Election Day", range=[25, 0])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

        # ── Checkpoints table ───────────────────────────────────────────────
        section("Checkpoints")
        rows = []
        for d_ in (25, 22, 20, 19, 15, 12, 8, 5, 3, 1):
            r = A.at_days_out(cum22, d_)
            row = {"days": d_, "share_in": r["frac_of_final"], "ret22": r["cum_total"],
                   "dr22": r["dr_margin"],
                   "net22": r["dr_margin"] - reg_m22["dr"] if reg_m22 else np.nan}
            m = s26[s26["days_before_eday"] == d_] if not s26.empty else pd.DataFrame()
            if len(m):
                row.update({"ret26": m["total"].iloc[0], "dr26": m["dr_margin"].iloc[0],
                            "net26": m["dr_margin"].iloc[0] - reg_dr26})
            rows.append(row)
        ck = pd.DataFrame(rows)
        for c in ("ret26", "dr26", "net26"):
            if c not in ck:
                ck[c] = np.nan
            ck[c] = pd.to_numeric(ck[c], errors="coerce").astype(float)
        ck = pct_cols(ck, ["share_in", "dr22", "net22", "dr26", "net26"])
        show_df(ck[["days", "share_in", "ret22", "dr22", "net22", "ret26", "dr26", "net26"]],
                     hide_index=True, width="stretch",
                     signed=['dr22', 'net22', 'dr26', 'net26'], column_config={"days": st.column_config.NumberColumn("Days out", format="%d"),
                                    "share_in": pct_col("2022: share of early vote in"),
                                    "ret22": INT("2022 returned"), "dr22": pct_col("2022 D−R", True),
                                    "net22": pct_col("2022 vs reg", True), "ret26": INT("2026 returned"),
                                    "dr26": pct_col("2026 D−R", True), "net26": pct_col("2026 vs reg", True)})

        # ── Registration by county ──────────────────────────────────────────
        if reg22 is not None:
            section("Registration by county, 2022 → 2026",
                    "2022: SOS 15-day report (Oct 24, 2022); split counties estimated (county total × SD4 fraction, "
                    "2026 SD4-portion party mix), so their change is left blank. 2026: Optiq. The nine whole "
                    "counties are 90% of registration and drive the district change.")
            rg = reg22.set_index("county")[["registered", "D", "R", "O"]].copy()
            rg["dr22"] = (rg["D"] - rg["R"]) / rg["registered"]
            rg["D22"], rg["R22"] = rg["D"] / rg["registered"], rg["R"] / rg["registered"]
            if not cp_latest.empty:
                w = cp_latest.pivot_table(index="county", columns="party", values="registered", aggfunc="sum")
                w["T"] = w.sum(axis=1)
                rg["reg26"] = w["T"]
                rg["D26"], rg["R26"] = w["D"] / w["T"], w["R"] / w["T"]
                rg["dr26"] = (w["D"] - w["R"]) / w["T"]
                rg["chg"] = rg["dr26"] - rg["dr22"]
            tot = {"registered": rg["registered"].sum(), "D": rg["D"].sum(), "R": rg["R"].sum()}
            trow = {"registered": tot["registered"], "D22": tot["D"] / tot["registered"],
                    "R22": tot["R"] / tot["registered"], "dr22": (tot["D"] - tot["R"]) / tot["registered"]}
            if "reg26" in rg:
                trow.update({"reg26": w["T"].sum(), "D26": w["D"].sum() / w["T"].sum(),
                             "R26": w["R"].sum() / w["T"].sum(), "dr26": (w["D"].sum() - w["R"].sum()) / w["T"].sum()})
                trow["chg"] = trow["dr26"] - trow["dr22"]
            rg = pd.concat([rg.reset_index(), pd.DataFrame([{"county": "SD4 total", **trow}])], ignore_index=True)
            cols = ["county", "registered", "D22", "R22", "dr22"] + (["reg26", "D26", "R26", "dr26", "chg"]
                                                                       if "reg26" in rg else [])
            split = rg["county"].isin(D.PARTIAL_COUNTIES)
            if "chg" in rg:
                rg.loc[split, "chg"] = np.nan   # 2022 party mix was borrowed from 2026, so change isn't measured
            rg.loc[split, "county"] = rg.loc[split, "county"] + " (est.)"
            rgv = pct_cols(rg, [c for c in ("D22", "R22", "dr22", "D26", "R26", "dr26", "chg") if c in rg])
            show_df(rgv[cols], hide_index=True, width="stretch", height=38 + 35 * len(rgv),
                         signed=['dr22', 'dr26', 'chg'], column_config={"county": "County", "registered": INT("2022 reg"),
                                        "D22": pct_col("2022 D"), "R22": pct_col("2022 R"),
                                        "dr22": pct_col("2022 D−R", True), "reg26": INT("2026 reg"),
                                        "D26": pct_col("2026 D"), "R26": pct_col("2026 R"),
                                        "dr26": pct_col("2026 D−R", True), "chg": pct_col("Change", True)})

        # ── 2022 daily table ────────────────────────────────────────────────
        with st.expander("2022 daily data"):
            dd = cum22[["days_before_eday", "date_2022", "D", "R", "O", "total", "cum_total", "frac_of_final",
                        "dr_margin"]].copy()
            dd = pct_cols(dd, ["frac_of_final", "dr_margin"])
            show_df(dd, hide_index=True, width="stretch",
                         signed=['dr_margin'], column_config={"days_before_eday": st.column_config.NumberColumn("Days out", format="%d"),
                                        "date_2022": "2022 date", "D": INT("Dem"), "R": INT("Rep"),
                                        "O": INT("O/NPP"), "total": INT("Total"), "cum_total": INT("Running total"),
                                        "frac_of_final": pct_col("Share in"), "dr_margin": pct_col("Running D−R", True)})

# ══════════════════════════════════════════════════════════════════════════════
# METHOD & DATA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    section("SOS vs Optiq — latest counts",
            "Each source's most recent figures, side by side, to show where the totals differ. "
            "Optiq's 'registered' is its VBM universe (SD4 voters sent a ballot); SOS 'issued' is ballots "
            "issued. Optiq figures are raw here — Alpine is not filled in from SOS.")
    _sl = sos[sos["snapshot_date"] == sos["snapshot_date"].max()].set_index("county")
    _sos_date = sos_latest_report
    if not cp_latest.empty:
        _ol = cp_latest.groupby("county").agg(o_reg=("registered", "sum"), o_ret=("returned", "sum"),
                                              o_last=("latest_return_date", "max"))
    else:
        _ol = pd.DataFrame(columns=["o_reg", "o_ret", "o_last"])
    cmp_ = pd.DataFrame({"county": D.SD4_COUNTIES}).set_index("county")
    cmp_["s_iss"] = _sl["vbm_issued"]
    cmp_["s_ret"] = _sl["vbm_returned"]
    cmp_ = cmp_.join(_ol)
    cmp_ = cmp_.reset_index()
    tot_ = cmp_[["s_iss", "s_ret", "o_reg", "o_ret"]].sum()
    tot_row = {"county": "SD4 total", **tot_.to_dict(), "o_last": cmp_["o_last"].max()}
    cmp_ = pd.concat([cmp_, pd.DataFrame([tot_row])], ignore_index=True)
    cmp_["d_iss"] = cmp_["o_reg"] - cmp_["s_iss"]
    cmp_["d_ret"] = cmp_["o_ret"] - cmp_["s_ret"]
    cmp_["d_ret_pct"] = (cmp_["d_ret"] / cmp_["s_ret"].where(cmp_["s_ret"] > 0)) * 100
    cmp_["s_rate"] = cmp_["s_ret"] / cmp_["s_iss"].where(cmp_["s_iss"] > 0) * 100
    cmp_["o_rate"] = cmp_["o_ret"] / cmp_["o_reg"].where(cmp_["o_reg"] > 0) * 100

    def _why(r):
        notes = []
        if r["county"] in D.PARTIAL_COUNTIES:
            notes.append("split county: SOS is county total × SD4 share")
        if r["county"] != "SD4 total":
            if pd.notna(r["o_last"]) and r["o_last"] < _sos_date - pd.Timedelta(days=1):
                notes.append(f"Optiq's last return {r['o_last']:%b %-d}")
            if (r["o_ret"] or 0) == 0 and (r["s_ret"] or 0) > 0:
                notes.append("Optiq shows no returns yet")
            elif pd.notna(r["d_ret_pct"]) and r["d_ret_pct"] > 0:
                notes.append("Optiq has newer returns" if _o_date > _sos_date else "Optiq ahead of SOS")
        return "; ".join(notes)
    _o_date = cp_latest["snapshot_date"].max() if not cp_latest.empty else _sos_date
    cmp_["why"] = cmp_.apply(_why, axis=1)
    _gap_days = (_o_date - _sos_date).days
    if _gap_days > 0:
        st.markdown(f"**Timing:** the latest SOS report is from {_sos_date:%b %-d}, {_gap_days} day"
                    f"{'s' if _gap_days != 1 else ''} older than Optiq ({_o_date:%b %-d}), so part of every "
                    "gap below is ballots Optiq has counted that SOS hasn't reported yet.")
    show_df(
        cmp_[["county", "s_iss", "o_reg", "d_iss", "s_ret", "o_ret", "d_ret", "d_ret_pct", "s_rate", "o_rate",
              "o_last", "why"]],
        hide_index=True, width="stretch", height=38 + 35 * len(cmp_),
        column_config={
            "county": "County",
            "s_iss": INT("SOS issued"), "o_reg": INT("Optiq registered"),
            "d_iss": st.column_config.NumberColumn("Δ issued", format="%+d", help="Optiq − SOS"),
            "s_ret": INT("SOS returned"), "o_ret": INT("Optiq returned"),
            "d_ret": st.column_config.NumberColumn("Δ returned", format="%+d", help="Optiq − SOS"),
            "d_ret_pct": st.column_config.NumberColumn("Δ returned %", format="%+.1f%%"),
            "s_rate": pct_col("SOS rate"), "o_rate": pct_col("Optiq rate"),
            "o_last": st.column_config.DateColumn("Optiq last return", format="MMM D"),
            "why": st.column_config.TextColumn("Likely reason", width="large"),
        })
    _ct_note = ""
    if not ct_latest.empty:
        _ct_iss, _ct_ret = int(ct_latest["issued"].sum()), int(ct_latest["returned"].sum())
        _ct_note = (f" Optiq crosstabs (the party × age × gender × ethnicity feed): {_ct_iss:,} issued, "
                    f"{_ct_ret:,} returned — vs {int(tot_['o_reg']):,} / {int(tot_['o_ret']):,} in its county feed.")
    st.caption(f"SOS report {_sos_date:%b %-d}; Optiq {'live' if cp_source == 'live' else 'saved snapshot'} "
               f"{cp_latest['snapshot_date'].max():%b %-d}." + _ct_note + " " + SOURCE_GAP_NOTE)

    section("Sources")
    st.markdown(f"""
- **SOS ballot status report** (`ballot_returns.csv`) — county VBM issued / returned / accepted, pulled every
  4 h by the *Update Ballot Returns* Action. {meta['n_snapshots']} distinct snapshots since
  {meta['first_snapshot']:%b %-d}.
- **Optiq crosstabs** (`optiq_crosstabs.csv`) — party × age × gender × ethnicity issued/returned, voter-file
  matched (~98% coverage). {ct_saved['snapshot_date'].nunique() if not ct_saved.empty else 0} distinct snapshots.
- **Optiq county × party** (`optiq_county_party.csv`) — county D/R/O registered and returned.
  {cp_saved['snapshot_date'].nunique() if not cp_saved.empty else 0} saved snapshots.
- **Optiq precincts** (`optiq_precincts.csv`, shapes in `sd4_precincts.geojson`) — precinct D/R/O registered and
  returned, read from the precinct map tiles on abev.civiq.vote. Saved only when each whole county's precinct
  total matches Optiq's county figure; split counties keep the precincts whose centre falls inside SD4.
- **2022 baseline** (`vbm_2022_baseline.csv`) — SD4-portion 2022 VBM issued/returned by county.
- **Registration** (`sd4_party_registration.csv`) — SOS report dated {reg.get('report_date', 'n/a')}.
""")
    section("Definitions")
    st.markdown("""
- **Return rate** — returned ÷ issued within a group.
- **Share of returns** — a group's returned ballots ÷ all returned ballots.
- **Representation** — share of returns − share of issued. Positive = over-represented among early returns.
- **D−R margin (returns)** — (D returned − R returned) ÷ all returned. Compare with the same margin in registration.
- **Velocity** — new returns ÷ days between two distinct reports.
""")
    section("Caveats")
    st.markdown(f"""
- **SOS vs Optiq totals differ.** {SOURCE_GAP_NOTE}
- **Stale snapshots are dropped.** If a feed re-publishes identical numbers on a new date, that date is removed so velocity isn't understated.
- **2022 pace baseline: {meta['baseline_kind']}.** The end-point is the last 2022 pre-election report
  ({pct(meta['rate_2022_baseline'])} district return rate), which is the right target for tracking the early vote.
  The path to it comes from SD4's 2022 daily returns by party, read from PDI's tracker chart
  (`pdi_2022_daily_sd4.csv`, average of three readings, roughly ±300 per party per day), unless the sidebar is set
  to an assumed shape. A county-level `vbm_2022_daily.csv` (`county, days_before_eday, vbm_returned`) would
  override both.
- **2022 registration** (`sd4_party_registration_2022.csv`) is the SOS 15-day report (Oct 24, 2022). Split
  counties use the county total × SD4 fraction with the 2026 SD4-portion party mix; whole counties are exact.
- **Split counties** use a fixed registration-share fraction for the SOS figures; Optiq counts are already district-specific.
- **Party ≠ vote.** Everything here is party registration of returned ballots, not candidate support.
""")
