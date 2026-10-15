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


def dim_color(dim: str, code: str, i: int) -> str:
    if dim == "party":
        return D.PARTY_COLORS.get(code, "#666")
    return SEQ_COLORS[i % len(SEQ_COLORS)]


# ── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def load_all():
    return {
        "sos": D.load_sos_returns(),
        "base": D.load_2022_baseline(),
        "daily22": D.load_2022_daily(),
        "pdi22": D.load_pdi_2022(),
        "reg22": D.load_registration_2022(),
        "reg": D.load_party_registration(),
        "ct": D.load_optiq_crosstabs(),
        "cp": D.load_optiq_county_party(),
    }


@st.cache_data(ttl=3600, show_spinner="Fetching live Optiq data…")
def load_live():
    return D.fetch_optiq_live(), D.fetch_optiq_county_party_live()


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

live_cells, live_cp = (pd.DataFrame(), pd.DataFrame())
if use_live:
    live_cells, live_cp = load_live()

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
ct_hist = ct_saved
if ct_source == "live" and (ct_saved.empty or today > ct_saved["snapshot_date"].max()):
    ct_hist = pd.concat([ct_saved, ct_latest], ignore_index=True)

if not live_cp.empty:
    cp_latest, cp_source = live_cp.assign(snapshot_date=today), "live"
else:
    cp_latest, cp_source = A.latest(cp_saved), "saved"
cp_hist = cp_saved
if cp_source == "live" and (cp_saved.empty or today > cp_saved["snapshot_date"].max()):
    cp_hist = pd.concat([cp_saved, cp_latest], ignore_index=True)

reg_counts = reg.get("counts", {})
reg_total = sum(reg_counts.values())

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 📬 SD4 Ballot Return Tracker")
if sos.empty:
    st.warning("No SOS ballot return data found (ballot_returns.csv).")
    st.stop()

county_t, meta = A.county_table(sos, base, daily22, curve_k, curve)
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
    f'<span class="src-badge">SOS data · {meta["snapshot"]:%b %-d} (E−{meta["days_left"]})</span>'
    + (f'<span class="src-badge {"live" if ct_source == "live" else ""}">'
       f'Optiq · {"live" if ct_source == "live" else f"{ct_date:%b %-d}"}</span>' if ct_date is not None else "")
)
st.markdown(badges, unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Counties", "Crosstabs", "Trends", "Outstanding", "2022 comparison",
                "Method & data"])

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    section("District — SOS ballot status report")
    c = st.columns(4)
    with c[0]:
        card("Ballots returned", f"{meta['returned']:,}", f"of {meta['issued']:,} issued")
    with c[1]:
        card("Return rate", pct(meta["rate"]), f"2022 last pre-election report: {pct(meta['rate_2022_baseline'])}")
    with c[2]:
        card("Returns / day", f"{meta['per_day']:,.0f}" if np.isfinite(meta["per_day"]) else "—",
             "over the latest reporting interval")
    with c[3]:
        r = meta["pace_ratio"]
        col = "#2e7d32" if r >= 1.10 else ("#d97706" if r >= 0.85 else "#b91c1c")
        card(f"vs 2022 · {pace_name}", f"{r:.0%}" if np.isfinite(r) else "—",
             f"SOS data is from E−{meta['days_left']} ({meta['snapshot']:%b %-d}); 2022 at E−{meta['days_left']}: "
             f"{pct(meta['rate_2022_equiv'])}",
             color=col)
    if daily22 is None:
        sens = A.pace_sensitivity(sos, base)
        parts = " · ".join(f"k={r.k:g}: {r.pace_ratio:.0%}" for r in sens.itertuples())
        if curve is not None:
            st.caption(f"The 2022 path uses SD4's real 2022 daily returns (PDI, read from their chart), aligned by "
                       f"days before Election Day — both elections fall on a Tuesday, so weekdays line up. One "
                       f"district curve is applied to every county. For comparison, assumed shapes give: {parts}.")
        else:
            st.caption(f"⚠️ The 2022 path to its pre-election end-point is assumed here, not observed. "
                       f"District pace under different shapes — {parts}. Switch the sidebar to "
                       "\"2022 daily returns\" to use the real 2022 curve.")

    if not ct_latest.empty:
        section("By party — Optiq voter-file match",
                "Share of returns vs registration share is the like-for-like comparison: "
                "a party ahead of its registration share is banking a larger slice of the early vote.")
        ps = A.group_summary(ct_latest, ["party"]).set_index("party")
        cols = st.columns(3)
        for i, p in enumerate(D.PARTIES):
            if p not in ps.index:
                continue
            r = ps.loc[p]
            reg_share = reg_counts.get(p, 0) / reg_total if reg_total else np.nan
            with cols[i]:
                card(D.PARTY_LABELS[p], f"{int(r['returned']):,}",
                     f"{pct(r['rate'])} returned · {pct(r['share_returned'])} of returns "
                     f"vs {pct(reg_share)} of reg ({pp(r['share_returned'] - reg_share)})",
                     color=D.PARTY_COLORS[p])
        d_ret, r_ret = ps.loc["D", "returned"], ps.loc["R", "returned"]
        tot = ps["returned"].sum()
        reg_margin = (reg_counts.get("D", 0) - reg_counts.get("R", 0)) / reg_total if reg_total else np.nan
        st.caption(f"D−R margin among returned ballots: **{pp((d_ret - r_ret) / tot)}** "
                   f"(registration: {pp(reg_margin)}) · raw D−R: **{int(d_ret - r_ret):+,}**")
        st.caption(
            f"ℹ️ **SOS and Optiq totals don't match exactly** — SOS: {meta['returned']:,} returned of "
            f"{meta['issued']:,} issued ({meta['snapshot']:%b %-d}); Optiq: {int(ct_latest['returned'].sum()):,} "
            f"of {int(ct_latest['issued'].sum()):,} "
            f"({'live' if ct_source == 'live' else f'{ct_date:%b %-d}'}). " + SOURCE_GAP_NOTE)

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
            ret_src, ret_date = "Optiq county feed", pd.Timestamp(cp_latest["snapshot_date"].max())
            # Optiq runs about a day behind: date the data by its latest recorded return
            if "latest_return_date" in cp_latest and cp_latest["latest_return_date"].notna().any():
                ret_date = min(ret_date, pd.Timestamp(cp_latest["latest_return_date"].max()))
        else:
            ret26 = A.group_summary(ct_latest, ["party"]).set_index("party")
            ret_src, ret_date = "Optiq crosstabs", pd.Timestamp(ct_date)
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
            st.dataframe(pct_cols(cmp, ["D", "R", "O", "dr"]), hide_index=True, width="stretch",
                         column_config={"D": pct_col("Dem"), "R": pct_col("Rep"), "O": pct_col("O/NPP"),
                                        "dr": pct_col("D−R", True)})
            st.caption(f"2022 returns are estimates read from PDI's chart (about ±0.5pp on these shares); "
                       f"{r22['frac_of_final']:.0%} of 2022's early vote was in by this point. "
                       "See the 2022 comparison tab for the full picture.")

    section("Cumulative returns")
    ss = A.sos_series(sos)
    dist = ss[ss["county"] == "SD4 total"]
    fig = go.Figure()
    fig.add_bar(x=dist["snapshot_date"], y=dist["per_day"], name="Returns / day",
                marker_color="#c3dece", yaxis="y2", hovertemplate="%{y:,.0f}/day")
    fig.add_scatter(x=dist["snapshot_date"], y=dist["vbm_returned"], name="Cumulative returned",
                    mode="lines+markers", line=dict(color="#1a6b3c", width=2.5),
                    hovertemplate="%{y:,}")
    fig.update_layout(**PLOT_LAYOUT, height=330,
                      yaxis2=dict(overlaying="y", side="right", showgrid=False, title=None))
    st.plotly_chart(fig, width="stretch")
    st.caption("Snapshots where SOS figures didn't change (weekends, no new report) are dropped; "
               "per-day velocity divides by the real gap between reports.")

# ══════════════════════════════════════════════════════════════════════════════
# COUNTIES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    use_optiq = not cp_hist.empty
    if use_optiq:
        oct_, ometa = A.optiq_county_table(cp_hist, base, daily22, curve_k, curve)
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
        show["pace_ratio"] = show["pace_ratio"] * 100
        st.dataframe(
            show[["county", "tot_reg", "tot_ret", "rate", "last_drop", "last_drop_date", "rate_2022_equiv",
                  "pace_ratio", "pace", "share_of_district_returns", "share_of_district_reg", "outstanding"]],
            hide_index=True, width="stretch", height=38 + 35 * len(show),
            column_config={
                "county": "County", "tot_reg": INT("Registered"), "tot_ret": INT("Returned"),
                "rate": pct_col("Rate"),
                **DROP_COLS("Optiq snapshot"),
                "rate_2022_equiv": pct_col(f"2022 est. ({pace_short})"),
                "pace_ratio": st.column_config.NumberColumn(f"vs 2022 ({pace_short})", format="%.0f%%"),
                "pace": f"Pace ({pace_short})",
                "share_of_district_returns": pct_col("Share of returns"),
                "share_of_district_reg": pct_col("Share of reg."), "outstanding": INT("Not yet returned"),
            },
        )
        _cap = ("Drop date is Optiq's most recent ballot return date for the county. "
                "Registered is Optiq's VBM universe (sums to the same total as the crosstab 'issued'). ")
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
                 + [c for p_ in D.PARTIES for c in (f"{p_}_ret", f"{p_}_share_ret")]
                 + ["raw_dr", "dr_margin_ret", "dr_margin_reg", "margin_shift"])
        cfg2 = {"county": "County",
                "raw_dr": st.column_config.NumberColumn("D−R returned", format="%+d",
                                                        help="Democratic minus Republican ballots returned."),
                "dr_margin_ret": pct_col("D−R returns", True),
                "dr_margin_reg": pct_col("D−R reg", True),
                "margin_shift": pct_col("Shift vs reg", True)}
        short = {"D": "D", "R": "R", "O": "O/NPP"}
        for p_ in D.PARTIES:
            cfg2[f"{p_}_ret"] = INT(f"{short[p_]} ret")
            cfg2[f"{p_}_share_ret"] = pct_col(f"{short[p_]} ret %")
        st.dataframe(v[cols2], hide_index=True, width="stretch", height=38 + 35 * len(v),
                     column_config=cfg2)
    else:
        cp_tab = pd.DataFrame()
        oct_ = None
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
        st.dataframe(
            show[["county", "issued", "returned", "rate", "last_drop", "last_drop_date", "rate_2022_equiv",
                  "pace_ratio", "pace", "share_of_district_returns", "share_of_district_issued", "outstanding"]],
            hide_index=True, width="stretch", height=38 + 35 * len(show),
            column_config={
                "county": "County", "issued": INT("Issued"), "returned": INT("Returned"),
                "rate": pct_col("Rate"),
                **DROP_COLS("SOS report"),
                "rate_2022_equiv": pct_col(f"2022 est. ({pace_short})"),
                "pace_ratio": st.column_config.NumberColumn(f"vs 2022 ({pace_short})", format="%.0f%%"),
                "pace": f"Pace ({pace_short})", "share_of_district_returns": pct_col("Share of returns"),
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
            st.dataframe(top, hide_index=True, width="stretch",
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
# CROSSTABS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
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
                st.dataframe(
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
        st.dataframe(gdf, hide_index=True, width="stretch", height=38 + 35 * len(gdf), column_config={
            "D rate": pct_col("D rate"), "R rate": pct_col("R rate"), "O rate": pct_col("O/NPP rate"),
            "rate_gap": pct_col("D−R rate gap", True), "margin_ret": pct_col("D−R (returns)", True),
            "margin_iss": pct_col("D−R (issued)", True), "shift": pct_col("Shift", True)})

        section("Cell explorer", "Every party × age × gender × ethnicity cell, sortable.")
        cells = A.label_codes(A.group_summary(cx, ["party", "age", "gender", "ethnicity"]),
                              ["party", "age", "gender", "ethnicity"])
        min_n = st.slider("Minimum ballots issued", 0, 5000, 500, 100)
        cells = cells[cells["issued"] >= min_n].sort_values("representation", ascending=False)
        cells = pct_cols(cells, ["rate", "share_issued", "share_returned", "representation"])
        st.dataframe(cells[["party", "age", "gender", "ethnicity", "issued", "returned", "rate",
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
with tabs[3]:
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
with tabs[4]:
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
            st.dataframe(s[[dim, "issued", "returned", "rate", "outstanding",
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
            st.dataframe(bc, hide_index=True, width="stretch", column_config={
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

        section("Outstanding by county — SOS")
        oc = county_t[["county", "issued", "returned", "outstanding", "rate"]].copy()
        oc["share_outstanding"] = oc["outstanding"] / oc["outstanding"].sum()
        oc = pct_cols(oc, ["rate", "share_outstanding"]).sort_values("outstanding", ascending=False)
        st.dataframe(oc, hide_index=True, width="stretch",
                     column_config={"county": "County", "issued": INT("Issued"), "returned": INT("Returned"),
                                    "outstanding": INT("Outstanding"), "rate": pct_col("Rate"),
                                    "share_outstanding": pct_col("% of outstanding")})

# ══════════════════════════════════════════════════════════════════════════════
# 2022 COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
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
        st.dataframe(ck[["days", "share_in", "ret22", "dr22", "net22", "ret26", "dr26", "net26"]],
                     hide_index=True, width="stretch",
                     column_config={"days": st.column_config.NumberColumn("Days out", format="%d"),
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
            st.dataframe(rgv[cols], hide_index=True, width="stretch", height=38 + 35 * len(rgv),
                         column_config={"county": "County", "registered": INT("2022 reg"),
                                        "D22": pct_col("2022 D"), "R22": pct_col("2022 R"),
                                        "dr22": pct_col("2022 D−R", True), "reg26": INT("2026 reg"),
                                        "D26": pct_col("2026 D"), "R26": pct_col("2026 R"),
                                        "dr26": pct_col("2026 D−R", True), "chg": pct_col("Change", True)})

        # ── 2022 daily table ────────────────────────────────────────────────
        with st.expander("2022 daily data"):
            dd = cum22[["days_before_eday", "date_2022", "D", "R", "O", "total", "cum_total", "frac_of_final",
                        "dr_margin"]].copy()
            dd = pct_cols(dd, ["frac_of_final", "dr_margin"])
            st.dataframe(dd, hide_index=True, width="stretch",
                         column_config={"days_before_eday": st.column_config.NumberColumn("Days out", format="%d"),
                                        "date_2022": "2022 date", "D": INT("Dem"), "R": INT("Rep"),
                                        "O": INT("O/NPP"), "total": INT("Total"), "cum_total": INT("Running total"),
                                        "frac_of_final": pct_col("Share in"), "dr_margin": pct_col("Running D−R", True)})

# ══════════════════════════════════════════════════════════════════════════════
# METHOD & DATA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    section("Sources")
    st.markdown(f"""
- **SOS ballot status report** (`ballot_returns.csv`) — county VBM issued / returned / accepted, pulled every
  4 h by the *Update Ballot Returns* Action. {meta['n_snapshots']} distinct snapshots since
  {meta['first_snapshot']:%b %-d}.
- **Optiq crosstabs** (`optiq_crosstabs.csv`) — party × age × gender × ethnicity issued/returned, voter-file
  matched (~98% coverage). {ct_saved['snapshot_date'].nunique() if not ct_saved.empty else 0} distinct snapshots.
- **Optiq county × party** (`optiq_county_party.csv`) — county D/R/O registered and returned.
  {cp_saved['snapshot_date'].nunique() if not cp_saved.empty else 0} saved snapshots.
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
