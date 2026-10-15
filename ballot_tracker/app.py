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

import analysis as A
import data as D

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

with st.sidebar:
    st.markdown("### 📬 SD4 Ballot Returns")
    use_live = st.toggle("Use live Optiq data", value=True,
                         help="Pull the Optiq API directly (cached 1 h). Falls back to the saved "
                              "snapshots if the API can't be reached.")
    st.markdown("**2022 pace curve**")
    curve_k = st.slider(
        "Shape (k)", 1.0, 3.0, 1.0, 0.1,
        disabled=daily22 is not None,
        help="Share of the 2022 early vote assumed in by day t = (t / window) ^ k. "
             "1 = linear (default). Higher = more back-loaded: fewer ballots early, more late. "
             "There is no 2022 day-by-day data to fit k, so treat it as a sensitivity check.",
    )
    if daily22 is not None:
        st.caption("Using vbm_2022_daily.csv — curve shape not needed.")
    if st.button("↻ Refresh data"):
        st.cache_data.clear()
        st.rerun()

live_cells, live_cp = (pd.DataFrame(), pd.DataFrame())
if use_live:
    live_cells, live_cp = load_live()

today = pd.Timestamp.today().normalize()
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
    cp_latest, cp_source = live_cp, "live"
else:
    cp_latest, cp_source = A.latest(cp_saved), "saved"

reg_counts = reg.get("counts", {})
reg_total = sum(reg_counts.values())

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 📬 SD4 Ballot Return Tracker")
if sos.empty:
    st.warning("No SOS ballot return data found (ballot_returns.csv).")
    st.stop()

county_t, meta = A.county_table(sos, base, daily22, curve_k)
ct_date = (ct_latest["snapshot_date"].max() if not ct_latest.empty else None)
badges = (
    f'<span class="src-badge">SOS · {meta["snapshot"]:%b %-d}</span>'
    + (f'<span class="src-badge {"live" if ct_source == "live" else ""}">'
       f'Optiq · {"live" if ct_source == "live" else f"{ct_date:%b %-d}"}</span>' if ct_date is not None else "")
    + f'<span class="src-badge">{meta["days_left"]} days to Nov 3</span>'
)
st.markdown(badges, unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Counties", "Crosstabs", "Trends", "Outstanding", "Method & data"])

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
        card("vs 2022 pace", f"{r:.0%}" if np.isfinite(r) else "—",
             f"2022 at E−{meta['days_left']}: {pct(meta['rate_2022_equiv'])} ({meta['baseline_kind']})",
             color=col)
    if daily22 is None:
        sens = A.pace_sensitivity(sos, base)
        parts = " · ".join(f"k={r.k:g}: {r.pace_ratio:.0%}" for r in sens.itertuples())
        st.caption(f"⚠️ The 2022 path to its pre-election end-point is assumed, not observed "
                   f"(shape set in the sidebar). District pace under different shapes — {parts}. "
                   "Early in the window the shape dominates the answer; lean on the party and county "
                   "share comparisons, which don't depend on it.")

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
    section("County returns — SOS",
            "Split counties (Madera, Merced, Nevada, Placer) are scaled to the SD4 portion "
            "using partial_county_fractions.csv, which assumes the in-district share of ballots "
            "matches the registration share.")
    ctab = county_t.copy()
    ctab["pace"] = ctab["pace_ratio"].map(A.pace_label)
    show = pct_cols(ctab, ["rate", "rate_2022_equiv", "share_of_district_returns", "share_of_district_issued"])
    show["pace_ratio"] = show["pace_ratio"] * 100
    st.dataframe(
        show[["county", "issued", "returned", "rate", "per_day", "rate_2022_equiv", "pace_ratio", "pace",
              "share_of_district_returns", "share_of_district_issued", "outstanding"]],
        hide_index=True, width="stretch",
        column_config={
            "county": "County", "issued": INT("Issued"), "returned": INT("Returned"),
            "rate": pct_col("Rate"), "per_day": st.column_config.NumberColumn("Per day", format="%.0f"),
            "rate_2022_equiv": pct_col("2022 equiv."),
            "pace_ratio": st.column_config.NumberColumn("vs 2022", format="%.0f%%"),
            "pace": "Pace", "share_of_district_returns": pct_col("Share of returns"),
            "share_of_district_issued": pct_col("Share of issued"), "outstanding": INT("Outstanding"),
        },
        height=38 + 35 * len(show),
    )

    mcol, tcol = st.columns([3, 2])
    cp_tab = A.county_party_table(cp_latest)
    metric_opts = {
        "Return rate (SOS)": (dict(zip(county_t["county"], county_t["rate"])), False, "{:.1%}", "Greens"),
        "vs 2022 pace (SOS)": (dict(zip(county_t["county"], county_t["pace_ratio"] - 1)), True, "{:+.0%}", "RdYlGn"),
        "Share of district returns − share of issued": (
            dict(zip(county_t["county"], county_t["share_of_district_returns"] - county_t["share_of_district_issued"])),
            True, "{:+.1%}", "PiYG"),
    }
    if not cp_tab.empty:
        cpi = cp_tab.set_index("county")
        metric_opts.update({
            "Dem return rate": (cpi["D_rate"].to_dict(), False, "{:.1%}", "Blues"),
            "Rep return rate": (cpi["R_rate"].to_dict(), False, "{:.1%}", "Reds"),
            "D−R margin among returns": (cpi["dr_margin_ret"].to_dict(), True, "{:+.1%}", "RdBu"),
            "Margin shift vs registration": (cpi["margin_shift"].to_dict(), True, "{:+.1%}", "RdBu"),
            "D − R return-rate gap": (cpi["turnout_gap"].to_dict(), True, "{:+.1%}", "RdBu"),
            "Raw D − R returned": (cpi["raw_dr"].astype(float).to_dict(), True, "{:+,.0f}", "RdBu"),
        })
    with mcol:
        sel = st.selectbox("Map metric", list(metric_opts))
        vals, div, fmt, cs = metric_opts[sel]
        vals = {cn: float(vals.get(cn, np.nan)) for cn in D.SD4_COUNTIES}
        county_map(vals, sel, div, fmt, cs)
    with tcol:
        st.markdown("**Biggest daily contributors**")
        top = county_t.sort_values("per_day", ascending=False)[["county", "per_day", "returned"]].head(6)
        st.dataframe(top, hide_index=True, width="stretch",
                     column_config={"county": "County",
                                    "per_day": st.column_config.NumberColumn("Per day", format="%.0f"),
                                    "returned": INT("Returned")})
        st.caption("Stanislaus is ~47% of SD4's issued ballots, so its return speed moves the "
                   "district number more than every small county combined.")

    section("County × party — Optiq",
            f"Source: {'live API' if cp_source == 'live' else 'saved snapshot'}. "
            "Margin shift = D−R margin among returns minus D−R margin in registration "
            "(positive = Democrats banking a bigger share than their registration).")
    if cp_tab.empty:
        st.info("County × party data unavailable — the Optiq API couldn't be reached and no "
                "optiq_county_party.csv snapshots exist yet. The updated GitHub Action will start saving them.")
    else:
        v = pct_cols(cp_tab, ["D_rate", "R_rate", "O_rate", "D_share_ret", "R_share_ret", "O_share_ret",
                              "dr_margin_ret", "dr_margin_reg", "margin_shift", "turnout_gap"])
        st.dataframe(
            v[["county", "D_ret", "R_ret", "O_ret", "D_rate", "R_rate", "O_rate",
               "D_share_ret", "R_share_ret", "O_share_ret", "dr_margin_ret", "dr_margin_reg", "margin_shift",
               "turnout_gap"]],
            hide_index=True, width="stretch",
            column_config={
                "county": "County", "D_ret": INT("D ret"), "R_ret": INT("R ret"), "O_ret": INT("O/NPP ret"),
                "D_rate": pct_col("D rate"), "R_rate": pct_col("R rate"), "O_rate": pct_col("O/NPP rate"),
                "D_share_ret": pct_col("D share"), "R_share_ret": pct_col("R share"),
                "O_share_ret": pct_col("O/NPP share"),
                "dr_margin_ret": pct_col("D−R (returns)", True), "dr_margin_reg": pct_col("D−R (reg)", True),
                "margin_shift": pct_col("Shift", True), "turnout_gap": pct_col("D−R rate gap", True),
            },
        )

    section("County return rate over time — SOS")
    ss = A.sos_series(sos)
    pick = st.multiselect("Counties", D.SD4_COUNTIES + ["SD4 total"],
                          default=["SD4 total", "Stanislaus", "El Dorado", "Madera", "Amador"])
    fig = go.Figure()
    for i, cn in enumerate(pick):
        s = ss[ss["county"] == cn]
        fig.add_scatter(x=s["snapshot_date"], y=s["rate"], name=cn, mode="lines+markers",
                        line=dict(width=3 if cn == "SD4 total" else 1.8,
                                  color=COUNTY_COLORS[cn]),
                        hovertemplate="%{y:.1%}")
    fig.update_layout(**PLOT_LAYOUT, height=340)
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    if not cp_saved.empty and cp_saved["snapshot_date"].nunique() >= 2:
        section("County D−R margin among returns over time — Optiq")
        frames = []
        for d_, g in cp_saved.groupby("snapshot_date"):
            t = A.county_party_table(g)
            t["snapshot_date"] = d_
            frames.append(t)
        cps = pd.concat(frames)
        fig = go.Figure()
        for i, cn in enumerate(D.SD4_COUNTIES):
            s = cps[cps["county"] == cn]
            fig.add_scatter(x=s["snapshot_date"], y=s["dr_margin_ret"], name=cn, mode="lines",
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
                "Unreturned ballots by group, and how that pool compares to who has already voted.")
        oc = st.columns(3)
        for i, dim in enumerate(["party", "age", "ethnicity"]):
            s = A.label_codes(A.group_summary(ct_latest, [dim]), [dim])
            s = s.set_index(dim).reindex([v for v in D.DIM_LABELS[dim].values() if v in set(s[dim])]).reset_index()
            s["gap"] = s["share_outstanding"] - s["share_returned"]
            s = pct_cols(s, ["share_returned", "share_outstanding", "gap"])
            with oc[i]:
                st.markdown(f"**{D.DIM_NAMES[dim]}**")
                st.dataframe(s[[dim, "outstanding", "share_returned", "share_outstanding", "gap"]],
                             hide_index=True, width="stretch",
                             column_config={dim: D.DIM_NAMES[dim], "outstanding": INT("Out."),
                                            "share_returned": pct_col("% of ret."),
                                            "share_outstanding": pct_col("% of out."),
                                            "gap": pct_col("Gap", True)})

        section("Final VBM electorate scenario",
                "Assume a final return rate for each party and project the composition of the "
                "finished mail electorate. These are assumptions you set, not model output.")
        sc = st.columns([1, 2])
        with sc[0]:
            uniform = st.slider("Final return rate (all parties)", 0.30, 0.85, 0.55, 0.01)
            differ = st.toggle("Set party rates separately", value=False)
            if differ:
                rates = {p: st.slider(f"{D.PARTY_LABELS[p]}", 0.30, 0.90, uniform, 0.01, key=f"fr_{p}")
                         for p in D.PARTIES}
            else:
                rates = {p: uniform for p in D.PARTIES}
        proj_p = A.project_final(ct_latest, rates, "party").set_index("party")
        with sc[1]:
            tot_final = proj_p["proj_final"].sum()
            d_, r_ = proj_p.loc["D", "proj_final"], proj_p.loc["R", "proj_final"]
            k = st.columns(3)
            with k[0]:
                card("Projected VBM ballots", f"{tot_final:,.0f}",
                     f"{tot_final - proj_p['returned'].sum():,.0f} still to come")
            with k[1]:
                card("D−R margin, final VBM", pp((d_ - r_) / tot_final),
                     f"returned so far: {pp((proj_p.loc['D','returned'] - proj_p.loc['R','returned']) / proj_p['returned'].sum())}")
            with k[2]:
                card("D−R among ballots to come", pp(
                    (proj_p.loc["D", "still_to_come"] - proj_p.loc["R", "still_to_come"])
                    / proj_p["still_to_come"].sum()) if proj_p["still_to_come"].sum() else "—",
                     "registration-based, not vote share")
        pcols = st.columns(2)
        for i, dim in enumerate(["party", "age"]):
            pr = A.project_final(ct_latest, rates, dim)
            pr = pr.set_index("label").reindex([v for v in D.DIM_LABELS[dim].values() if v in set(pr["label"])])
            fig = go.Figure()
            for name, col, colr in (("Returned so far", "share_returned", "#9ca3af"),
                                    ("Still to come", "share_still_to_come", "#d97706"),
                                    ("Projected final", "share_proj_final", "#1a6b3c")):
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
# METHOD & DATA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
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
- **SOS vs Optiq totals differ** — different update times and Optiq's ~98% voter-file match. Use each source on its own terms.
- **Stale snapshots are dropped.** If a feed re-publishes identical numbers on a new date, that date is removed so velocity isn't understated.
- **2022 pace baseline: {meta['baseline_kind']}.** The end-point is the last 2022 pre-election report
  ({pct(meta['rate_2022_baseline'])} district return rate), which is the right target for tracking the early vote.
  The path to it is assumed: (days elapsed ÷ window)^k, with k set in the sidebar (1 = linear). Early in the window the result is very sensitive to k. To replace it, add
  `vbm_2022_daily.csv` with columns
  `county, days_before_eday, vbm_returned` (SD4 portion).
- **Split counties** use a fixed registration-share fraction for the SOS figures; Optiq counts are already district-specific.
- **Party ≠ vote.** Everything here is party registration of returned ballots, not candidate support.
""")
