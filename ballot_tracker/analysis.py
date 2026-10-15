"""
analysis.py — computations for the SD4 ballot return tracker.

Pure functions over the frames from data.py. Conventions:
  * "rate"            = returned / issued within a group
  * "share of returns"= group returned / all returned
  * "share of issued" = group issued / all issued
  * "representation"  = share of returns − share of issued (pp). Positive means
                        the group is over-represented among ballots returned so far.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data import (
    EDAY_2022, EDAY_2026, MAILOUT_2022, PARTIES, SD4_COUNTIES, DIM_LABELS,
)


# ── Generic crosstab engine ──────────────────────────────────────────────────
def latest(df: pd.DataFrame, date_col: str = "snapshot_date") -> pd.DataFrame:
    if df.empty:
        return df
    return df[df[date_col] == df[date_col].max()]


def group_summary(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Issued / returned / rate / shares / representation for any grouping."""
    g = df.groupby(by, as_index=False)[["issued", "returned"]].sum()
    tot_i, tot_r = g["issued"].sum(), g["returned"].sum()
    g["outstanding"] = (g["issued"] - g["returned"]).clip(lower=0)
    g["rate"] = g["returned"] / g["issued"].replace(0, np.nan)
    g["share_issued"] = g["issued"] / tot_i if tot_i else np.nan
    g["share_returned"] = g["returned"] / tot_r if tot_r else np.nan
    g["representation"] = g["share_returned"] - g["share_issued"]
    tot_o = g["outstanding"].sum()
    g["share_outstanding"] = g["outstanding"] / tot_o if tot_o else np.nan
    return g


def two_way(df: pd.DataFrame, row: str, col: str, metric: str) -> pd.DataFrame:
    """Pivot a two-way crosstab.

    For share metrics the denominator is the whole universe (so the full grid
    sums to 100%), except "col_share", which is the composition within each
    column (e.g. party mix of returns within each age group).
    """
    s = group_summary(df, [row, col])
    if metric == "col_share":
        s["col_share"] = s["returned"] / s.groupby(col)["returned"].transform("sum").replace(0, np.nan)
    p = s.pivot(index=row, columns=col, values=metric)
    # Order + label axes
    for axis, dim in ((0, row), (1, col)):
        order = [k for k in DIM_LABELS[dim] if k in (p.index if axis == 0 else p.columns)]
        p = p.reindex(order, axis=axis)
    p.index = [DIM_LABELS[row].get(k, k) for k in p.index]
    p.columns = [DIM_LABELS[col].get(k, k) for k in p.columns]
    return p


def label_codes(s: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    s = s.copy()
    for d in dims:
        s[d] = s[d].map(lambda v, d=d: DIM_LABELS[d].get(v, v))
    return s


# ── SOS county pace vs 2022 ──────────────────────────────────────────────────
def equiv_2022_fraction(snapshot: pd.Timestamp, k: float = 1.0) -> tuple[int, float]:
    """Days to E-day and the share of the 2022 early vote assumed in by the
    equivalent day: (days elapsed / window) ** k. k = 1 is linear; k > 1 is
    back-loaded (most ballots arrive late in the window)."""
    days_left = (EDAY_2026 - snapshot).days
    equiv = EDAY_2022 - pd.Timedelta(days=days_left)
    window = (EDAY_2022 - MAILOUT_2022).days
    t = float(np.clip((equiv - MAILOUT_2022).days / window, 0, 1))
    return days_left, t ** k


def baseline_2022_returned(base: pd.DataFrame, daily22: pd.DataFrame | None,
                           days_left: int, frac: float) -> pd.Series:
    """2022 returned count per county at the equivalent point in the cycle."""
    if daily22 is not None and not daily22.empty:
        d = daily22[daily22["days_before_eday"] >= days_left]
        # Last observation at or before the equivalent day
        at = (d.sort_values("days_before_eday")
                .groupby("county")["vbm_returned"].first())
        return at.reindex(base.index).fillna(0)
    return base["vbm_returned_sd4"] * frac


def county_table(sos: pd.DataFrame, base: pd.DataFrame,
                 daily22: pd.DataFrame | None, k: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """Latest SOS figures per county with velocity and 2022 pace."""
    snaps = sorted(sos["snapshot_date"].unique())
    cur_date = snaps[-1]
    cur = sos[sos["snapshot_date"] == cur_date].set_index("county")
    days_left, frac = equiv_2022_fraction(cur_date, k)
    b22 = baseline_2022_returned(base, daily22, days_left, frac)

    # Velocity over the last real (non-stale) interval
    if len(snaps) >= 2:
        prev_date = snaps[-2]
        prev = sos[sos["snapshot_date"] == prev_date].set_index("county")
        gap = max((cur_date - prev_date).days, 1)
    else:
        prev, gap = None, 1

    rows = []
    for cn in SD4_COUNTIES:
        if cn not in cur.index:
            continue
        r = cur.loc[cn]
        issued, ret = int(r["vbm_issued"]), int(r["vbm_returned"])
        iss22 = int(base.loc[cn, "vbm_issued_sd4"]) if cn in base.index else 0
        ret22_eq = float(b22.get(cn, 0))
        rate = ret / issued if issued else np.nan
        rate22_eq = ret22_eq / iss22 if iss22 else np.nan
        per_day = ((ret - int(prev.loc[cn, "vbm_returned"])) / gap
                   if prev is not None and cn in prev.index else np.nan)
        rows.append({
            "county": cn, "issued": issued, "returned": ret,
            "accepted": int(r["vbm_accepted"]),
            "outstanding": max(issued - ret, 0),
            "rate": rate, "rate_2022_equiv": rate22_eq,
            "pace_ratio": rate / rate22_eq if rate22_eq and rate22_eq > 0 else np.nan,
            "per_day": per_day,
            "share_of_district_returns": np.nan,  # filled below
        })
    t = pd.DataFrame(rows)
    tot_ret = t["returned"].sum()
    t["share_of_district_returns"] = t["returned"] / tot_ret if tot_ret else np.nan
    t["share_of_district_issued"] = t["issued"] / t["issued"].sum()

    iss22_tot = base["vbm_issued_sd4"].sum()
    ret22_tot = float(b22.sum())
    meta = {
        "snapshot": cur_date, "days_left": days_left, "interp_frac": frac,
        "baseline_kind": ("2022 daily curve" if daily22 is not None
                          else "linear path" if k == 1 else f"curve k = {k:g}"),
        "k": k,
        "issued": int(t["issued"].sum()), "returned": int(tot_ret),
        "rate": tot_ret / t["issued"].sum() if t["issued"].sum() else np.nan,
        "rate_2022_equiv": ret22_tot / iss22_tot if iss22_tot else np.nan,
        "rate_2022_baseline": base["vbm_returned_sd4"].sum() / iss22_tot if iss22_tot else np.nan,
        "per_day": t["per_day"].sum(min_count=1),
        "n_snapshots": len(snaps), "first_snapshot": snaps[0],
    }
    meta["pace_ratio"] = (meta["rate"] / meta["rate_2022_equiv"]
                          if meta["rate_2022_equiv"] else np.nan)
    return t, meta


def pace_sensitivity(sos: pd.DataFrame, base: pd.DataFrame,
                     ks=(1.0, 1.5, 2.0, 2.5)) -> pd.DataFrame:
    """District pace vs 2022 under several curve shapes."""
    rows = []
    for k in ks:
        _, m = county_table(sos, base, None, k)
        rows.append({"k": k, "share_in": m["interp_frac"],
                     "rate_2022_equiv": m["rate_2022_equiv"], "pace_ratio": m["pace_ratio"]})
    return pd.DataFrame(rows)


def pace_label(ratio: float) -> str:
    if ratio is None or not np.isfinite(ratio):
        return "—"
    if ratio >= 1.10:
        return "Ahead"
    if ratio >= 0.85:
        return "On pace"
    return "Behind"


# ── Time series ──────────────────────────────────────────────────────────────
def sos_series(sos: pd.DataFrame) -> pd.DataFrame:
    """District + county cumulative returns with per-day velocity."""
    d = sos.groupby(["snapshot_date", "county"], as_index=False)[["vbm_issued", "vbm_returned"]].sum()
    dist = d.groupby("snapshot_date", as_index=False)[["vbm_issued", "vbm_returned"]].sum()
    dist["county"] = "SD4 total"
    d = pd.concat([d, dist], ignore_index=True).sort_values(["county", "snapshot_date"])
    d["rate"] = d["vbm_returned"] / d["vbm_issued"].replace(0, np.nan)
    d["gap_days"] = d.groupby("county")["snapshot_date"].diff().dt.days
    d["new"] = d.groupby("county")["vbm_returned"].diff()
    d["per_day"] = d["new"] / d["gap_days"]
    d["days_left"] = (EDAY_2026 - d["snapshot_date"]).dt.days
    return d


def optiq_series(ct: pd.DataFrame, dim: str) -> pd.DataFrame:
    """Per-snapshot summary by one dimension: rate, share of returns, velocity."""
    frames = []
    for date, g in ct.groupby("snapshot_date"):
        s = group_summary(g, [dim])
        s["snapshot_date"] = date
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).sort_values([dim, "snapshot_date"])
    out["gap_days"] = out.groupby(dim)["snapshot_date"].diff().dt.days
    out["per_day"] = out.groupby(dim)["returned"].diff() / out["gap_days"]
    # Composition of the *new* returns in each interval
    new = out.groupby(dim)["returned"].diff()
    out["new"] = new
    interval_total = out.groupby("snapshot_date")["new"].transform("sum")
    # Intervals with a trickle of new ballots give meaningless compositions
    out["share_of_new"] = np.where(interval_total >= 100, new / interval_total, np.nan)
    out["label"] = out[dim].map(lambda v: DIM_LABELS[dim].get(v, v))
    return out


def live_daily_curve(live: pd.DataFrame, dim: str) -> pd.DataFrame:
    """Exact daily returns by return date from a live Optiq pull."""
    if live.empty or live["return_date"].notna().sum() == 0:
        return pd.DataFrame()
    issued = live.groupby(dim)["voters"].sum()
    r = (live.dropna(subset=["return_date"])
             .groupby(["return_date", dim])["voters"].sum()
             .unstack(dim, fill_value=0).sort_index())
    full = pd.date_range(r.index.min(), r.index.max(), freq="D")
    r = r.reindex(full, fill_value=0)
    cum = r.cumsum()
    rows = []
    for k in r.columns:
        for dt in r.index:
            rows.append({"date": dt, dim: k, "new": int(r.loc[dt, k]),
                         "cum": int(cum.loc[dt, k]),
                         "rate": cum.loc[dt, k] / issued[k] if issued[k] else np.nan,
                         "label": DIM_LABELS[dim].get(k, k)})
    return pd.DataFrame(rows)


# ── County × party (Optiq geography) ─────────────────────────────────────────
def county_party_table(cp: pd.DataFrame) -> pd.DataFrame:
    """Wide per-county D/R/O table with rates, shares and margins."""
    if cp.empty:
        return pd.DataFrame()
    w = cp.pivot_table(index="county", columns="party", values=["registered", "returned"], aggfunc="sum")
    out = pd.DataFrame(index=w.index)
    for p in PARTIES:
        out[f"{p}_reg"] = w[("registered", p)] if ("registered", p) in w else 0
        out[f"{p}_ret"] = w[("returned", p)] if ("returned", p) in w else 0
        out[f"{p}_rate"] = out[f"{p}_ret"] / out[f"{p}_reg"].replace(0, np.nan)
    out["tot_reg"] = out[[f"{p}_reg" for p in PARTIES]].sum(axis=1)
    out["tot_ret"] = out[[f"{p}_ret" for p in PARTIES]].sum(axis=1)
    for p in PARTIES:
        out[f"{p}_share_ret"] = out[f"{p}_ret"] / out["tot_ret"].replace(0, np.nan)
        out[f"{p}_share_reg"] = out[f"{p}_reg"] / out["tot_reg"].replace(0, np.nan)
    out["dr_margin_ret"] = (out["D_ret"] - out["R_ret"]) / out["tot_ret"].replace(0, np.nan)
    out["dr_margin_reg"] = (out["D_reg"] - out["R_reg"]) / out["tot_reg"].replace(0, np.nan)
    out["margin_shift"] = out["dr_margin_ret"] - out["dr_margin_reg"]
    out["turnout_gap"] = out["D_rate"] - out["R_rate"]
    out["raw_dr"] = out["D_ret"] - out["R_ret"]
    return out.reset_index()


# ── Outstanding-ballot scenario ──────────────────────────────────────────────
def project_final(ct_latest: pd.DataFrame, final_rates: dict, dim: str = "party") -> pd.DataFrame:
    """Project the final VBM electorate under assumed final return rates by party.

    final_rates: {party: final return rate (0–1)}. A group whose current rate
    already exceeds its assumed final is held at its current count.
    Returns one row per value of `dim` with current and projected composition.
    """
    cells = ct_latest.copy()
    cells["final_rate"] = cells["party"].map(final_rates).fillna(np.mean(list(final_rates.values())))
    # Apply the party rate within each cell, but never below what's already in
    cells["proj_final"] = np.maximum(cells["issued"] * cells["final_rate"], cells["returned"])
    cells["still_to_come"] = cells["proj_final"] - cells["returned"]
    g = cells.groupby(dim, as_index=False)[["issued", "returned", "proj_final", "still_to_come"]].sum()
    for c, tot in (("returned", g["returned"].sum()), ("proj_final", g["proj_final"].sum()),
                   ("still_to_come", g["still_to_come"].sum())):
        g[f"share_{c}"] = g[c] / tot if tot else np.nan
    g["label"] = g[dim].map(lambda v: DIM_LABELS[dim].get(v, v))
    return g
