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
def equiv_2022_fraction(snapshot: pd.Timestamp, k: float = 1.0,
                        curve: pd.Series | None = None) -> tuple[int, float]:
    """Days to E-day and the share of the 2022 early vote in by the same days out.

    curve: share of the 2022 early vote in by each days_before_eday (from the
    PDI daily returns). Without it, (days elapsed / window) ** k is assumed:
    k = 1 is linear; k > 1 is back-loaded.
    """
    days_left = (EDAY_2026 - snapshot).days
    if curve is not None and len(curve):
        if days_left > curve.index.max():
            return days_left, 0.0
        if days_left < curve.index.min():
            return days_left, 1.0
        return days_left, float(curve.loc[curve.index >= days_left].iloc[0])
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
                 daily22: pd.DataFrame | None, k: float = 1.0,
                 curve: pd.Series | None = None,
                 compare_date: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """compare_date: the day to compare with 2022 at (default: the SOS report
    date). The apps pass today, so 2022 is always taken at today's E-number."""
    """Latest SOS figures per county with velocity and 2022 pace."""
    snaps = sorted(sos["snapshot_date"].unique())
    cur_date = snaps[-1]
    cur = sos[sos["snapshot_date"] == cur_date].set_index("county")
    days_left, frac = equiv_2022_fraction(pd.Timestamp(compare_date) if compare_date is not None else cur_date,
                                          k, curve)
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
        "baseline_kind": ("2022 county daily curve" if daily22 is not None
                          else "2022 PDI curve" if curve is not None
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
                     ks=(1.0, 1.5, 2.0, 2.5), compare_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """District pace vs 2022 under several curve shapes."""
    rows = []
    for k in ks:
        _, m = county_table(sos, base, None, k, None, compare_date)
        rows.append({"k": k, "share_in": m["interp_frac"],
                     "rate_2022_equiv": m["rate_2022_equiv"], "pace_ratio": m["pace_ratio"]})
    return pd.DataFrame(rows)


def pace_label(ratio: float) -> str:
    if ratio is None or not np.isfinite(ratio):
        return "—"
    if ratio >= 1.10:
        return "Above"
    if ratio >= 0.85:
        return "Near"
    return "Below"


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


def baseline_2022_rates(base: pd.DataFrame, issued_2026: pd.Series,
                        return_rates: dict) -> dict:
    """District-level 2022 VBM return rates to use as scenario defaults.

    pre   = 2022 VBM returned at the last pre-election report / issued.
    final = estimated 2022 final VBM accepted / issued, where final accepted
            = pre-election accepted / return_rate (vbm_turnout.COUNTY_PARAMS:
            pre-election accepted ÷ final accepted, avg of 2022 & 2024).

    County rates are weighted by 2026 ballots issued, so a change in county
    mix since 2022 doesn't move the default.
    """
    b = base.copy()
    b["pre"] = b["vbm_returned_sd4"] / b["vbm_issued_sd4"].replace(0, np.nan)
    rr = pd.Series(return_rates, dtype=float)
    b["final"] = (b["vbm_accepted_sd4"] / rr.reindex(b.index)) / b["vbm_issued_sd4"].replace(0, np.nan)
    w = issued_2026.reindex(b.index).fillna(0)
    if w.sum() == 0:
        w = b["vbm_issued_sd4"]
    out = {}
    for k in ("pre", "final"):
        ok = b[k].notna()
        out[k] = float((b.loc[ok, k] * w[ok]).sum() / w[ok].sum())
    out["by_county"] = b[["pre", "final"]]
    return out


def forecast_vbm_rates(params: dict, issued_2026: pd.Series, return_rates: dict,
                       vbm_pcts: dict) -> dict | None:
    """Early-vote targets implied by the SD4 forecast.

    Per county: forecast total ballots = registration × forecast turnout;
    final VBM = total × vbm_pct_of_total; pre-election VBM = final VBM ×
    return_rate (both from vbm_turnout.COUNTY_PARAMS). Rates divide by 2026
    ballots issued (SOS).
    """
    if not params or "counties" not in params:
        return None
    rows = []
    for c, d in params["counties"].items():
        if c not in return_rates or c not in issued_2026.index:
            continue
        total = float(d["registration"]) * float(d["turnout"])
        final = total * vbm_pcts[c]
        pre = final * return_rates[c]
        rows.append({"county": c, "total": total, "final_vbm": final, "pre_vbm": pre,
                     "issued": float(issued_2026[c])})
    if not rows:
        return None
    t = pd.DataFrame(rows).set_index("county")
    t["pre"] = t["pre_vbm"] / t["issued"].replace(0, np.nan)
    t["final"] = t["final_vbm"] / t["issued"].replace(0, np.nan)
    return {
        "pre": t["pre_vbm"].sum() / t["issued"].sum(),
        "final": t["final_vbm"].sum() / t["issued"].sum(),
        "pre_vbm": t["pre_vbm"].sum(), "final_vbm": t["final_vbm"].sum(),
        "total": t["total"].sum(),
        "by_county": t[["pre", "final", "pre_vbm", "final_vbm"]],
    }


# ── Optiq county table ───────────────────────────────────────────────────────
def optiq_county_series(cp_hist: pd.DataFrame) -> pd.DataFrame:
    """County × snapshot totals and party detail from Optiq county snapshots,
    with per-day velocity between distinct snapshots."""
    if cp_hist.empty:
        return pd.DataFrame()
    frames = []
    for d, g in cp_hist.groupby("snapshot_date"):
        t = county_party_table(g)
        t["snapshot_date"] = d
        frames.append(t)
    s = pd.concat(frames, ignore_index=True).sort_values(["county", "snapshot_date"])
    s["rate"] = s["tot_ret"] / s["tot_reg"].replace(0, np.nan)
    s["gap_days"] = s.groupby("county")["snapshot_date"].diff().dt.days
    s["per_day"] = s.groupby("county")["tot_ret"].diff() / s["gap_days"]
    dist = (s.groupby("snapshot_date", as_index=False)[["tot_reg", "tot_ret"]].sum()
              .assign(county="SD4 total"))
    dist["rate"] = dist["tot_ret"] / dist["tot_reg"].replace(0, np.nan)
    return pd.concat([s, dist], ignore_index=True)


def optiq_county_table(cp_hist: pd.DataFrame, base: pd.DataFrame,
                       daily22: pd.DataFrame | None, k: float = 1.0,
                       curve: pd.Series | None = None,
                       sos: pd.DataFrame | None = None,
                       compare_date: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """Latest Optiq county figures with party detail, velocity and 2022 pace.

    Denominator is Optiq's per-county registered count. Velocity needs two
    distinct Optiq snapshots (saved by the GitHub Action, plus today's live pull).

    sos: SOS ballot_returns rows. Where Optiq shows 0 returned for a county but
    the latest SOS report has some, the SOS count is used for that county's
    total (and its data date becomes the SOS report date). Party detail for
    those ballots isn't available, so party tables are unaffected.

    compare_date: every county is compared with 2022 at this day's E-number
    (the apps pass today; default is the snapshot date).
    """
    s = optiq_county_series(cp_hist)
    s = s[s["county"] != "SD4 total"]
    dates = sorted(s["snapshot_date"].unique())
    cur_date = dates[-1]
    t = s[s["snapshot_date"] == cur_date].copy()
    # Fill counties Optiq shows as 0 returned from the latest SOS report
    t["sos_filled"] = False
    sos_dates = {}
    if sos is not None and not sos.empty:
        sl = sos[sos["snapshot_date"] == sos["snapshot_date"].max()].set_index("county")
        for i, row in t.iterrows():
            c = row["county"]
            if row["tot_ret"] == 0 and c in sl.index and sl.loc[c, "vbm_returned"] > 0:
                t.at[i, "tot_ret"] = int(sl.loc[c, "vbm_returned"])
                t.at[i, "rate"] = t.at[i, "tot_ret"] / row["tot_reg"] if row["tot_reg"] else np.nan
                t.at[i, "sos_filled"] = True
                sos_dates[c] = pd.Timestamp(sl.loc[c, "snapshot_date"])
    # One comparison day for every county: 2022 at compare_date's E-number
    as_of = pd.Timestamp(compare_date) if compare_date is not None else pd.Timestamp(cur_date)
    days_left, frac = equiv_2022_fraction(as_of, k, curve)
    b22 = baseline_2022_returned(base, daily22, days_left, frac)
    iss22 = base["vbm_issued_sd4"]
    t["rate_2022_equiv"] = t["county"].map(
        lambda c: b22.get(c, np.nan) / iss22.get(c, np.nan) if iss22.get(c, 0) else np.nan)
    t["days_left"] = days_left
    t["pace_ratio"] = t["rate"] / t["rate_2022_equiv"].replace(0, np.nan)
    t["outstanding"] = (t["tot_reg"] - t["tot_ret"]).clip(lower=0)
    tot_ret, tot_reg = t["tot_ret"].sum(), t["tot_reg"].sum()
    t["share_of_district_returns"] = t["tot_ret"] / tot_ret if tot_ret else np.nan
    t["share_of_district_reg"] = t["tot_reg"] / tot_reg
    # Optiq's own most recent return date per county (same on every party row)
    if "latest_return_date" in cp_hist:
        cur_raw = cp_hist[cp_hist["snapshot_date"] == cur_date]
        lr = cur_raw.groupby("county")["latest_return_date"].max()
        t["latest_return_date"] = t["county"].map(lr)
    else:
        t["latest_return_date"] = pd.NaT
    for c, d in sos_dates.items():
        t.loc[t["county"] == c, "latest_return_date"] = d
    ok = t["rate_2022_equiv"].notna()
    w22 = t["county"].map(iss22).fillna(0)
    rate22 = float((t.loc[ok, "rate_2022_equiv"] * w22[ok]).sum() / w22[ok].sum()) if w22[ok].sum() else np.nan
    meta = {
        "snapshot": pd.Timestamp(cur_date), "as_of": as_of, "days_left": days_left, "interp_frac": frac,
        "n_snapshots": len(dates),
        "prev_snapshot": pd.Timestamp(dates[-2]) if len(dates) >= 2 else None,
        "registered": int(tot_reg), "returned": int(tot_ret),
        "rate": tot_ret / tot_reg if tot_reg else np.nan,
        "rate_2022_equiv": rate22,
    }
    meta["pace_ratio"] = meta["rate"] / rate22 if rate22 else np.nan
    return t.reset_index(drop=True), meta


def last_drop(series: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Size and date of each county's most recent increase in returns.

    series: one row per county per snapshot (county, snapshot_date, value_col).
    A county whose numbers didn't move in the latest snapshot keeps its
    previous drop, so slow-reporting counties still show when they last updated.
    """
    s = series[series["county"] != "SD4 total"].sort_values(["county", "snapshot_date"]).copy()
    s["new"] = s.groupby("county")[value_col].diff()
    drops = s[s["new"] > 0]
    last = drops.groupby("county").tail(1).set_index("county")
    out = pd.DataFrame(index=sorted(s["county"].unique()))
    out["last_drop"] = last["new"]
    out["last_drop_date"] = last["snapshot_date"]
    return out.rename_axis("county").reset_index()


# ── District total rows ──────────────────────────────────────────────────────
def with_total_row(t: pd.DataFrame, base: pd.DataFrame, n_col: str, r_col: str,
                   share_cols: list[str]) -> pd.DataFrame:
    """Append an "SD4 total" row to a county totals table.

    Rates are recomputed from summed counts (not averaged). The 2022
    comparison is weighted by each county's 2022 ballots issued. Last drop is
    left blank because counties drop on different days; drop date is the
    most recent one.
    """
    row = {"county": "SD4 total", n_col: t[n_col].sum(), r_col: t[r_col].sum()}
    row["rate"] = row[r_col] / row[n_col] if row[n_col] else np.nan
    if "outstanding" in t:
        row["outstanding"] = t["outstanding"].sum()
    for c in share_cols:
        row[c] = 1.0
    if "last_drop_date" in t:
        row["last_drop_date"] = t["last_drop_date"].max()
    if "rate_2022_equiv" in t:
        iss22 = base["vbm_issued_sd4"].reindex(t["county"]).values
        ok = t["rate_2022_equiv"].notna().values
        row["rate_2022_equiv"] = ((t["rate_2022_equiv"].values[ok] * iss22[ok]).sum() / iss22[ok].sum()
                                  if iss22[ok].sum() else np.nan)
        row["pace_ratio"] = row["rate"] / row["rate_2022_equiv"] if row["rate_2022_equiv"] else np.nan
        row["pace"] = pace_label(row["pace_ratio"])
    return pd.concat([t, pd.DataFrame([row])], ignore_index=True)


def county_party_with_total(cp: pd.DataFrame) -> pd.DataFrame:
    """county_party_table plus an "SD4 total" row built from summed counts."""
    t = county_party_table(cp)
    dist = cp.groupby("party", as_index=False)[["registered", "returned"]].sum().assign(county="SD4 total")
    return pd.concat([t, county_party_table(dist)], ignore_index=True)


# ── Shift decomposition: within-county vs county mix ─────────────────────────
def margin_decomposition(cp: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Split the district D−R shift (returns margin − registration margin)
    into two exact parts:

      within = Σ_c  w_ret_c · (m_ret_c − m_reg_c)
               parties returning at different rates inside each county
      mix    = Σ_c (w_ret_c − w_reg_c) · (m_reg_c − M_reg)
               some counties making up more of the returns than of registration

    w = county share of district returns / registration; m = county D−R margin
    (÷ all ballots, incl. O/NPP); M_reg = district registration margin.
    within + mix = M_ret − M_reg exactly. Centring the mix term on M_reg makes
    each county's piece read as "over-returning a county more R (or D) than
    the district"; it doesn't change the total because Σ(w_ret − w_reg) = 0.
    A county with no returns gets a within-county term of zero.
    """
    t = county_party_table(cp)
    t["w_ret"] = t["tot_ret"] / t["tot_ret"].sum()
    t["w_reg"] = t["tot_reg"] / t["tot_reg"].sum()
    m_reg = t["dr_margin_reg"]
    m_ret = t["dr_margin_ret"].fillna(m_reg)
    M_reg = float((t["w_reg"] * m_reg).sum())
    M_ret = float((t["D_ret"].sum() - t["R_ret"].sum()) / t["tot_ret"].sum())
    t["within"] = t["w_ret"] * (m_ret - m_reg)
    t["mix"] = (t["w_ret"] - t["w_reg"]) * (m_reg - M_reg)
    out = {"reg": M_reg, "ret": M_ret, "shift": M_ret - M_reg,
           "within": float(t["within"].sum()), "mix": float(t["mix"].sum())}
    return out, t[["county", "w_ret", "w_reg", "dr_margin_reg", "dr_margin_ret", "within", "mix"]]


def decomposition_series(cp_hist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in cp_hist.groupby("snapshot_date"):
        if g["returned"].sum() == 0:
            continue
        tot, _ = margin_decomposition(g)
        rows.append({"snapshot_date": d, **tot})
    return pd.DataFrame(rows)



# ── 2022 vs 2026 by days out ─────────────────────────────────────────────────
def pdi_cumulative(pdi: pd.DataFrame) -> pd.DataFrame:
    """Running totals of the 2022 PDI daily returns, by days before E-day."""
    d = pdi.sort_values("days_before_eday", ascending=False).copy()
    for c in ("D", "R", "O", "total"):
        d[f"cum_{c}"] = d[c].cumsum()
    t = d[["cum_D", "cum_R", "cum_O"]].sum(axis=1).replace(0, np.nan)
    for c in ("D", "R", "O"):
        d[f"share_{c}"] = d[f"cum_{c}"] / t
    d["dr_margin"] = (d["cum_D"] - d["cum_R"]) / t
    d["frac_of_final"] = d["cum_total"] / d["cum_total"].iloc[-1]
    return d.reset_index(drop=True)


def pdi_curve(pdi: pd.DataFrame | None) -> pd.Series | None:
    """Share of the 2022 early vote in by each days_before_eday (index)."""
    if pdi is None or pdi.empty:
        return None
    c = pdi_cumulative(pdi)
    return c.set_index("days_before_eday")["frac_of_final"].sort_index()


def at_days_out(cum22: pd.DataFrame, days_out: int) -> pd.Series | None:
    """2022 running totals as of a given days-before-E-day (inclusive)."""
    rows = cum22[cum22["days_before_eday"] >= days_out]
    return rows.iloc[-1] if len(rows) else None


def party_by_days_out(ct: pd.DataFrame) -> pd.DataFrame:
    """2026 Optiq party totals for each snapshot, keyed by days before E-day."""
    if ct.empty:
        return pd.DataFrame()
    g = ct.groupby(["snapshot_date", "party"])["returned"].sum().unstack("party", fill_value=0)
    for p in PARTIES:
        if p not in g:
            g[p] = 0
    g = g[PARTIES].reset_index()
    g["days_before_eday"] = (EDAY_2026 - g["snapshot_date"]).dt.days
    g["total"] = g[PARTIES].sum(axis=1)
    t = g["total"].replace(0, np.nan)
    for p in PARTIES:
        g[f"share_{p}"] = g[p] / t
    g["dr_margin"] = (g["D"] - g["R"]) / t
    return g.sort_values("days_before_eday", ascending=False).reset_index(drop=True)


def registration_margin(reg_df: pd.DataFrame) -> dict:
    """District D/R/O shares and D−R margin from a county registration table."""
    T = reg_df["registered"].sum()
    return {"registered": int(T), "D": reg_df["D"].sum() / T, "R": reg_df["R"].sum() / T,
            "O": reg_df["O"].sum() / T, "dr": (reg_df["D"].sum() - reg_df["R"].sum()) / T}


def party_by_days_out_cp(cp_hist: pd.DataFrame) -> pd.DataFrame:
    """2026 party totals from Optiq county-feed snapshots, keyed by days out
    (each snapshot dated by the day it was fetched)."""
    if cp_hist.empty:
        return pd.DataFrame()
    g = cp_hist.groupby(["snapshot_date", "party"])["returned"].sum().unstack("party", fill_value=0)
    for p in PARTIES:
        if p not in g:
            g[p] = 0
    g = g[PARTIES].reset_index()
    g["as_of"] = g["snapshot_date"]   # each snapshot is dated by the day it was fetched
    g["days_before_eday"] = (EDAY_2026 - pd.to_datetime(g["as_of"])).dt.days
    g["total"] = g[PARTIES].sum(axis=1)
    t = g["total"].replace(0, np.nan)
    for p in PARTIES:
        g[f"share_{p}"] = g[p] / t
    g["dr_margin"] = (g["D"] - g["R"]) / t
    return g.sort_values("days_before_eday", ascending=False).reset_index(drop=True)


def series_2026(ct_hist: pd.DataFrame, cp_hist: pd.DataFrame) -> pd.DataFrame:
    """One 2026 point per days-out: county feed where available, crosstabs otherwise."""
    a = party_by_days_out_cp(cp_hist).assign(source="Optiq county feed")
    b = party_by_days_out(ct_hist).assign(source="Optiq crosstabs")
    both = pd.concat([a, b], ignore_index=True)
    if both.empty:
        return both
    both["_pri"] = (both["source"] != "Optiq county feed").astype(int)
    both = both.sort_values(["days_before_eday", "_pri"]).drop_duplicates("days_before_eday")
    return both.drop(columns="_pri").sort_values("days_before_eday", ascending=False).reset_index(drop=True)
