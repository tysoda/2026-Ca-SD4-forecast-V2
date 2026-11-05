"""
late_model.py — win probability from the votes still to be counted.

For each county:
  final = votes counted + remaining votes × late-vote Dem share

Late-vote Dem share (two-party), per county:
  prior    the county's mail-ballot Dem share where the county reports results by
           vote type (mail_splits.csv), otherwise its first election-night drop
           (first_drop.csv, almost all early mail), otherwise its current share
  update   blended with the county's post-election-day batches
           (posted at or after LATE_COUNT_START), weighted by votes:
             lean = (k·prior + n_late·late_share) / (k + n_late)

Uncertainty, simulated:
  shared   one district-wide shift applied to every county's late lean, so
           counties' late votes can lean the same way together
  county   independent county shift, shrinking as late batches arrive
  size     remaining votes scaled by a lognormal factor (tighter when the SOS
           unprocessed-ballot figure is used)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LATE_COUNT_START = pd.Timestamp("2026-11-04 05:00")   # Pacific; earlier batches = election-night count


@dataclass
class LateSettings:
    prior_weight: float = 3000.0     # k, in votes
    county_sd: float = 0.03          # county late-lean SD before any late batches
    shared_sd: float = 0.02          # district-wide late-lean SD
    rem_sd_model: float = 0.20       # remaining-vote size SD (model estimate)
    rem_sd_sos: float = 0.07         # remaining-vote size SD (SOS unprocessed figure)
    n_sims: int = 20000
    seed: int = 2026


def county_inputs(current: pd.DataFrame, remaining: dict, mail_splits: pd.DataFrame | None,
                  first_drop: pd.DataFrame | None, late_batches: pd.DataFrame | None,
                  mock_mail: bool = False, s: LateSettings = LateSettings()) -> pd.DataFrame:
    """One row per county with counted votes, remaining votes and the late lean to use."""
    ms = {} if mail_splits is None or mail_splits.empty else mail_splits.set_index("county").to_dict("index")
    fd = {} if first_drop is None or first_drop.empty else first_drop.set_index("county").to_dict("index")
    lb = pd.DataFrame() if late_batches is None else late_batches
    rows = []
    for _, r in current.iterrows():
        cn = r["county"]
        d = 0 if pd.isna(r.get("dem_votes")) else int(r["dem_votes"])
        rp = 0 if pd.isna(r.get("rep_votes")) else int(r["rep_votes"])
        n_rem, rem_src = remaining.get(cn, (None, "none"))
        # Prior: mail split > first drop > current share
        if mock_mail and pd.notna(r.get("mail_dem")) and (r.get("mail_dem", 0) + r.get("mail_rep", 0)) > 0:
            prior, src = r["mail_dem"] / (r["mail_dem"] + r["mail_rep"]), "mock mail split"
        elif cn in ms and ms[cn]["mail_dem"] + ms[cn]["mail_rep"] > 0:
            m = ms[cn]
            prior, src = m["mail_dem"] / (m["mail_dem"] + m["mail_rep"]), "county mail split"
        elif cn in fd and fd[cn]["dem_votes"] + fd[cn]["rep_votes"] > 0:
            f = fd[cn]
            prior, src = f["dem_votes"] / (f["dem_votes"] + f["rep_votes"]), "first drop"
        elif d + rp > 0:
            prior, src = d / (d + rp), "current share"
        else:
            prior, src = np.nan, "none"
        # Post-election-day batches
        n_late = s_late = 0.0
        if not lb.empty:
            g = lb[lb["county"] == cn]
            n_late = float((g["new_dem"] + g["new_rep"]).sum())
            s_late = float(g["new_dem"].sum() / n_late) if n_late > 0 else 0.0
        k = s.prior_weight
        lean = (k * prior + n_late * s_late) / (k + n_late) if not np.isnan(prior) else np.nan
        rows.append({
            "county": cn, "counted_dem": d, "counted_rep": rp,
            "remaining": n_rem if n_rem is not None else 0, "remaining_source": rem_src,
            "prior": prior, "prior_source": src, "late_votes": int(n_late),
            "late_share": s_late if n_late > 0 else np.nan, "lean": lean,
            "lean_sd": s.county_sd * np.sqrt(k / (k + n_late)),
        })
    out = pd.DataFrame(rows)
    out["exp_net"] = out["remaining"] * (2 * out["lean"].fillna(0.5) - 1)
    return out


def simulate(ci: pd.DataFrame, s: LateSettings = LateSettings()) -> dict:
    rng = np.random.default_rng(s.seed)
    n = s.n_sims
    shared = rng.normal(0, s.shared_sd, n)
    D = np.full(n, float(ci["counted_dem"].sum()))
    R = np.full(n, float(ci["counted_rep"].sum()))
    for _, c in ci.iterrows():
        if c["remaining"] <= 0 or np.isnan(c["lean"]):
            continue
        sd = s.rem_sd_sos if c["remaining_source"] == "sos" else s.rem_sd_model
        size = c["remaining"] * np.exp(rng.normal(-sd**2 / 2, sd, n))
        p = np.clip(c["lean"] + shared + rng.normal(0, c["lean_sd"], n), 0, 1)
        D += size * p
        R += size * (1 - p)
    share, margin = D / (D + R), D - R
    cd, cr, rem = ci["counted_dem"].sum(), ci["counted_rep"].sum(), ci["remaining"].sum()
    return {
        "win_prob": float(np.mean(share >= 0.5)),
        "share_mean": float(share.mean()),
        "share_p05": float(np.percentile(share, 5)), "share_p95": float(np.percentile(share, 95)),
        "margin_median": float(np.median(margin)),
        "margin_p05": float(np.percentile(margin, 5)), "margin_p95": float(np.percentile(margin, 95)),
        "remaining": float(rem),
        "needed_share": float((0.5 * (cd + cr + rem) - cd) / rem) if rem > 0 else np.nan,
        "expected_late_share": float((ci["remaining"] * ci["lean"]).sum() / rem) if rem > 0 else np.nan,
        "share_sims": share,
    }
