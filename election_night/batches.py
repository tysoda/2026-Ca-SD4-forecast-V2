"""
batches.py — turn election_night_history.csv into vote batches.

The history has one row per county each time its count changes (written by
fetch_election_results.py). A batch is the difference between two consecutive
rows for a county: the votes that county added in one update.

Used by the Election Night app and scripts/notify_county_updates.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VOTE_COLS = ["dem_votes", "rep_votes", "other_votes"]


def compute_batches(hist: pd.DataFrame) -> pd.DataFrame:
    """One row per county update after the county's first recorded count.

    Columns: county, fetched_at, last_updated, new_dem, new_rep, new_other,
    new_votes, dem_share_new (two-party), dem_share_before, dem_share_after,
    net_dem (new_dem − new_rep), dem_after, rep_after.
    """
    cols = ["county", "fetched_at", "last_updated", "new_dem", "new_rep", "new_other", "new_votes",
            "dem_share_new", "dem_share_before", "dem_share_after", "net_dem", "dem_after", "rep_after"]
    if hist is None or hist.empty:
        return pd.DataFrame(columns=cols)
    h = hist.copy()
    for c in VOTE_COLS:
        h[c] = pd.to_numeric(h[c], errors="coerce").fillna(0).astype(int)
    h = h.sort_values(["county", "fetched_at"], kind="stable")
    rows = []
    for cn, g in h.groupby("county", sort=False):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            p, r = g.loc[i - 1], g.loc[i]
            nd, nr, no = (int(r[c] - p[c]) for c in VOTE_COLS)
            two_new, two_before, two_after = nd + nr, p.dem_votes + p.rep_votes, r.dem_votes + r.rep_votes
            rows.append({
                "county": cn, "fetched_at": r.fetched_at, "last_updated": r.last_updated,
                "new_dem": nd, "new_rep": nr, "new_other": no, "new_votes": nd + nr + no,
                "dem_share_new": nd / two_new if two_new > 0 else np.nan,
                "dem_share_before": p.dem_votes / two_before if two_before > 0 else np.nan,
                "dem_share_after": r.dem_votes / two_after if two_after > 0 else np.nan,
                "net_dem": nd - nr, "dem_after": int(r.dem_votes), "rep_after": int(r.rep_votes),
            })
    return pd.DataFrame(rows, columns=cols).sort_values("fetched_at", kind="stable").reset_index(drop=True)


def since_baseline(hist: pd.DataFrame) -> dict | None:
    """District totals for all votes added after each county's first recorded count."""
    b = compute_batches(hist)
    if b.empty:
        return None
    nd, nr, no = int(b.new_dem.sum()), int(b.new_rep.sum()), int(b.new_other.sum())
    h = hist.copy()
    for c in VOTE_COLS:
        h[c] = pd.to_numeric(h[c], errors="coerce").fillna(0).astype(int)
    first = h.sort_values("fetched_at", kind="stable").groupby("county").head(1)
    bd, br = int(first.dem_votes.sum()), int(first.rep_votes.sum())
    return {
        "baseline_at": str(first.fetched_at.min()),
        "new_dem": nd, "new_rep": nr, "new_other": no, "new_votes": nd + nr + no,
        "dem_share_new": nd / (nd + nr) if nd + nr > 0 else np.nan,
        "dem_share_baseline": bd / (bd + br) if bd + br > 0 else np.nan,
        "net_dem": nd - nr,
    }
