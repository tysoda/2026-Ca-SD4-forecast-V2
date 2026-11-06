"""
remaining.py — remaining votes from SOS unprocessed-ballot figures, aged by later
batches, and the history of the Dem share needed of the remaining vote.

An SOS unprocessed figure is a snapshot: ballots still to count as of its `as_of`
time. When a county posts a batch after that, those ballots come out of the
unprocessed pile, so the remaining SD4 votes are

    unprocessed × SD4 share − (SD4 votes counted now − SD4 votes counted at as_of)

`as_of` is Pacific time (YYYY-MM-DD HH:MM). Set it to the county's results update
the SOS figure goes with (the "last updated" time of the count it was reported
alongside). Counted-at-as_of comes from election_night_history.csv, using each
row's SOS posting time. If as_of is blank or earlier than the history, no batches
are subtracted.

Needed share (two-party, same formula as late_model.simulate):
    needed = (0.5 × (D + R + remaining) − D) / remaining
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VOTE_COLS = ["dem_votes", "rep_votes", "other_votes"]


def parse_pacific(s) -> pd.Timestamp:
    """Naive Pacific Timestamp from 'YYYY-MM-DD HH:MM' (read as Pacific, not UTC), the
    SOS 'November 4, 2026, 4:22 p.m.' format, or an ISO string with a Z/offset."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return pd.NaT
    t = str(s).strip()
    if not t:
        return pd.NaT
    sos = t.replace("a.m.", "AM").replace("p.m.", "PM")
    for fmt in ("%B %d, %Y, %I:%M %p", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return pd.to_datetime(sos, format=fmt)
        except (ValueError, TypeError):
            pass
    ts = pd.to_datetime(t, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is not None:
        ts = ts.tz_convert("America/Los_Angeles").tz_localize(None)
    return ts


def load_unprocessed(p) -> dict:
    """county → {"unprocessed": float, "as_of": str, "as_of_ts": Timestamp|NaT}."""
    try:
        u = pd.read_csv(p, dtype=str)
    except Exception:
        return {}
    out = {}
    for _, r in u.iterrows():
        cn = str(r.get("county", "")).strip()
        n = pd.to_numeric(str(r.get("unprocessed_ballots", "")).replace(",", "").strip(), errors="coerce")
        if not cn or pd.isna(n):
            continue
        a = r.get("as_of", "")
        a = "" if pd.isna(a) else str(a).strip()
        out[cn] = {"unprocessed": float(n), "as_of": a, "as_of_ts": parse_pacific(a)}
    return out


def prep_history(hist: pd.DataFrame | None) -> pd.DataFrame:
    """History with integer votes, a `total` column and a Pacific `posted` time
    (SOS update time, else fetch time), sorted by posted time."""
    cols = ["county", "fetched_at", "last_updated", *VOTE_COLS, "total", "posted"]
    if hist is None or hist.empty:
        return pd.DataFrame(columns=cols)
    h = hist.copy()
    for c in VOTE_COLS:
        h[c] = pd.to_numeric(h[c], errors="coerce").fillna(0).astype(int)
    h["total"] = h[VOTE_COLS].sum(axis=1)
    posted = h["last_updated"].map(parse_pacific)
    h["posted"] = posted.where(posted.notna(), h["fetched_at"].map(parse_pacific))
    return h.sort_values(["posted", "fetched_at"], kind="stable").reset_index(drop=True)


def counted_at(h: pd.DataFrame, county: str, t) -> int | None:
    """County's SD4 total in the last history row posted at or before t (None if none)."""
    if pd.isna(t) or h.empty:
        return None
    rows = h[(h["county"] == county) & (h["posted"] <= t)]
    return None if rows.empty else int(rows["total"].iloc[-1])


def sos_remaining(entry: dict, frac: float, counted_now: int, h: pd.DataFrame, county: str):
    """(remaining SD4 votes, votes subtracted for batches since as_of)."""
    base = entry["unprocessed"] * frac
    at = counted_at(h, county, entry["as_of_ts"])
    since = max(0, counted_now - at) if at is not None else 0
    return max(0, int(round(base - since))), since


def needed_share(dem, rep, remaining):
    if remaining is None or remaining <= 0:
        return np.nan
    return (0.5 * (dem + rep + remaining) - dem) / remaining


def need_history(h: pd.DataFrame, unproc: dict, frac: dict, model_total: dict) -> pd.DataFrame:
    """Replay the history: after each county update, district totals, the update's own
    two-party Dem share, remaining votes and the Dem share needed of them.

    Remaining per county at time t: the aged SOS figure once t ≥ its as_of (or always,
    when as_of is blank), otherwise the model (registration × turnout − counted).
    """
    if h.empty:
        return pd.DataFrame()
    state, rows = {}, []
    for i, r in h.iterrows():
        cn, t = r["county"], r["posted"]
        prev = state.get(cn)
        state[cn] = (int(r["dem_votes"]), int(r["rep_votes"]), int(r["total"]))
        D = sum(v[0] for v in state.values())
        R = sum(v[1] for v in state.values())
        past = h.iloc[: i + 1]
        rem = 0
        for c, est in model_total.items():
            counted = state.get(c, (0, 0, 0))[2]
            e = unproc.get(c)
            if e is not None and (pd.isna(e["as_of_ts"]) or t >= e["as_of_ts"]):
                rem += sos_remaining(e, frac.get(c, 1.0), counted, past, c)[0]
            else:
                rem += max(0, int(est - counted))
        nd = state[cn][0] - (prev[0] if prev else 0)
        nr = state[cn][1] - (prev[1] if prev else 0)
        rows.append({
            "posted": t, "county": cn, "first": prev is None, "new_votes": nd + nr,
            "batch_dem_2p": nd / (nd + nr) if (nd + nr) > 0 else np.nan,
            "dem_2p": D / (D + R) if (D + R) > 0 else np.nan,
            "remaining": rem, "needed": needed_share(D, R, rem),
        })
    return pd.DataFrame(rows)
