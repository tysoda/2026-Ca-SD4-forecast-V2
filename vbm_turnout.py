"""
vbm_turnout.py
==============
Estimates implied final turnout per county from current pre-election VBM
returns, using historical calibration parameters derived from 2022 and 2024
California general elections.

Methodology
-----------
Four-step conversion:
  0. snapshot_vbm      →  pre_vbm             = snapshot_vbm / window_fraction
  1. pre_vbm           →  implied_final_vbm   = pre_vbm / return_rate
  2. implied_final_vbm →  implied_total_votes  = implied_final_vbm / vbm_pct_of_total
  3. implied_total_votes vs forecast_total     → turnout_adj (pp above/below forecast)

Where:
  return_rate      = pre-election VBM accepted ÷ final VBM accepted
                     (avg of 2022 and 2024; Placer uses 2024-only due to
                      methodology change in 2022)
  vbm_pct_of_total = final VBM accepted ÷ total ballots accepted
                     (avg of 2022 and 2024)

Uncertainty
-----------
A mid-window snapshot is extrapolated to the final pre-election total using
each county's own pace relative to 2022: window_fraction = vbm_accepted_2026 /
vbm_accepted_2022_final (from vbm_2022_baseline.csv).  This avoids the error
of applying the district-level 2022 return schedule uniformly — counties mail
out on different dates and voters return at different rates.  The district-level
curve (pdi_2022_daily_sd4.csv) is kept as a fallback when the baseline is
unavailable.  No estimate is produced until at least 30% of a 2022-sized early
vote (by issued ballots) is in.
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent

# ── Calibration parameters ────────────────────────────────────────────────────
# return_rate: pre-election VBM accepted / final VBM accepted
# vbm_pct:     final VBM accepted / total ballots accepted
# Source: 2022 and 2024 CA General Election SoS data.
# Placer 2022 return rate excluded (methodology change); uses 2024 only.

COUNTY_PARAMS = {
    #              return_rate  vbm_pct
    "Alpine":      (0.711,      1.000),
    "Amador":      (0.675,      0.916),
    "Calaveras":   (0.663,      0.864),
    "El Dorado":   (0.600,      0.910),
    "Inyo":        (0.538,      0.842),
    "Madera":      (0.583,      0.878),
    "Mariposa":    (0.608,      0.989),
    "Merced":      (0.504,      0.876),
    "Mono":        (0.560,      0.841),
    "Nevada":      (0.621,      0.898),
    "Placer":      (0.597,      0.812),   # 2024 only
    "Stanislaus":  (0.520,      0.897),
    "Tuolumne":    (0.644,      0.944),
}

# Minimum fraction of the 2022 early vote (2022 pre-election VBM returned,
# scaled to this cycle's issued ballots) that must be in before we produce an
# estimate. Early in the window the linear time-to-E-day extrapolation below
# understates back-loaded returns badly, so this is deliberately conservative.
MIN_OBSERVED_FRACTION = 0.30

# Early-vote window used to extrapolate a mid-window snapshot to the last
# pre-election report — now via the 2022 SD4 return curve (below).
MAILOUT_2026 = pd.Timestamp("2026-10-05")
EDAY_2026    = pd.Timestamp("2026-11-03")


PDI_2022_CSV = DATA_DIR / "pdi_2022_daily_sd4.csv"
_CURVE_CACHE: dict = {}


def _curve_2022() -> pd.Series | None:
    """Share of SD4's 2022 early vote in by each day before Election Day.

    From pdi_2022_daily_sd4.csv (2022 daily returns by party, read from PDI's
    tracker chart). Index = days before E-day; value = cumulative share (0–1).
    """
    if "curve" in _CURVE_CACHE:
        return _CURVE_CACHE["curve"]
    curve = None
    if PDI_2022_CSV.exists():
        d = pd.read_csv(PDI_2022_CSV).sort_values("days_before_eday", ascending=False)
        cum = d["total"].cumsum()
        if cum.iloc[-1] > 0:
            curve = pd.Series((cum / cum.iloc[-1]).values, index=d["days_before_eday"].values).sort_index()
    _CURVE_CACHE["curve"] = curve
    return curve


def early_vote_fraction_2022(days_before_eday: int) -> float:
    """Share of the 2022 early vote (to the last pre-election report) that was in
    by the same number of days before Election Day.

    Uses the real 2022 SD4 curve when pdi_2022_daily_sd4.csv is present; falls
    back to a straight line over the 2022 window (Oct 10 → Nov 8) otherwise.
    Both elections fall on a Tuesday, so days-out also lines up weekdays.
    """
    curve = _curve_2022()
    if curve is not None:
        if days_before_eday > curve.index.max():
            return 0.0
        if days_before_eday < curve.index.min():
            return 1.0
        return float(curve.loc[curve.index >= days_before_eday].iloc[0])
    window = (pd.Timestamp("2022-11-08") - pd.Timestamp("2022-10-10")).days
    return float(np.clip((window - days_before_eday) / window, 0.0, 1.0))


def window_fraction(snapshot_date, mailout=MAILOUT_2026, eday=EDAY_2026) -> float:
    """Share of the early vote expected in by snapshot_date (0–1), using the
    2022 SD4 return curve (see early_vote_fraction_2022)."""
    return early_vote_fraction_2022((pd.Timestamp(eday) - pd.Timestamp(snapshot_date)).days)


def get_latest_returns(ballot_returns_path: Path | None = None) -> pd.DataFrame:
    """
    Load ballot_returns.csv and return the most recent snapshot row per county.
    """
    path = ballot_returns_path or DATA_DIR / "ballot_returns.csv"
    df = pd.read_csv(path, parse_dates=["snapshot_date"])
    # Latest snapshot per county
    idx = df.groupby("county")["snapshot_date"].idxmax()
    return df.loc[idx].set_index("county")


def estimate_turnout(
    ballot_returns_path: Path | None = None,
    registered_voters: dict | None = None,
    min_fraction: float = MIN_OBSERVED_FRACTION,
    verbose: bool = True,
    baseline_path: Path | None = None,
) -> dict:
    """
    For each SD4 county, estimate implied final turnout from current VBM returns.

    Steps (per county):
      1. pre_election_vbm  = vbm_accepted / window_fraction(snapshot)
         (extrapolate this snapshot to the last pre-election report using the
          share of 2022's early vote that was in at the same days out)
      2. final_vbm         = pre_election_vbm / return_rate
      3. implied_total     = final_vbm / vbm_pct_of_total

    Progress gate: observed_fraction = vbm_returned / (vbm_issued × 2022
    pre-election return rate), i.e. how much of a 2022-sized early vote is in.
    This is independent of the calibration parameters, unlike the previous
    check (vbm_acc / (vbm_acc / return_rate)), which always equalled
    return_rate and so never flagged a county as too early.

    Parameters
    ----------
    ballot_returns_path : path to ballot_returns.csv (default: DATA_DIR)
    registered_voters   : unused; kept for backwards compatibility
    min_fraction        : minimum observed_fraction before producing an estimate
    baseline_path       : path to vbm_2022_baseline.csv (default: DATA_DIR)

    Returns
    -------
    dict keyed by county with keys:
        snapshot_date       : date of most recent return data
        vbm_accepted        : observed VBM accepted at that snapshot
        window_fraction     : share of the early-vote window elapsed
        expected_pre_vbm    : extrapolated pre-election VBM accepted
        expected_final_vbm  : implied final VBM (expected_pre_vbm / return_rate)
        implied_total       : implied total ballots cast
        vbm_pct_observed    : observed_fraction (progress indicator, see above)
        return_rate         : calibration return_rate used
        vbm_pct_of_total    : calibration vbm_pct used
        status              : "ok" | "too_early" | "no_data"
    """
    returns = get_latest_returns(ballot_returns_path)
    base_file = baseline_path or DATA_DIR / "vbm_2022_baseline.csv"
    base = pd.read_csv(base_file).set_index("county") if Path(base_file).exists() else None
    results = {}

    def _int(v):
        try:
            return int(v) if pd.notna(v) else 0
        except (ValueError, TypeError):
            return 0

    for county, (return_rate, vbm_pct) in COUNTY_PARAMS.items():
        if county not in returns.index:
            results[county] = {"status": "no_data"}
            continue

        row = returns.loc[county]
        vbm_acc = _int(row.get("vbm_accepted", 0))
        vbm_ret = _int(row.get("vbm_returned", 0))
        vbm_iss = _int(row.get("vbm_issued", 0))
        snapshot_date = row.get("snapshot_date", None)

        # ── County-specific window fraction ───────────────────────────────────
        # Instead of using the district-level 2022 return curve (the same for
        # every county), we compute each county's own pace: how much of its
        # 2022 final pre-election VBM has come in so far this cycle.
        #
        #   county_wf = vbm_accepted_2026 / vbm_accepted_2022_final
        #
        # This captures real county-level variation in mailing schedules and
        # voter behaviour.  The district-level curve is kept as a fallback when
        # the 2022 baseline is unavailable.
        b_acc_22 = 0
        b_iss = 0
        b_ret = 0
        pre_rate_22 = None
        if base is not None and county in base.index:
            b_acc_22    = _int(base.loc[county, "vbm_accepted_sd4"])
            b_iss       = _int(base.loc[county, "vbm_issued_sd4"])
            b_ret       = _int(base.loc[county, "vbm_returned_sd4"])
            pre_rate_22 = b_ret / b_iss if b_iss > 0 else None

        if b_acc_22 > 0:
            frac = vbm_acc / b_acc_22  # county-level pace
        else:
            frac = window_fraction(snapshot_date) if snapshot_date is not None else 0.0

        expected_pre = vbm_acc / frac if frac > 0 else 0
        expected_final_vbm = expected_pre / return_rate if return_rate > 0 else 0
        implied_total = expected_final_vbm / vbm_pct if vbm_pct > 0 else 0

        # Progress vs a 2022-sized early vote (unchanged — used for status gate)
        expected_early = vbm_iss * pre_rate_22 if pre_rate_22 else 0
        observed_fraction = vbm_ret / expected_early if expected_early > 0 else 0

        if vbm_acc == 0:
            status = "no_data"
        elif observed_fraction < min_fraction or frac == 0:
            status = "too_early"
        else:
            status = "ok"

        results[county] = {
            "snapshot_date":      snapshot_date,
            "vbm_accepted":       vbm_acc,
            "window_fraction":    round(frac, 4),
            "expected_pre_vbm":   round(expected_pre),
            "expected_final_vbm": round(expected_final_vbm),
            "implied_total":      round(implied_total),
            "vbm_pct_observed":   round(observed_fraction, 4),
            "return_rate":        return_rate,
            "vbm_pct_of_total":   vbm_pct,
            "status":             status,
        }

    if verbose:
        _print_summary(results)

    return results


def _print_summary(results: dict) -> None:
    print("\n── VBM Turnout Estimates ─────────────────────────────────────────────")
    print(f"{'County':<12} {'VBM Acc':>9} {'Exp Final VBM':>14} {'Imp Total':>10} "
          f"{'Progress':>9} {'Status'}")
    print("-" * 70)
    for county, r in results.items():
        if r["status"] == "no_data":
            print(f"{county:<12} {'—':>9} {'—':>14} {'—':>10} {'—':>9}  no data")
            continue
        prog = f"{r['vbm_pct_observed']:.1%}"
        print(f"{county:<12} {r['vbm_accepted']:>9,} {r['expected_final_vbm']:>14,} "
              f"{r['implied_total']:>10,} {prog:>9}  {r['status']}")


def turnout_vs_forecast(
    results: dict,
    forecast_totals: dict,
) -> dict:
    """
    Compare implied turnout to forecast totals and return pp adjustments.

    Parameters
    ----------
    results         : output of estimate_turnout()
    forecast_totals : dict of {county: forecast_total_votes}

    Returns
    -------
    dict of {county: adj_pp} where adj_pp is the implied turnout adjustment
    in percentage points relative to forecast (positive = higher than forecast).
    Only counties with status "ok" are included.
    """
    adjs = {}
    for county, r in results.items():
        if r["status"] != "ok":
            continue
        if county not in forecast_totals or forecast_totals[county] <= 0:
            continue
        diff = r["implied_total"] - forecast_totals[county]
        adj_pp = diff / forecast_totals[county]
        adjs[county] = round(adj_pp, 4)
    return adjs


if __name__ == "__main__":
    results = estimate_turnout(verbose=True)

    # Quick sanity check: show where we'd expect to be given current returns
    print("\n── Calibration reference ─────────────────────────────────────────────")
    print(f"{'County':<12} {'ReturnRate':>10} {'VBM%Total':>10}  (late mail %)")
    print("-" * 45)
    for county, (rr, vp) in COUNTY_PARAMS.items():
        print(f"{county:<12} {rr:>10.1%} {vp:>10.1%}  ({1-rr:.1%} arrive after election day)")
