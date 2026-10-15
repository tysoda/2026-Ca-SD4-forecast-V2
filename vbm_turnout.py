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
A mid-window snapshot is first extrapolated to the last pre-election report
assuming returns arrive linearly in time; real returns are back-loaded, so
early estimates run low. No estimate is produced until at least 25% of a
2022-sized early vote (by issued ballots) is in.
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
MIN_OBSERVED_FRACTION = 0.25

# Early-vote window used to extrapolate a mid-window snapshot to the last
# pre-election report. Mail-out ≈ first SOS snapshot; window ends the day
# before Election Day. Linear in time — the same assumption as the 2022 pace
# baseline on the Ballot Returns tab.
MAILOUT_2026 = pd.Timestamp("2026-10-05")
EDAY_2026    = pd.Timestamp("2026-11-03")


def window_fraction(snapshot_date, mailout=MAILOUT_2026, eday=EDAY_2026) -> float:
    """Share of the early-vote window elapsed at snapshot_date (0–1)."""
    window = (eday - pd.Timedelta(days=1) - mailout).days
    elapsed = (pd.Timestamp(snapshot_date) - mailout).days
    return float(np.clip(elapsed / window, 0.0, 1.0)) if window > 0 else 1.0


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
         (extrapolate this snapshot to the last pre-election report)
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

        frac = window_fraction(snapshot_date) if snapshot_date is not None else 0.0
        expected_pre = vbm_acc / frac if frac > 0 else 0
        expected_final_vbm = expected_pre / return_rate if return_rate > 0 else 0
        implied_total = expected_final_vbm / vbm_pct if vbm_pct > 0 else 0

        # Progress vs a 2022-sized early vote
        pre_rate_22 = None
        if base is not None and county in base.index:
            b_iss = _int(base.loc[county, "vbm_issued_sd4"])
            b_ret = _int(base.loc[county, "vbm_returned_sd4"])
            pre_rate_22 = b_ret / b_iss if b_iss > 0 else None
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
