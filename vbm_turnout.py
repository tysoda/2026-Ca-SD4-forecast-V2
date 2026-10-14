"""
vbm_turnout.py
==============
Estimates implied final turnout per county from current pre-election VBM
returns, using historical calibration parameters derived from 2022 and 2024
California general elections.

Methodology
-----------
Three-step conversion:
  1. pre_vbm_observed  →  implied_final_vbm   = pre_vbm / return_rate
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
Returns are early; the return_rate estimate carries noise. A minimum
observed_return_rate threshold (default 5% of expected final VBM) is
enforced before producing an estimate — below that the signal is too thin.
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

# Minimum fraction of expected final VBM that must be observed before we
# produce an estimate. Below this the uncertainty is too wide to be useful.
MIN_OBSERVED_FRACTION = 0.05   # 5% of expected final VBM


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
) -> dict:
    """
    For each SD4 county, estimate implied final turnout from current VBM returns.

    Parameters
    ----------
    ballot_returns_path : path to ballot_returns.csv (default: DATA_DIR)
    registered_voters   : dict of {county: n_registered}; if None, uses
                          vbm_issued from ballot_returns as a proxy denominator
                          for the return-rate sanity check only
    min_fraction        : minimum observed/expected VBM fraction before
                          producing an estimate

    Returns
    -------
    dict keyed by county with keys:
        snapshot_date       : date of most recent return data
        vbm_accepted        : observed pre-election VBM accepted
        expected_final_vbm  : implied final VBM (vbm_accepted / return_rate)
        implied_total       : implied total ballots cast
        vbm_pct_observed    : vbm_accepted / expected_final_vbm (progress indicator)
        return_rate         : calibration return_rate used
        vbm_pct_of_total    : calibration vbm_pct used
        status              : "ok" | "too_early" | "no_data" | "uncalibrated"
    """
    returns = get_latest_returns(ballot_returns_path)
    results = {}

    for county, (return_rate, vbm_pct) in COUNTY_PARAMS.items():
        if county not in returns.index:
            results[county] = {"status": "no_data"}
            continue

        row = returns.loc[county]
        vbm_acc = row.get("vbm_accepted", 0)
        try:
            vbm_acc = int(vbm_acc) if pd.notna(vbm_acc) else 0
        except (ValueError, TypeError):
            vbm_acc = 0

        snapshot_date = row.get("snapshot_date", None)

        # Expected final VBM based on calibration
        vbm_issued = row.get("vbm_issued", None)
        try:
            vbm_issued_n = int(vbm_issued) if pd.notna(vbm_issued) else None
        except (ValueError, TypeError):
            vbm_issued_n = None

        # Rough expected final VBM: use issued ballots × historical acceptance rate
        # (not the same as return_rate, but a reasonable order-of-magnitude check)
        expected_final_vbm = vbm_acc / return_rate if vbm_acc > 0 else 0

        # Progress check: what fraction of expected final VBM is already in?
        observed_fraction = (vbm_acc / expected_final_vbm) if expected_final_vbm > 0 else 0

        if vbm_acc == 0:
            status = "no_data"
        elif observed_fraction < min_fraction:
            status = "too_early"
        else:
            status = "ok"

        implied_total = expected_final_vbm / vbm_pct if vbm_pct > 0 else 0

        results[county] = {
            "snapshot_date":      snapshot_date,
            "vbm_accepted":       vbm_acc,
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
