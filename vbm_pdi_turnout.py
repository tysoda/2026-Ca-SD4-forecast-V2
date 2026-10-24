"""
vbm_pdi_turnout.py
==================
County-level VBM turnout estimates using PDI daily return curves.

Methodology
-----------
For each of the 9 PDI counties (those wholly within SD4), we:

  1. Build a pace curve from prior elections (22G, 24G, 26P):
     pace[day] = cumulative_returned_at_day / final_pre_election_total
     This tells us "at D-N, X% of the final pre-election VBM was in."

  2. Compute today's window fraction:
     wf = vbm_accepted_2026 / (vbm_issued_2026 × avg_pace[today_days_out])
     ... actually wf = vbm_accepted_2026 / implied_final_pre_election

  3. Project to final:
     implied_final_pre_vbm = vbm_accepted_2026 / pace[today_days_out]
     implied_final_vbm     = implied_final_pre_vbm / return_rate
     implied_total         = implied_final_vbm / vbm_pct_of_total

     where return_rate and vbm_pct_of_total come from COUNTY_PARAMS
     (same as vbm_turnout.py — calibrated from 2022/2024 SoS data).

  4. Uncertainty:
     For each prior election at the same days-out, compute what the
     implied_final_pre_vbm would have been and compare to what actually
     happened.  The std deviation of those prediction errors (as a
     fraction of the actual final) becomes sigma_frac, giving a ±1 SD
     range on implied_total.

  5. D/R split:
     Current 26G D%/R%/O% of returned VBM is taken as the partisan
     composition estimate.  Applied to implied_final_vbm to give
     implied_D_vbm, implied_R_vbm, implied_O_vbm.

For the 4 split counties (Madera, Merced, Nevada, Placer), no PDI county
curve exists (their IDs return whole-county totals).  We fall back to
vbm_turnout.py's logic (county-level window fraction from vbm_2022_baseline)
and use the district-average sigma.

Elections used for pace curves
-------------------------------
- 22G and 24G: general election priors (same cycle type as 26G)
- 26P: most recent same-district signal; also the source of D/R breakdown

25S is excluded — it is a special election with atypical turnout dynamics
and a very narrow geographic scope.
"""

from pathlib import Path
import csv
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent

# ── Calibration parameters (from vbm_turnout.py) ─────────────────────────────
# return_rate:    pre-election VBM accepted / final VBM accepted
# vbm_pct:        final VBM accepted / total ballots cast
COUNTY_PARAMS = {
    "Alpine":     (0.711, 1.000),
    "Amador":     (0.675, 0.916),
    "Calaveras":  (0.663, 0.864),
    "El Dorado":  (0.600, 0.910),
    "Inyo":       (0.538, 0.842),
    "Madera":     (0.583, 0.878),
    "Mariposa":   (0.608, 0.989),
    "Merced":     (0.504, 0.876),
    "Mono":       (0.560, 0.841),
    "Nevada":     (0.621, 0.898),
    "Placer":     (0.597, 0.812),
    "Stanislaus": (0.520, 0.897),
    "Tuolumne":   (0.644, 0.944),
}

# Counties with PDI daily curves (wholly within SD4)
PDI_COUNTIES = {
    "Alpine", "Amador", "Calaveras", "El Dorado", "Inyo",
    "Mariposa", "Mono", "Stanislaus", "Tuolumne",
}

# Prior elections to use for pace curves
PACE_ELECTIONS = ["22G", "24G", "26P"]

# Minimum window fraction (share of expected final pre-election VBM) before
# we produce an estimate
MIN_WF = 0.20


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_pdi(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0).astype(int)
    df["D"]     = pd.to_numeric(df["D"],     errors="coerce").fillna(0).astype(int)
    df["R"]     = pd.to_numeric(df["R"],     errors="coerce").fillna(0).astype(int)
    df["O"]     = pd.to_numeric(df["O"],     errors="coerce").fillna(0).astype(int)
    df["days_before_eday"] = df["days_before_eday"].astype(int)
    return df


def _load_ballot_returns(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["snapshot_date"])
    idx = df.groupby("county")["snapshot_date"].idxmax()
    return df.loc[idx].set_index("county")


def _load_baseline(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path).set_index("county")


# ── Pace curve construction ───────────────────────────────────────────────────

def build_pace_curves(pdi: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    """
    Returns {county: {election_code: pace_series}}
    where pace_series.index = days_before_eday (descending)
    and values = fraction of final pre-election total returned by that day.
    Final = value at the lowest available day (D-11 cutoff = most recent).
    """
    curves = {}
    for county in pdi["county"].unique():
        curves[county] = {}
        cdf = pdi[pdi["county"] == county]
        for code in PACE_ELECTIONS:
            edf = cdf[cdf["code"] == code].sort_values("days_before_eday", ascending=False)
            nz = edf[edf["total"] > 0]
            if len(nz) < 3:
                continue  # not enough data
            final = nz["total"].iloc[-1]  # smallest days_out = most complete
            if final == 0:
                continue
            pace = pd.Series(
                (nz["total"] / final).values,
                index=nz["days_before_eday"].values,
            ).sort_index(ascending=False)
            curves[county][code] = pace
    return curves


def avg_pace_at_day(curves_for_county: dict[str, pd.Series], days_out: int) -> float | None:
    """Average pace fraction at a given days_out across available elections."""
    fracs = []
    for code, pace in curves_for_county.items():
        # Find the nearest available day (same or later — more conservative)
        candidates = pace.index[pace.index >= days_out]
        if len(candidates) == 0:
            # All days are earlier than days_out — extrapolate using minimum
            fracs.append(float(pace.iloc[-1]))
        else:
            fracs.append(float(pace.loc[candidates[-1]]))
    return float(np.mean(fracs)) if fracs else None


# ── Uncertainty from historical back-testing ─────────────────────────────────

def compute_sigma(
    curves_for_county: dict[str, pd.Series],
    days_out: int,
) -> float:
    """
    For each prior election, ask: if we had used the pace-curve method at
    days_out, what error would we have made relative to the actual final?
    Return the std dev of those fractional errors.
    """
    errors = []
    for code, pace in curves_for_county.items():
        candidates = pace.index[pace.index >= days_out]
        if len(candidates) == 0:
            p = float(pace.iloc[-1])
        else:
            p = float(pace.loc[candidates[-1]])
        if p <= 0:
            continue
        # The "actual" final is pace=1.0 by definition; the prediction using
        # all other elections' average pace gives us how far off we'd be.
        # Here we measure the leave-one-out error: predict this election's
        # final using the other elections' average pace at that day.
        others = {k: v for k, v in curves_for_county.items() if k != code}
        pred_p = avg_pace_at_day(others, days_out)
        if pred_p is None or pred_p <= 0:
            continue
        # Fractional error: (predicted_final / actual_final) - 1
        # predicted_final = observed_at_day / pred_p
        # actual_final    = observed_at_day / p
        # ratio           = p / pred_p
        errors.append(p / pred_p - 1.0)
    if len(errors) < 2:
        return 0.08  # fallback: 8% if not enough elections to estimate
    return float(np.std(errors, ddof=1))


# ── Main estimation function ──────────────────────────────────────────────────

def estimate_turnout(
    pdi_path: Path | None = None,
    ballot_returns_path: Path | None = None,
    baseline_path: Path | None = None,
    min_wf: float = MIN_WF,
    verbose: bool = True,
) -> dict:
    """
    Estimate county-level implied total turnout from current VBM returns,
    using PDI daily curves for calibration.

    Returns dict keyed by county:
      implied_total       int    — point estimate of total ballots cast
      sigma_total         float  — 1-SD uncertainty (same units as implied_total)
      implied_D_vbm       int    — implied Dem VBM (PDI counties only, else None)
      implied_R_vbm       int    — implied Rep VBM (PDI counties only, else None)
      implied_O_vbm       int    — implied Other VBM (PDI counties only, else None)
      d_pct               float  — current D share of returned VBM
      r_pct               float  — current R share of returned VBM
      wf                  float  — window fraction used
      pace_elections      list   — elections that contributed to pace curve
      days_out            int    — days before E-day at snapshot date
      vbm_accepted        int    — observed VBM accepted
      expected_pre_vbm    int    — extrapolated pre-election VBM
      expected_final_vbm  int    — implied final VBM
      status              str    — "ok" | "too_early" | "no_data" | "fallback"
    """
    pdi_path    = pdi_path    or DATA_DIR / "pdi_county_daily.csv"
    br_path     = ballot_returns_path or DATA_DIR / "ballot_returns.csv"
    base_path   = baseline_path or DATA_DIR / "vbm_2022_baseline.csv"

    pdi      = _load_pdi(pdi_path)
    returns  = _load_ballot_returns(br_path)
    baseline = _load_baseline(base_path)

    # Election Day 2026-11-03
    eday = pd.Timestamp("2026-11-03")

    pace_curves = build_pace_curves(pdi)

    # District-average sigma (fallback for split counties)
    all_sigmas = []

    results = {}

    for county, (return_rate, vbm_pct) in COUNTY_PARAMS.items():
        if county not in returns.index:
            results[county] = {"status": "no_data"}
            continue

        row       = returns.loc[county]
        vbm_acc   = int(row.get("vbm_accepted", 0) or 0)
        snap_date = row.get("snapshot_date", None)

        if vbm_acc == 0:
            results[county] = {"status": "no_data"}
            continue

        days_out = int((eday - pd.Timestamp(snap_date)).days) if snap_date else None

        # ── PDI counties: use pace curves ─────────────────────────────────────
        if county in PDI_COUNTIES and county in pace_curves and days_out is not None:
            ccurves = pace_curves[county]
            wf = avg_pace_at_day(ccurves, days_out)

            if wf is None or wf < min_wf:
                results[county] = {
                    "status": "too_early",
                    "days_out": days_out,
                    "wf": wf,
                    "vbm_accepted": vbm_acc,
                    "pace_elections": list(ccurves.keys()),
                }
                continue

            sigma_frac = compute_sigma(ccurves, days_out)
            all_sigmas.append(sigma_frac)

            expected_pre  = round(vbm_acc / wf)
            expected_final_vbm = round(expected_pre / return_rate)
            implied_total = round(expected_final_vbm / vbm_pct)
            sigma_total   = round(implied_total * sigma_frac)

            # D/R split from current 26G snapshot
            cur = pdi[
                (pdi["county"] == county) &
                (pdi["code"] == "26G") &
                (pdi["days_before_eday"] == days_out)
            ]
            if cur.empty:
                # Use nearest available day
                avail = pdi[
                    (pdi["county"] == county) &
                    (pdi["code"] == "26G") &
                    (pdi["total"] > 0)
                ].sort_values("days_before_eday")
                cur = avail.iloc[[-1]] if not avail.empty else cur

            if not cur.empty and int(cur["total"].iloc[0]) > 0:
                d_pct = int(cur["D"].iloc[0]) / int(cur["total"].iloc[0])
                r_pct = int(cur["R"].iloc[0]) / int(cur["total"].iloc[0])
                o_pct = int(cur["O"].iloc[0]) / int(cur["total"].iloc[0])
                implied_D_vbm = round(expected_final_vbm * d_pct)
                implied_R_vbm = round(expected_final_vbm * r_pct)
                implied_O_vbm = round(expected_final_vbm * o_pct)
            else:
                d_pct = r_pct = o_pct = None
                implied_D_vbm = implied_R_vbm = implied_O_vbm = None

            results[county] = {
                "status":            "ok",
                "days_out":          days_out,
                "vbm_accepted":      vbm_acc,
                "wf":                round(wf, 4),
                "expected_pre_vbm":  expected_pre,
                "expected_final_vbm": expected_final_vbm,
                "implied_total":     implied_total,
                "sigma_total":       sigma_total,
                "sigma_frac":        round(sigma_frac, 4),
                "implied_D_vbm":     implied_D_vbm,
                "implied_R_vbm":     implied_R_vbm,
                "implied_O_vbm":     implied_O_vbm,
                "d_pct":             round(d_pct, 4) if d_pct is not None else None,
                "r_pct":             round(r_pct, 4) if r_pct is not None else None,
                "pace_elections":    list(ccurves.keys()),
                "return_rate":       return_rate,
                "vbm_pct_of_total":  vbm_pct,
            }

        # ── Split counties: fallback to 2022 baseline window fraction ─────────
        else:
            if baseline is None or county not in baseline.index:
                results[county] = {"status": "no_data"}
                continue

            b_acc_22 = int(baseline.loc[county, "vbm_accepted_sd4"] or 0)
            if b_acc_22 == 0:
                results[county] = {"status": "no_data"}
                continue

            wf = vbm_acc / b_acc_22
            if wf < min_wf:
                results[county] = {
                    "status": "too_early",
                    "days_out": days_out,
                    "wf": round(wf, 4),
                    "vbm_accepted": vbm_acc,
                    "pace_elections": [],
                }
                continue

            expected_pre  = round(vbm_acc / wf)
            expected_final_vbm = round(expected_pre / return_rate)
            implied_total = round(expected_final_vbm / vbm_pct)

            results[county] = {
                "status":            "fallback",
                "days_out":          days_out,
                "vbm_accepted":      vbm_acc,
                "wf":                round(wf, 4),
                "expected_pre_vbm":  expected_pre,
                "expected_final_vbm": expected_final_vbm,
                "implied_total":     implied_total,
                "sigma_total":       None,  # filled in below
                "sigma_frac":        None,
                "implied_D_vbm":     None,
                "implied_R_vbm":     None,
                "implied_O_vbm":     None,
                "d_pct":             None,
                "r_pct":             None,
                "pace_elections":    [],
                "return_rate":       return_rate,
                "vbm_pct_of_total":  vbm_pct,
            }

    # Fill district-average sigma into fallback counties
    district_sigma_frac = float(np.mean(all_sigmas)) if all_sigmas else 0.08
    for county, r in results.items():
        if r.get("status") == "fallback" and r.get("sigma_total") is None:
            r["sigma_frac"] = round(district_sigma_frac, 4)
            r["sigma_total"] = round(r["implied_total"] * district_sigma_frac)

    if verbose:
        _print_summary(results, district_sigma_frac)

    return results


# ── Console output ────────────────────────────────────────────────────────────

def _print_summary(results: dict, district_sigma_frac: float) -> None:
    print("\n── PDI-Calibrated VBM Turnout Estimates ──────────────────────────────────")
    print(f"{'County':<12} {'VBMacc':>7} {'WF':>6} {'ExpFinalVBM':>12} "
          f"{'ImpTotal':>10} {'±1SD':>7} {'D%':>5} {'R%':>5}  Status")
    print("-" * 90)
    for county, r in results.items():
        if r["status"] in ("no_data", "too_early"):
            wf_str = f"{r.get('wf') or 0:.1%}" if r.get('wf') else "—"
            print(f"{county:<12} {r.get('vbm_accepted',0):>7,} {wf_str:>6}  "
                  f"{'—':>12} {'—':>10} {'—':>7}  {r['status']}")
            continue
        d_str = f"{r['d_pct']:.0%}" if r.get("d_pct") else "—"
        r_str = f"{r['r_pct']:.0%}" if r.get("r_pct") else "—"
        sig_str = f"±{r['sigma_total']:,}" if r.get("sigma_total") else "—"
        fb = " (fallback)" if r["status"] == "fallback" else ""
        print(f"{county:<12} {r['vbm_accepted']:>7,} {r['wf']:>6.1%} "
              f"{r['expected_final_vbm']:>12,} {r['implied_total']:>10,} "
              f"{sig_str:>7}  {d_str:>5} {r_str:>5}{fb}")

    ok = [r for r in results.values() if r.get("status") in ("ok", "fallback")
          and r.get("implied_total")]
    if ok:
        total = sum(r["implied_total"] for r in ok)
        print(f"\n{'DISTRICT TOTAL':<12} {'':>7} {'':>6} {'':>12} {total:>10,}")

    print(f"\nDistrict-average sigma (used for fallback counties): "
          f"{district_sigma_frac:.1%}")
    print("Note: 4 split counties (Madera, Merced, Nevada, Placer) use 2022 baseline "
          "window fraction\n      and district-average uncertainty (no PDI county curve available).")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = estimate_turnout(verbose=True)

    print("\n── Pace curve elections available per county ─────────────────────────────")
    for county, r in results.items():
        elections = r.get("pace_elections", [])
        print(f"  {county:<12} {', '.join(elections) if elections else '(fallback/none)'}")
