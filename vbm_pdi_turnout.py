"""
vbm_pdi_turnout.py
==================
County-level VBM turnout estimates anchored to the 2022 SoS baseline,
with D/R partisan splits from PDI daily data.

Methodology
-----------
Window fraction (all counties):
    wf = vbm_accepted_2026 / vbm_accepted_2022_sd4
    (from ballot_returns.csv and vbm_2022_baseline.csv)

    This is the fraction of 2022's final pre-election VBM that has come in
    so far in 2026.  The 2022 SoS SD4-slice figure is the correct denominator
    for all 13 counties — it is the same scale as ballot_returns.csv.

Projection:
    implied_pre_vbm   = vbm_accepted_2026 / wf
                      = vbm_accepted_2022_sd4   (tautologically, until 2026 > 2022)
    implied_final_vbm = implied_pre_vbm / return_rate
    implied_total     = implied_final_vbm / vbm_pct_of_total

Uncertainty:
    Cross-election sigma: we compute the same wf-based projection for 22G
    and 24G (using each election's own D-11 PDI total vs the SoS 22G anchor),
    compare to the SoS final, and take the std dev of fractional errors.
    For the 4 split counties, we use the district-average sigma.

D/R split (PDI counties only):
    Current 26G D%/R%/O% from pdi_county_daily.csv applied to implied_final_vbm.
    26P used as a secondary signal where 26G data is sparse.

PDI vs SoS note:
    PDI county IDs capture ~45-55% of the true SD4 VBM for most counties
    (PDI's 22G county final ≈ half the SoS vbm_accepted_sd4).  PDI pace
    fractions are therefore not useful for absolute-count projection; they
    are used only for D/R breakdown.  The 4 split counties (Madera, Merced,
    Nevada, Placer) have no useful PDI county data at all.
"""

from pathlib import Path
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

# Counties with PDI D/R data (wholly within SD4)
PDI_COUNTIES = {
    "Alpine", "Amador", "Calaveras", "El Dorado", "Inyo",
    "Mariposa", "Mono", "Stanislaus", "Tuolumne",
}

# Minimum window fraction before we produce an estimate
MIN_WF = 0.20


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_pdi(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["total", "D", "R", "O"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
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


# ── Uncertainty: cross-election sigma ────────────────────────────────────────

def compute_county_sigmas(
    baseline: pd.DataFrame,
    turnout_history_path: Path,
) -> dict[str, float]:
    """
    For each county, estimate fractional uncertainty as std dev of
    (model_predicted_total / actual_total - 1) across 2022 and 2024 generals.

    The model formula is: implied_total = sos_22g / return_rate / vbm_pct.
    This is a fixed prediction (it always predicts the same number regardless
    of current wf), so the error captures how much the structural parameters
    (return_rate, vbm_pct) drift between elections.

    Returns {county: sigma_frac} with 0.08 default for missing data.
    """
    if not turnout_history_path.exists():
        return {}

    hist = pd.read_csv(turnout_history_path)
    # Keep November generals only
    hist = hist[(hist["general"] == 1) & (hist["month"] == "Nov")].copy()
    hist = hist.set_index(["county", "year"])

    sigmas: dict[str, float] = {}
    for county, (return_rate, vbm_pct) in COUNTY_PARAMS.items():
        if county not in baseline.index:
            continue
        sos_22g = int(baseline.loc[county, "vbm_accepted_sd4"] or 0)
        if sos_22g == 0:
            continue
        model_total = (sos_22g / return_rate) / vbm_pct

        errors = []
        for year in [2022, 2024]:
            if (county, year) not in hist.index:
                continue
            actual = float(hist.loc[(county, year), "votes_cast"])
            if actual > 0:
                errors.append(model_total / actual - 1.0)

        if len(errors) >= 2:
            sigmas[county] = float(np.std(errors, ddof=1))
        elif len(errors) == 1:
            sigmas[county] = abs(errors[0])  # single-point: use magnitude
        # else: use district average (filled in later)
    return sigmas


# ── D/R split helper ─────────────────────────────────────────────────────────

def _get_pdi_splits(
    pdi: pd.DataFrame,
    county: str,
    days_out: int,
) -> tuple[float | None, float | None, float | None]:
    """
    Return (d_pct, r_pct, o_pct) from the most recent 26G PDI row for this
    county, falling back to 26P if 26G is missing.
    """
    for code in ("26G", "26P"):
        avail = pdi[
            (pdi["county"] == county) &
            (pdi["code"] == code) &
            (pdi["total"] > 0)
        ].sort_values("days_before_eday")
        if avail.empty:
            continue
        # Prefer exact match on days_out, else take the lowest days_before_eday row
        exact = avail[avail["days_before_eday"] == days_out]
        row = exact.iloc[0] if not exact.empty else avail.iloc[0]
        total = int(row["total"])
        if total > 0:
            return (
                int(row["D"]) / total,
                int(row["R"]) / total,
                int(row["O"]) / total,
            )
    return None, None, None


# ── Main estimation function ──────────────────────────────────────────────────

def estimate_turnout(
    pdi_path: Path | None = None,
    ballot_returns_path: Path | None = None,
    baseline_path: Path | None = None,
    min_wf: float = MIN_WF,
    verbose: bool = True,
) -> dict:
    """
    Estimate county-level implied total turnout from current VBM returns.

    Window fraction for ALL counties:
        wf = vbm_accepted_2026 / vbm_accepted_2022_sd4

    Projection for ALL counties:
        implied_pre_vbm   = vbm_accepted_2022_sd4   (= vbm_acc / wf, tautological)
        implied_final_vbm = implied_pre_vbm / return_rate
        implied_total     = implied_final_vbm / vbm_pct

    D/R split (PDI counties only):
        Current 26G (or 26P fallback) D%/R%/O% applied to implied_final_vbm.

    Uncertainty:
        Cross-election sigma from compute_sigma() for PDI counties.
        District-average sigma for the 4 split counties.

    Returns dict keyed by county with keys:
      status, days_out, vbm_accepted, wf,
      expected_pre_vbm, expected_final_vbm, implied_total, sigma_total, sigma_frac,
      implied_D_vbm, implied_R_vbm, implied_O_vbm, d_pct, r_pct,
      return_rate, vbm_pct_of_total
    """
    pdi_path  = pdi_path            or DATA_DIR / "pdi_county_daily.csv"
    br_path   = ballot_returns_path or DATA_DIR / "ballot_returns.csv"
    base_path = baseline_path       or DATA_DIR / "vbm_2022_baseline.csv"
    hist_path = DATA_DIR / "historical_turnout.csv"

    pdi      = _load_pdi(pdi_path)
    returns  = _load_ballot_returns(br_path)
    baseline = _load_baseline(base_path)

    eday = pd.Timestamp("2026-11-03")

    # Pre-compute per-county sigmas from historical elections
    county_sigmas: dict[str, float] = {}
    if baseline is not None:
        county_sigmas = compute_county_sigmas(baseline, hist_path)

    results: dict = {}

    for county, (return_rate, vbm_pct) in COUNTY_PARAMS.items():
        # ── Basic data availability ───────────────────────────────────────────
        if county not in returns.index:
            results[county] = {"status": "no_data"}
            continue

        row     = returns.loc[county]
        vbm_acc = int(row.get("vbm_accepted", 0) or 0)
        snap    = row.get("snapshot_date", None)

        if vbm_acc == 0:
            results[county] = {"status": "no_data"}
            continue

        days_out = int((eday - pd.Timestamp(snap)).days) if snap else None

        if baseline is None or county not in baseline.index:
            results[county] = {"status": "no_data"}
            continue

        sos_22g = int(baseline.loc[county, "vbm_accepted_sd4"] or 0)
        if sos_22g == 0:
            results[county] = {"status": "no_data"}
            continue

        # ── Window fraction ───────────────────────────────────────────────────
        wf = vbm_acc / sos_22g

        if wf < min_wf:
            results[county] = {
                "status":       "too_early",
                "days_out":     days_out,
                "wf":           round(wf, 4),
                "vbm_accepted": vbm_acc,
            }
            continue

        # ── Projection ────────────────────────────────────────────────────────
        # implied_pre_vbm = vbm_acc / wf = sos_22g (exact, by construction)
        implied_pre       = sos_22g
        implied_final_vbm = round(implied_pre / return_rate)
        implied_total     = round(implied_final_vbm / vbm_pct)

        # ── Uncertainty ───────────────────────────────────────────────────────
        is_pdi_county = county in PDI_COUNTIES
        sigma_frac = county_sigmas.get(county)  # None for split counties
        sigma_total = round(implied_total * sigma_frac) if sigma_frac else None

        # ── D/R splits (PDI counties only) ────────────────────────────────────
        if is_pdi_county and days_out is not None:
            d_pct, r_pct, o_pct = _get_pdi_splits(pdi, county, days_out)
        else:
            d_pct = r_pct = o_pct = None

        implied_D_vbm = round(implied_final_vbm * d_pct) if d_pct is not None else None
        implied_R_vbm = round(implied_final_vbm * r_pct) if r_pct is not None else None
        implied_O_vbm = round(implied_final_vbm * o_pct) if o_pct is not None else None

        status = "ok" if is_pdi_county else "fallback"
        results[county] = {
            "status":             status,
            "days_out":           days_out,
            "vbm_accepted":       vbm_acc,
            "wf":                 round(wf, 4),
            "expected_pre_vbm":   implied_pre,
            "expected_final_vbm": implied_final_vbm,
            "implied_total":      implied_total,
            "sigma_total":        sigma_total,
            "sigma_frac":         round(sigma_frac, 4) if sigma_frac else None,
            "implied_D_vbm":      implied_D_vbm,
            "implied_R_vbm":      implied_R_vbm,
            "implied_O_vbm":      implied_O_vbm,
            "d_pct":              round(d_pct, 4) if d_pct is not None else None,
            "r_pct":              round(r_pct, 4) if r_pct is not None else None,
            "return_rate":        return_rate,
            "vbm_pct_of_total":   vbm_pct,
        }

    # ── Fill district-average sigma into split counties ───────────────────────
    known = [v for v in county_sigmas.values() if v > 0]
    district_sigma_frac = float(np.mean(known)) if known else 0.08
    for r in results.values():
        if r.get("status") == "fallback" and r.get("sigma_total") is None:
            sf = district_sigma_frac
            r["sigma_frac"]  = round(sf, 4)
            r["sigma_total"] = round(r["implied_total"] * sf)

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
