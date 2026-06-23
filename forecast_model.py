"""
forecast_model.py
=================
Computes all SD4 election forecast parameters from historical CSV data.
Run this (or trigger via the Streamlit app) whenever new data is available.
Outputs forecast_params.json which is read by election_forecast.py.

Usage:
    python forecast_model.py [general] [approval=0.42] [inflation=3.2]
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent

# 2026 election context — update these as new information arrives
ELECTION_CONTEXT = {
    "year":                2026,
    "general":             False,   # False=primary, True=general
    "presidential":        False,   # midterm cycle
    "inflation":           3.8,     # CPI YoY % at time of election
    "approval":            0.39,    # Presidential net approval (0–1 scale)
    "trump_in_office":     True,
    "trump_on_ballot":     False,
}

COUNTIES = [
    "Alpine", "Amador", "Calaveras", "El Dorado", "Inyo",
    "Madera", "Mariposa", "Merced", "Mono", "Nevada",
    "Placer", "Stanislaus", "Tuolumne",
]

# Partial-county fractions: fraction of full county registration that is in SD4
# Applies to Madera, Merced, Nevada, Placer — others are 1.0
PARTIAL_COUNTIES = {"Madera", "Merced", "Nevada", "Placer"}


# ── OLS helper ────────────────────────────────────────────────────────────────
def ols(X: np.ndarray, y: np.ndarray):
    """OLS via normal equations. X should NOT include intercept column."""
    X_b = np.column_stack([np.ones(len(X)), X])
    coeffs, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
    y_hat = X_b @ coeffs
    resid = y - y_hat
    resid_std = float(np.std(resid, ddof=X_b.shape[1]))
    return coeffs, resid_std, resid


def predict_ols(coeffs: np.ndarray, x_row: list) -> float:
    return float(coeffs[0] + np.dot(coeffs[1:], x_row))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Registration
# Use the already-forecasted Nov 2026 values from registration_history.csv,
# adjusted by partial-county fractions where applicable.
# When 2026 Nov actuals become available, add them to the CSV and re-run.
# ══════════════════════════════════════════════════════════════════════════════
def forecast_registration() -> dict:
    df = pd.read_csv(DATA_DIR / "registration_history.csv")
    partial = pd.read_csv(DATA_DIR / "partial_county_fractions.csv")
    partial_dict = dict(zip(partial.county, partial.sd4_fraction))

    # Use Nov 2026 forecasted row; fall back to most recent Nov if not present
    results = {}
    for county in COUNTIES:
        c_df = df[df.county == county].copy()
        nov26 = c_df[(c_df.year == 2026) & (c_df.month == 11)]
        if len(nov26) > 0:
            reg_total = float(nov26.iloc[-1]["registered_voters"])
        else:
            # Fallback: linear extrapolation from last two Nov values
            nov_df = c_df[c_df.month == 11].sort_values("year")
            if len(nov_df) >= 2:
                last_two = nov_df.tail(2)
                slope = (last_two.iloc[1]["registered_voters"] -
                         last_two.iloc[0]["registered_voters"])
                reg_total = float(last_two.iloc[1]["registered_voters"] + slope)
            else:
                reg_total = float(nov_df.iloc[-1]["registered_voters"])

        fraction = partial_dict.get(county, 1.0)
        results[county] = {
            "predicted_registration_total": round(reg_total),
            "sd4_fraction":                 round(fraction, 8),
            "predicted_registration_sd4":   round(reg_total * fraction),
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Turnout forecast
# Panel OLS: county fixed effects + presidential dummy + general dummy
#            + previous election turnout (same election type)
# ══════════════════════════════════════════════════════════════════════════════
def forecast_turnout(ctx: dict) -> tuple:
    df = pd.read_csv(DATA_DIR / "historical_turnout.csv")
    df = df[df.votes_cast.notna()].copy()
    df = df.sort_values(["county", "year", "month"]).reset_index(drop=True)

    # Previous turnout: last election of the same type (primary or general) per county
    df["prev_turnout"] = (
        df.groupby(["county", "general"])["turnout_rate"]
        .shift(1)
    )
    df = df[df.prev_turnout.notna()].copy()

    # Set Tuolumne as reference category by putting it first after sorting
    county_order = ["Tuolumne"] + [c for c in sorted(df["county"].unique()) if c != "Tuolumne"]
    df["county"] = pd.Categorical(df["county"], categories=county_order, ordered=False)
    county_dummies = pd.get_dummies(df["county"], drop_first=True, dtype=float)
    dummy_cols = list(county_dummies.columns)

    X = pd.concat(
        [df[["presidential", "general", "prev_turnout"]], county_dummies], axis=1
    ).values
    y = df["turnout_rate"].values

    coeffs, resid_std, residuals = ols(X, y)
    coeff_names = (
        ["intercept", "presidential", "general", "prev_turnout"] + dummy_cols
    )

    is_general = bool(ctx["general"])
    results = {}

    for county in COUNTIES:
        # Most recent same-type turnout for this county
        same_type = df[(df.county == county) & (df.general == int(is_general))]
        if len(same_type) > 0:
            prev = float(same_type.iloc[-1]["turnout_rate"])
        else:
            prev = float(df[df.county == county]["turnout_rate"].mean())

        county_fe = {c: 0.0 for c in dummy_cols}
        if county in county_fe:
            county_fe[county] = 1.0

        x_row = (
            [float(ctx["presidential"]), float(is_general), prev]
            + [county_fe[c] for c in dummy_cols]
        )
        predicted = float(np.clip(predict_ols(coeffs, x_row), 0.01, 1.0))

        results[county] = {
            "predicted_turnout":  round(predicted, 6),
            "prev_turnout_used":  round(prev, 6),
        }

    return (
        results,
        round(float(resid_std), 6),
        {n: round(float(c), 8) for n, c in zip(coeff_names, coeffs)},
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: State environment forecast
# OLS: presidential dummy + inflation + presidential approval
# ══════════════════════════════════════════════════════════════════════════════
def forecast_state_environment(ctx: dict) -> dict:
    df = pd.read_csv(DATA_DIR / "state_environment_history.csv")

   # Add general dummy and collapse duplicate feature rows to avoid
    # artificial precision from same-year multi-race observations
    df["is_general"] = df["election"].str.contains("General").astype(float)
    df["feature_key"] = df[["presidential","inflation","approval","is_general"]].apply(tuple, axis=1)
    df_collapsed = df.groupby("feature_key", as_index=False).agg(
        dem_share=("dem_share","mean"),
        presidential=("presidential","first"),
        inflation=("inflation","first"),
        approval=("approval","first"),
        is_general=("is_general","first"),
    )
    X = df_collapsed[["presidential","inflation","approval","is_general"]].values
    y = df_collapsed["dem_share"].values
    coeffs, resid_std, residuals = ols(X, y)
    # Use forecast SE instead of residual SD to capture both residual
    # variance and parameter uncertainty (important for small samples)
    X_b = np.column_stack([np.ones(len(X)), X])
    n, p = X_b.shape
    mse = float(np.sum(residuals**2) / (n - p))
    # Forecast variance for the 2026 prediction point
    x_pred = np.array([1.0, float(ctx["presidential"]), float(ctx["inflation"]), float(ctx["approval"])])
    XtX_inv = np.linalg.inv(X_b.T @ X_b)
    forecast_var = mse * (1 + float(x_pred @ XtX_inv @ x_pred))
    resid_std = float(np.sqrt(forecast_var))
    coeff_names = ["intercept", "presidential", "inflation", "approval"]

    x_row = [float(ctx["presidential"]), float(ctx["inflation"]), float(ctx["approval"])]
    predicted = float(predict_ols(coeffs, x_row))

    # Back-test residuals for SD (already computed in ols)
    return {
        "predicted_state_env": round(predicted, 6),
        "state_env_sd":        round(float(resid_std), 6),
        "coefficients":        {n: round(float(c), 8) for n, c in zip(coeff_names, coeffs)},
        "n_obs":               int(len(y)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: County lean calculation
# lean = county_dem_share − state_dem_share, weighted average across elections
# Linear lean: OLS on election sequence to project trend forward
# SD: standard deviation of historical lean values around the weighted mean
# ══════════════════════════════════════════════════════════════════════════════
def forecast_county_leans() -> dict:
    df = pd.read_csv(DATA_DIR / "historical_voting.csv")

    # Ordered elections for linear regression (oldest to newest)
    ELECTION_ORDER = [
        "2022_ss4_pri", "2022_gov_pri", "2022_senate_pri",
        "2022_gov_gen", "2022_senate_gen",
        "2024_senate_pri", "2024_pres_gen", "2024_senate_gen",
    ]

    results = {}
    for county in COUNTIES:
        c_df = df[(df.county == county) & (df.state_dem_share.notna())].copy()

        # Weighted average lean
        total_w = c_df["weight"].sum()
        wavg = float((c_df["county_lean"] * c_df["weight"]).sum() / total_w) if total_w > 0 else float(c_df["county_lean"].mean())

        # Lean SD: std of historical leans
        lean_sd = float(c_df["county_lean"].std(ddof=1)) if len(c_df) > 1 else 0.02

        # Linear lean: OLS on election sequence
        c_ordered = c_df.copy()
        c_ordered["order"] = c_ordered["election"].apply(
            lambda e: ELECTION_ORDER.index(e) if e in ELECTION_ORDER else -1
        )
        c_ordered = c_ordered[c_ordered.order >= 0].sort_values("order")

        if len(c_ordered) >= 3:
            X_lin = c_ordered["order"].values.reshape(-1, 1).astype(float)
            y_lin = c_ordered["county_lean"].values
            lin_coeffs, _, _ = ols(X_lin, y_lin)
            # Predict at next election index
            lean_lin = float(predict_ols(lin_coeffs, [float(len(ELECTION_ORDER))]))
        else:
            lean_lin = wavg

        hist_avg = float(c_df["county_dem_share"].mean())

        results[county] = {
            "lean_avg":  round(wavg,     6),
            "lean_lin":  round(lean_lin, 6),
            "lean_sd":   round(lean_sd,  6),
            "hist_avg":  round(hist_avg, 6),
            "n_elections": int(len(c_df)),
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Turnout SDs
# District SD: SD of regression residuals from the turnout model (inter-election)
# County SD:   average within-county SD of residuals from the model
# These represent unexplained variance AFTER the model accounts for
# presidential/general/prev-turnout effects.
# ══════════════════════════════════════════════════════════════════════════════
def estimate_turnout_sds_from_model(ctx: dict) -> dict:
    df = pd.read_csv(DATA_DIR / "historical_turnout.csv")
    df = df[df.votes_cast.notna()].copy()
    df = df.sort_values(["county", "year", "month"]).reset_index(drop=True)
    df["prev_turnout"] = df.groupby(["county", "general"])["turnout_rate"].shift(1)
    df = df[df.prev_turnout.notna()].copy()

    county_dummies = pd.get_dummies(df["county"], drop_first=True, dtype=float)
    X = pd.concat([df[["presidential", "general", "prev_turnout"]], county_dummies], axis=1).values
    y = df["turnout_rate"].values
    coeffs, _, residuals = ols(X, y)

    # District SD = overall residual SD (how much elections deviate from model)
    district_sd = float(np.std(residuals, ddof=X.shape[1]))

    # County SD = mean of per-county residual SD
    df["resid"] = residuals
    county_sd = float(df.groupby("county")["resid"].std(ddof=1).mean())

    return {
        "district_turnout_sd": round(district_sd, 6),
        "county_turnout_sd":   round(county_sd,   6),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run_forecast(ctx: dict = None, verbose: bool = True) -> dict:
    if ctx is None:
        ctx = dict(ELECTION_CONTEXT)

    if verbose:
        print("Running SD4 Election Forecast Model...")
        print(f"  Context: {ctx['year']} {'General' if ctx['general'] else 'Primary'}")
        print(f"  Inflation: {ctx['inflation']}%  Approval: {ctx['approval']:.0%}")

    reg     = forecast_registration()
    turnout, _, turnout_coeffs = forecast_turnout(ctx)
    state   = forecast_state_environment(ctx)
    leans   = forecast_county_leans()
    sds     = estimate_turnout_sds_from_model(ctx)

    params = {
        "context":          ctx,
        "state_environment": state,
        "turnout_sds":      sds,
        "turnout_coefficients": turnout_coeffs,
        "counties":         {},
    }

    for county in COUNTIES:
        params["counties"][county] = {
            "registration": reg[county]["predicted_registration_sd4"],
            "turnout":      turnout[county]["predicted_turnout"],
            "lean_avg":     leans[county]["lean_avg"],
            "lean_lin":     leans[county]["lean_lin"],
            "lean_sd":      leans[county]["lean_sd"],
            "hist_avg":     leans[county]["hist_avg"],
            "sd4_fraction": reg[county]["sd4_fraction"],
        }
        if verbose:
            print(
                f"  {county:12s}  reg={reg[county]['predicted_registration_sd4']:>7,}"
                f"  turnout={turnout[county]['predicted_turnout']:.3f}"
                f"  lean_avg={leans[county]['lean_avg']:+.3f}"
                f"  lean_sd={leans[county]['lean_sd']:.3f}"
            )

    if verbose:
        print(f"\n  State env:         {state['predicted_state_env']:.4f}"
              f" (SD={state['state_env_sd']:.4f})")
        print(f"  District turnout SD: {sds['district_turnout_sd']:.4f}")
        print(f"  County turnout SD:   {sds['county_turnout_sd']:.4f}")

    out_path = DATA_DIR / "forecast_params.json"
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    if verbose:
        print(f"\nParameters written to {out_path}")

    return params


if __name__ == "__main__":
    import sys
    ctx = dict(ELECTION_CONTEXT)
    for arg in sys.argv[1:]:
        if arg in ("general", "nov"):
            ctx["general"] = True
        elif arg in ("primary", "jun"):
            ctx["general"] = False
        elif "=" in arg:
            k, v = arg.split("=", 1)
            if k in ctx:
                try:
                    ctx[k] = float(v)
                except ValueError:
                    ctx[k] = v.lower() == "true"
    run_forecast(ctx)
