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
    "general":             True,   # False=primary, True=general
    "presidential":        False,   # midterm cycle
    "inflation":           3.8,     # CPI YoY % at time of election
    "approval":            0.39,    # Presidential net approval (0–1 scale)
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

        # 2010 general turnout (used as lag for 2012 primaries, county alphabetical order)
    TURNOUT_2010 = {
        "Alpine":     0.7722, "Amador":     0.7758, "Calaveras":  0.7000,
        "El Dorado":  0.7284, "Inyo":       0.7574, "Madera":     0.6383,
        "Mariposa":   0.7371, "Merced":     0.5094, "Mono":       0.7176,
        "Nevada":     0.8083, "Placer":     0.7156, "Stanislaus": 0.5346,
        "Tuolumne":   0.7160,
    }

    df["prev_turnout"] = df.groupby("county")["turnout_rate"].shift(1)

    # Fill NaN prev_turnout for the first row of each county (2012 Jun primary)
    # using the 2010 general election turnout
    def fill_2010(row):
        if pd.isna(row["prev_turnout"]) and row["year"] == 2012:
            return TURNOUT_2010.get(row["county"], np.nan)
        return row["prev_turnout"]
    df["prev_turnout"] = df.apply(fill_2010, axis=1)
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
        # Most recent election of any type for this county
        prev_any = df[df.county == county]
        if len(prev_any) > 0:
            prev = float(prev_any.iloc[-1]["turnout_rate"])
        else:
            prev = float(df["turnout_rate"].mean())

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
    features = ["presidential", "inflation", "approval"]

    # Extract year from election name for year-by-year back-test
    df["year_inferred"] = df["election"].str.extract(r"(\d{4})").astype(int)
    test_years = sorted(df[df["year_inferred"] >= 2016]["year_inferred"].unique())

    # Back-test: for each year from 2016 onwards, fit on all prior years,
    # predict all elections in the current year. This matches the spreadsheet method.
    errors = []
    for yr in test_years:
        train_df = df[df["year_inferred"] < yr]
        test_df  = df[df["year_inferred"] == yr]
        X_tr = np.column_stack([np.ones(len(train_df)), train_df[features].values])
        y_tr = train_df["dem_share"].values
        coeffs_i, _, _, _ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        for _, row in test_df.iterrows():
            x = np.array([1.0] + [float(row[f]) for f in features])
            pred = float(x @ coeffs_i)
            errors.append(float(row["dem_share"]) - pred)

    walk_fwd_sd = float(np.std(errors, ddof=1))

    # Full-sample fit for the actual 2026 forecast
    X = df[features].values
    y = df["dem_share"].values
    X_b = np.column_stack([np.ones(len(X)), X])
    coeffs, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
    coeff_names = ["intercept"] + features

    x_row = [float(ctx["presidential"]), float(ctx["inflation"]), float(ctx["approval"])]
    predicted = float(predict_ols(coeffs, x_row))

    return {
        "predicted_state_env":   round(predicted, 6),
        "state_env_sd":          round(walk_fwd_sd, 6),
        "coefficients":          {n: round(float(c), 8) for n, c in zip(coeff_names, coeffs)},
        "n_obs":                 int(len(df)),
        "n_walk_fwd_errors":     int(len(errors)),
    }
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
        "2024_senate_pri", "2024_pres_gen", "2024_senate_gen", "2026_ss4_pri", "2026_gov_pri"
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

        c_ordered_clean = c_ordered[c_ordered["county_lean"].notna()]
        if len(c_ordered_clean) >= 3:
            X_lin = c_ordered_clean["order"].values.reshape(-1, 1).astype(float)
            y_lin = c_ordered_clean["county_lean"].values
            lin_coeffs, _, _ = ols(X_lin, y_lin)
            lean_lin = float(predict_ols(lin_coeffs, [float(len(ELECTION_ORDER))]))
            lean_lin_incomplete = len(c_ordered_clean) < len(c_ordered)
        else:
            lean_lin = wavg
            lean_lin_incomplete = True

        hist_avg = float(c_df["county_dem_share"].mean())

        results[county] = {
            "lean_avg":              round(wavg,     6),
            "lean_lin":              round(lean_lin, 6),
            "lean_lin_incomplete":   lean_lin_incomplete,
            "lean_sd":               round(lean_sd,  6),
            "hist_avg":              round(hist_avg, 6),
            "n_elections":           int(len(c_df)),
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
    df = df.sort_values(["county","year","month"]).reset_index(drop=True)
    df["prev_turnout"] = df.groupby(["county","general"])["turnout_rate"].shift(1)
    df = df[df.prev_turnout.notna()].copy()

    # Only use 2022 onwards for walk-forward (matching your approach)
    df_wf = df[df.year >= 2022].copy()

    county_order = ["Tuolumne"] + [c for c in sorted(df["county"].unique()) if c != "Tuolumne"]
    features = ["presidential","general","prev_turnout"]

    # Walk-forward: for each post-2022 election, train on all prior data
    all_errors = []
    county_errors = {cn: [] for cn in df["county"].unique()}

    for idx in df_wf.index:
        row = df_wf.loc[idx]
        train = df[df.index < idx]
        if len(train) < 5:
            continue
        train_cat = train.copy()
        train_cat["county_cat"] = pd.Categorical(train_cat["county"], categories=county_order)
        dummies = pd.get_dummies(train_cat["county_cat"], drop_first=True, dtype=float)
        X_tr = pd.concat([train_cat[features], dummies], axis=1).values
        X_tr = np.column_stack([np.ones(len(X_tr)), X_tr])
        y_tr = train["turnout_rate"].values
        coeffs_i, _, _, _ = np.linalg.lstsq(X_tr, y_tr, rcond=None)

        # Build test feature vector
        cn = row["county"]
        fe_vec = {c: 0.0 for c in dummies.columns}
        if cn in fe_vec:
            fe_vec[cn] = 1.0
        x_te = np.array([1.0, float(row["presidential"]), float(row["general"]),
                         float(row["prev_turnout"])] + [fe_vec[c] for c in dummies.columns])
        if len(x_te) != len(coeffs_i):
            continue
        pred  = float(x_te @ coeffs_i)
        err   = float(row["turnout_rate"]) - pred
        walk_fwd_coefficients = []
        for idx in df_wf.index:
            # ... existing prediction code ...
            pred  = float(x_te @ coeffs_i)
            err   = float(row["turnout_rate"]) - pred
            all_errors.append(err)
            county_errors[cn].append(err)

            # Store coefficients for this step
            coeff_names_i = ["intercept","presidential","general","prev_turnout"] + list(dummies.columns)
            walk_fwd_coefficients.append({
                "year":        int(row["year"]),
                "month":       str(row["month"]),
                "county":      cn,
                "type":        "General" if row["general"] else "Primary",
                "actual":      round(float(row["turnout_rate"]), 6),
                "predicted":   round(pred, 6),
                "error":       round(err, 6),
                "coefficients": {n: round(float(c), 8) for n, c in zip(coeff_names_i, coeffs_i)},
            })
        all_errors.append(err)
        county_errors[cn].append(err)

    district_sd = float(np.std(all_errors, ddof=1)) if len(all_errors) > 1 else 0.082
    county_sd_vals = [np.std(v, ddof=1) for v in county_errors.values() if len(v) > 1]
    county_sd = float(np.mean(county_sd_vals)) if county_sd_vals else 0.077

    return {
        "district_turnout_sd":    round(district_sd, 6),
        "county_turnout_sd":      round(county_sd,   6),
        "n_walk_fwd_errors":      int(len(all_errors)),
        "walk_fwd_predictions":   walk_fwd_coefficients,
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
        "context":               ctx,
        "state_environment":     state,
        "turnout_sds":           sds,
        "turnout_coefficients":  turnout_coeffs,
        "walk_fwd_predictions":  sds.get("walk_fwd_predictions", []),
        "counties":              {},
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
