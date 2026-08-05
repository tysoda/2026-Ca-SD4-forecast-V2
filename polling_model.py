"""
polling_model.py
================
Fits a poll accuracy regression on historical CA statewide polls and uses it
to weight current polls in a Bayesian blend with the structural model forecast.

Inputs:
    historical_polls.csv   — all historical polls with abs_dem_error column
    current_polls.csv      — polls for the current election cycle
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent

PARTISAN_DISCOUNT = 0.5
MIN_WEIGHT = 0.01


# ── MAE regression ─────────────────────────────────────────────────────────────

def fit_mae_regression(df: pd.DataFrame) -> np.ndarray:
    df2 = df[df["abs_dem_error"].notna() & df["days_out"].notna()].copy()
    X = np.column_stack([
        np.ones(len(df2)),
        df2["rv"].values,
        df2["lv"].values,
        df2["days_out"].values,
    ])
    y = df2["abs_dem_error"].values
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs


def predict_mae(coeffs: np.ndarray, rv: int, lv: int, days_out: float) -> float:
    raw = coeffs[0] + coeffs[1]*rv + coeffs[2]*lv + coeffs[3]*days_out
    return max(raw, 0.01)


# ── Partisan detection ─────────────────────────────────────────────────────────

def is_partisan(source: str) -> bool:
    return bool(source) and ("(D)" in source or "(R)" in source or
                             "(D-" in source or "(R-" in source)


# ── Sampling SD helper ─────────────────────────────────────────────────────────

def _sampling_sd(row) -> float:
    """
    Compute sampling SD for a poll row.
    Uses reported MoE (as decimal, e.g. 0.033) if available,
    otherwise falls back to theoretical SD from sample size.
    """
    moe_val = row.get("moe", None)
    try:
        moe_float = float(moe_val)
        if not pd.isna(moe_float) and moe_float > 0:
            return moe_float / 2   # MoE is 95% CI half-width → SD = MoE/2
    except (TypeError, ValueError):
        pass
    # Fallback: theoretical SD from sample size
    n = max(int(row.get("sample_size", 600)), 1)
    p = float(row.get("dem", 0.5)) if pd.notna(row.get("dem", None)) else 0.5
    return ((p * (1 - p)) / n) ** 0.5


# ── Poll weighting ─────────────────────────────────────────────────────────────

def compute_poll_weights(
    polls: pd.DataFrame,
    coeffs: np.ndarray,
    partisan_discount: float = PARTISAN_DISCOUNT,
) -> pd.Series:
    weights = []
    for _, row in polls.iterrows():
        mae = predict_mae(
            coeffs,
            int(row.get("rv", 0)),
            int(row.get("lv", 0)),
            float(row.get("days_out", 120)),
        )
        ssd = _sampling_sd(row)
        total_var = mae ** 2 + ssd ** 2
        w = 1.0 / total_var
        if is_partisan(str(row.get("source", ""))):
            w *= partisan_discount
        weights.append(max(w, MIN_WEIGHT))
    return pd.Series(weights, index=polls.index)


# ── Environment blending ───────────────────────────────────────────────────────

def blend_with_polls(
    model_env: float,
    model_sd: float,
    current_polls: pd.DataFrame,
    coeffs: np.ndarray,
) -> tuple:
    if len(current_polls) == 0:
        return model_env, model_sd, []

    polls = current_polls[current_polls["dem"].notna()].copy()
    if len(polls) == 0:
        return model_env, model_sd, []

    weights = compute_poll_weights(polls, coeffs)

    model_prec = 1.0 / (model_sd ** 2)
    poll_prec  = weights.sum()
    poll_mean  = float((polls["dem"] * weights).sum() / poll_prec)

    total_prec  = model_prec + poll_prec
    blended_env = (model_prec * model_env + poll_prec * poll_mean) / total_prec
    blended_sd  = float((1.0 / total_prec) ** 0.5)

    # Per-poll diagnostics
    details = []
    for (_, row), w in zip(polls.iterrows(), weights):
        mae = predict_mae(
            coeffs,
            int(row.get("rv", 0)),
            int(row.get("lv", 0)),
            float(row.get("days_out", 120)),
        )
        ssd = _sampling_sd(row)
        moe_val = row.get("moe", None)
        try:
            moe_display = float(moe_val) if moe_val is not None and pd.notna(moe_val) else None
        except (TypeError, ValueError):
            moe_display = None

        details.append({
            "source":      str(row.get("source", "")),
            "dem":         float(row["dem"]),
            "days_out":    float(row.get("days_out", 120)),
            "rv":          int(row.get("rv", 0)),
            "lv":          int(row.get("lv", 0)),
            "moe":         moe_display,
            "partisan":    is_partisan(str(row.get("source", ""))),
            "pred_mae":    round(mae, 4),
            "sampling_sd": round(ssd, 4),
            "total_sd":    round((mae**2 + ssd**2)**0.5, 4),
            "weight":      round(float(w), 4),
            "rel_weight":  0.0,
        })

    total_w = sum(d["weight"] for d in details)
    for d in details:
        d["rel_weight"] = round(d["weight"] / total_w, 4) if total_w > 0 else 0.0

    return float(blended_env), blended_sd, details


# ── Top-level function ─────────────────────────────────────────────────────────

def run_polling_model(verbose: bool = True) -> dict:
    import json

    hist_path = DATA_DIR / "historical_polls.csv"
    if not hist_path.exists():
        if verbose:
            print("  historical_polls.csv not found — skipping polling model")
        return {}
    df_hist = pd.read_csv(hist_path)
    coeffs  = fit_mae_regression(df_hist)

    if verbose:
        print(f"  MAE regression fitted on {len(df_hist)} historical polls")
        print(f"  Coefficients: intercept={coeffs[0]:.4f} RV={coeffs[1]:.4f} "
              f"LV={coeffs[2]:.4f} days_out={coeffs[3]:.6f}")

    curr_path = DATA_DIR / "current_polls.csv"
    if not curr_path.exists():
        if verbose:
            print("  current_polls.csv not found — no polls to blend")
        df_curr = pd.DataFrame()
    else:
        df_curr = pd.read_csv(curr_path)
        if verbose:
            print(f"  Current polls loaded: {len(df_curr)} rows")

    params_path = DATA_DIR / "forecast_params.json"
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
        model_env = params["state_environment"]["predicted_state_env"]
        model_sd  = params["state_environment"]["state_env_sd"]
    else:
        model_env = 0.598
        model_sd  = 0.063

    blended_env, blended_sd, details = blend_with_polls(
        model_env, model_sd, df_curr, coeffs
    )

    result = {
        "mae_coefficients": {
            "intercept": round(float(coeffs[0]), 8),
            "rv":        round(float(coeffs[1]), 8),
            "lv":        round(float(coeffs[2]), 8),
            "days_out":  round(float(coeffs[3]), 8),
        },
        "model_env":    round(model_env, 6),
        "model_sd":     round(model_sd,  6),
        "blended_env":  round(blended_env, 6),
        "blended_sd":   round(blended_sd,  6),
        "n_polls":      len(details),
        "poll_details": details,
    }

    if verbose:
        print(f"  Model env:   {model_env:.4f}")
        if details:
            print(f"  Poll mean:   {sum(d['dem'] for d in details)/len(details):.4f}")
        print(f"  Blended env: {blended_env:.4f} (SD={blended_sd:.4f})")
        for d in details:
            partisan_flag = " [partisan]" if d["partisan"] else ""
            moe_str = f" MoE={d['moe']:.3f}" if d["moe"] else ""
            print(f"    {d['source'][:40]:40s} dem={d['dem']:.3f} "
                  f"days={d['days_out']:.0f} MAE={d['pred_mae']:.4f}"
                  f"{moe_str} w={d['rel_weight']:.3f}{partisan_flag}")

    return result


if __name__ == "__main__":
    result = run_polling_model(verbose=True)
    print(f"\nBlended environment: {result['blended_env']:.4f} "
          f"(SD={result['blended_sd']:.4f})")
