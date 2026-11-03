"""
prior.py — the pre-results forecast distribution for the Election Night app.

Rebuilds the same Monte Carlo prior the forecast app's sidebar produces with its
default settings, without importing election_forecast.py:

  * county parameters from forecast_params.json (hardcoded fallback otherwise)
  * state environment = model prediction (optional manual margin override)
  * lean method Linear / Average / Block Group (Block Group default when the BG
    panel is present, as in the forecast app)
  * no turnout adjustments, no candidate adjustment, no polling blend
    (all opt-in in the forecast app and off by default)

run_simulation() is a copy of the forecast app's function so results match it
for the same inputs and seed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent

WIN_THRESHOLD = 0.50
N_DEFAULT = 10_000
SEED = 2026  # fixed so numbers don't jump between auto-refreshes or viewers

# Same fallback as election_forecast.py (used only if forecast_params.json is absent)
FALLBACK_COUNTIES = {
    "Alpine":     {"registration":917,   "turnout":0.7074,"lean_avg":0.0388,"lean_lin":0.0995,"lean_sd":0.0293,"hist_avg":0.6221},
    "Amador":     {"registration":26992, "turnout":0.7110,"lean_avg":-0.2465,"lean_lin":-0.2444,"lean_sd":0.0143,"hist_avg":0.3481},
    "Calaveras":  {"registration":32726, "turnout":0.6702,"lean_avg":-0.2404,"lean_lin":-0.2538,"lean_sd":0.0162,"hist_avg":0.3581},
    "El Dorado":  {"registration":137930,"turnout":0.6630,"lean_avg":-0.1784,"lean_lin":-0.1531,"lean_sd":0.0140,"hist_avg":0.4079},
    "Inyo":       {"registration":10926, "turnout":0.6890,"lean_avg":-0.1280,"lean_lin":-0.0973,"lean_sd":0.0122,"hist_avg":0.4570},
    "Madera":     {"registration":36556, "turnout":0.5690,"lean_avg":-0.2700,"lean_lin":-0.2325,"lean_sd":0.0182,"hist_avg":0.3155},
    "Mariposa":   {"registration":11799, "turnout":0.6941,"lean_avg":-0.2083,"lean_lin":-0.2038,"lean_sd":0.0065,"hist_avg":0.3772},
    "Merced":     {"registration":12881, "turnout":0.4989,"lean_avg":-0.2673,"lean_lin":-0.2239,"lean_sd":0.0231,"hist_avg":0.3182},
    "Mono":       {"registration":7960,  "turnout":0.6654,"lean_avg":-0.0251,"lean_lin":-0.0033,"lean_sd":0.0173,"hist_avg":0.5628},
    "Nevada":     {"registration":12605, "turnout":0.6959,"lean_avg":0.1353,"lean_lin":0.1741,"lean_sd":0.0181,"hist_avg":0.7208},
    "Placer":     {"registration":8607,  "turnout":0.6520,"lean_avg":0.1224,"lean_lin":0.1462,"lean_sd":0.0113,"hist_avg":0.7079},
    "Stanislaus": {"registration":301333,"turnout":0.5178,"lean_avg":-0.1553,"lean_lin":-0.1540,"lean_sd":0.0123,"hist_avg":0.4357},
    "Tuolumne":   {"registration":35784, "turnout":0.6736,"lean_avg":-0.2092,"lean_lin":-0.2034,"lean_sd":0.0140,"hist_avg":0.3763},
}
FALLBACK_SDS = {"district_turnout_sd": 0.0823, "county_turnout_sd": 0.0773,
                "state_env_sd": 0.0629, "state_env": 0.5980}


def load_params() -> dict:
    """County dict + SDs + model environment, as the forecast app initialises them."""
    p = REPO_DIR / "forecast_params.json"
    if p.exists():
        with open(p) as f:
            fp = json.load(f)
        counties = {
            cn: {
                "registration": int(d["registration"]),
                "turnout":      float(d["turnout"]),
                "lean_avg":     float(d["lean_avg"]),
                "lean_lin":     float(d["lean_lin"]),
                "lean_sd":      float(d["lean_sd"]),
                "hist_avg":     float(d["hist_avg"]),
            }
            for cn, d in fp["counties"].items()
        }
        # The forecast app rounds these to 2 dp of a percent in session state
        return {
            "counties": counties,
            "district_turnout_sd": round(fp["turnout_sds"]["district_turnout_sd"] * 100, 2) / 100,
            "county_turnout_sd":   round(fp["turnout_sds"]["county_turnout_sd"] * 100, 2) / 100,
            "state_env_sd":        round(fp["state_environment"]["state_env_sd"] * 100, 2) / 100,
            "state_env":           round(fp["state_environment"]["predicted_state_env"] * 100, 2) / 100,
            "source": "forecast_params.json",
        }
    return {"counties": FALLBACK_COUNTIES, **FALLBACK_SDS, "source": "fallback defaults"}


def load_bg_county_leans() -> dict | None:
    """Per-county block-group leans, or None if the BG panel isn't present."""
    try:
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))
        from forecast.bg_leans_sd4 import compute_bg_county_leans
        return compute_bg_county_leans()
    except Exception:
        return None


def bg_frozen(bg_leans: dict | None):
    if bg_leans is None:
        return None
    return tuple(
        (cn, d["lean_avg"], d["lean_lin"], d["lean_sd"])
        for cn, d in bg_leans.items()
        if np.isfinite(d["lean_avg"]) and np.isfinite(d["lean_lin"])
    )


def counties_to_frozen(d: dict):
    return tuple((n, tuple(sorted(v.items()))) for n, v in d.items())


# ── Copied from election_forecast.py ─────────────────────────────────────────
def run_simulation(n, forecast_env, lean_method_str, seed,
                   district_sd, county_sd, state_sd,
                   counties_frozen, turnout_adj_frozen,
                   bg_county_leans_frozen=None):
    rng = np.random.default_rng(seed); n = int(n)
    counties    = {name: dict(fields) for name, fields in counties_frozen}
    turnout_adj = dict(turnout_adj_frozen)
    bg_leans = ({name: {"lean_avg": la, "lean_lin": ll, "lean_sd": ls}
                 for name, la, ll, ls in bg_county_leans_frozen}
                if bg_county_leans_frozen is not None else None)

    env_error              = rng.normal(0, state_sd, n)
    sim_env                = forecast_env + env_error
    district_turnout_shift = rng.normal(0, district_sd, n)

    use_bg   = (lean_method_str == "Block Group" and bg_leans is not None)
    lean_key = "lean_avg" if lean_method_str == "Average" else "lean_lin"

    county_shares={}; county_votes={}; county_dem_votes={}; county_turnouts={}
    for name, data in counties.items():
        adj = turnout_adj.get(name, 0.0)
        turnout_sim = np.clip(
            data["turnout"] + adj + district_turnout_shift + rng.normal(0, county_sd, n), 0.01, 1.0)
        if use_bg and name in bg_leans:
            bg  = bg_leans[name]
            lean      = bg[lean_key]
            lean_sd   = bg["lean_sd"]
            lean_error = rng.normal(0, lean_sd, n)
            share = np.clip(sim_env + lean + lean_error, 0.0, 1.0)
        else:
            lean_error = rng.normal(0, data["lean_sd"], n)
            share      = np.clip(sim_env + data[lean_key] + lean_error, 0.0, 1.0)
        votes      = data["registration"] * turnout_sim
        dem_votes  = votes * share
        county_shares[name]=share; county_votes[name]=votes
        county_dem_votes[name]=dem_votes; county_turnouts[name]=turnout_sim
    total_votes     = sum(county_votes.values())
    total_dem_votes = sum(county_dem_votes.values())
    return {
        "district_share":   total_dem_votes / total_votes,
        "sim_env":          sim_env, "env_error": env_error,
        "county_shares":    county_shares, "county_votes": county_votes,
        "county_dem_votes": county_dem_votes, "county_turnouts": county_turnouts,
    }
