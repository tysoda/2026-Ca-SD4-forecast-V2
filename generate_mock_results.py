"""
generate_mock_results.py
Generates realistic election-night reporting snapshots for SD4 mock scenarios.

Usage:
    python3 generate_mock_results.py --scenario "narrow_win"
    python3 generate_mock_results.py --scenario "comfortable_win"
    python3 generate_mock_results.py --scenario "narrow_loss"
    python3 generate_mock_results.py --scenario all   (generates all three)

Output: writes CSV files into mock_results/<scenario>/<wave>.csv
These can be loaded directly by the Election Night tab.

═══════════════════════════════════════════════════════════════════════════════
ASSUMPTIONS AND DATA SOURCES
═══════════════════════════════════════════════════════════════════════════════

COUNTY PARAMETERS (COUNTIES dict)
──────────────────────────────────
  reg       Registration figures drawn from FALLBACK_COUNTIES in election_forecast.py,
            which sources from the CA Secretary of State voter registration data.
            These are 2026 registration estimates; verify against SOS prior to election.

  turnout   Forecast turnout rates from FALLBACK_COUNTIES in election_forecast.py,
            derived from the model's panel OLS regression on historical turnout data
            (2012–2026). These are the same values the Monte Carlo simulation uses,
            ensuring internal consistency between mock scenarios and model output.

  lean_lin  Linear lean coefficients from FALLBACK_COUNTIES in election_forecast.py.
            Each county's expected Dem vote share deviation from the statewide environment,
            fitted by regression. Negative = Republican-leaning county.

REPORTING SPEEDS (REPORTING_SPEED dict)
────────────────────────────────────────
  Values represent the fraction of final county votes reported by each wave:
    Wave 0 = 8:00 PM election night (first drop — pre-processed mail ballots)
    Wave 1 = 9:00 PM
    Wave 2 = 11:00 PM
    Wave 3 = Next Day (effectively final, though certification takes weeks)

  DATA-DERIVED (9 counties): Calculated from the California Voter Foundation
  Close Count Transparency Project 2024 General Election data, using unprocessed
  ballot counts at each timestamp divided by total votes cast (from historical_turnout.csv).
  Source: calvoter.org/content/close-count-transparency-project#2024project
  Election: November 5, 2024 General Election.

  The CD-3 sheet covers: Alpine, El Dorado, Inyo, Mono, Nevada, Placer.
  The CD-13 sheet covers: Madera, Merced, Stanislaus.
  Note: CD-13 has no election-night snapshot (first entry is Nov 7), so election-night
  figures for Madera, Merced, Stanislaus are extrapolated backward from the Nov 7 totals
  using the observed rate of change in the subsequent days.

  Key empirical findings from 2024 general:
    - Alpine (95.7%), Mono (95.1%): extremely fast — nearly all in on election night
    - El Dorado (78.1%): fast election night drop, then stalls (batch processor)
    - Placer (72.0%): similar batch pattern; big election-night drop, slow thereafter
    - Inyo (64.9%): moderate — most in by day 2 (92.3%)
    - Nevada (15.7%): OUTLIER — very slow; only 51.9% by day 7, 82.1% by day 14.
                      Likely reflects large provisional/cured ballot operation.
    - Merced (46.3% by day 2): slowest of the CV counties
    - Stanislaus (65.1% by day 2, barely moves to day 7): batch processor

  Wave fractions for the 2026 general are estimated from 2024 data with a slight
  downward adjustment (~5%) to account for potentially slower processing in a midterm
  vs presidential cycle (lower total volume can cut both ways; treat with caution).

  ESTIMATED (4 counties): Amador, Calaveras, Mariposa, Tuolumne have no direct
  tracking data. Estimates are interpolated from similar counties:
    - Amador:    Anchored to Inyo (similar small rural foothill character, ~11k reg)
    - Calaveras: Blend of Inyo + Madera (slightly larger, same foothill profile)
    - Mariposa:  Anchored to Mono (tiny county, ~12k reg, likely fast reporter)
    - Tuolumne:  Blend of Madera + Inyo (largest of the four, more mid-size rural)
  These should be updated if empirical data becomes available (e.g. from county
  registrar websites for 2022 or 2024).

MAIL BALLOT PERCENTAGES (MAIL_PCT dict)
────────────────────────────────────────
  Estimated from statewide CA trends (typically 63–76% mail depending on county
  type). Sierra/foothill counties tend toward higher mail rates; Central Valley
  agricultural counties toward lower. These are not county-specific empirical
  values — update from county registrar data if available.

MAIL DEMOCRATIC LEAN (MAIL_DEM_BOOST)
──────────────────────────────────────
  Mail voters lean approximately +4pp more Democratic than election-day voters
  in CA, based on published research on CA voting patterns (roughly +3 to +5pp
  range observed across recent elections). This affects the vote share mix in
  early waves, which are mail-heavy.

SCENARIOS (SCENARIOS dict)
───────────────────────────
  Base state environment: 59.80% (the model's predicted Dem share of the two-party
  statewide vote, used as the anchor). At this environment the district projects to
  ~44% Dem given county lean coefficients — SD4 is a Republican-leaning stretch district.

  env_shift values are calibrated so that the district-level Dem share hits:
    narrow_win      (+8.1pp shift → env ~67.9%): district ~52.7% Dem
    comfortable_win (+11.5pp shift → env ~71.3%): district ~56.1% Dem
    narrow_loss     (+4.2pp shift → env ~64.0%): district ~48.8% Dem

  These large environment shifts reflect what would be needed for SD4 to be
  genuinely competitive — consistent with it being a stretch/reach district.
  A narrow win requires roughly a wave election environment.

  County-level noise: ±1.5pp (rng.normal(0, 0.015)) per county draw.
═══════════════════════════════════════════════════════════════════════════════
"""
import argparse, os, random, json
import numpy as np
import pandas as pd

# ── County parameters ──────────────────────────────────────────────────────────
# Source: FALLBACK_COUNTIES in election_forecast.py (reg from CA SOS, turnout and
# lean_lin from model regression). See ASSUMPTIONS section above for full notes.
COUNTIES = {
    "Alpine":     {"reg": 944,    "turnout": 0.7074, "lean_lin":  0.0995},
    "Amador":     {"reg": 27416,  "turnout": 0.7110, "lean_lin": -0.2444},
    "Calaveras":  {"reg": 33312,  "turnout": 0.6702, "lean_lin": -0.2538},
    "El Dorado":  {"reg": 142947, "turnout": 0.6630, "lean_lin": -0.1531},
    "Inyo":       {"reg": 11037,  "turnout": 0.6890, "lean_lin": -0.0973},
    "Madera":     {"reg": 34903,  "turnout": 0.5690, "lean_lin": -0.2325},
    "Mariposa":   {"reg": 11862,  "turnout": 0.6941, "lean_lin": -0.2038},
    "Merced":     {"reg": 12842,  "turnout": 0.4989, "lean_lin": -0.2239},
    "Mono":       {"reg": 8286,   "turnout": 0.6654, "lean_lin": -0.0033},
    "Nevada":     {"reg": 12680,  "turnout": 0.6959, "lean_lin":  0.1741},
    "Placer":     {"reg": 8640,   "turnout": 0.6520, "lean_lin":  0.1462},
    "Stanislaus": {"reg": 304987, "turnout": 0.5178, "lean_lin": -0.1540},
    "Tuolumne":   {"reg": 36167,  "turnout": 0.6736, "lean_lin": -0.2034},
}

# Precinct counts (used for precincts_reporting field)
PRECINCTS = {
    "Alpine": 6, "Amador": 68, "Calaveras": 81, "El Dorado": 155,
    "Inyo": 42, "Madera": 113, "Mariposa": 34, "Merced": 128,
    "Mono": 27, "Nevada": 70, "Placer": 233, "Stanislaus": 230,
    "Tuolumne": 69,
}

# ── Reporting speeds ───────────────────────────────────────────────────────────
# Fraction of final county votes reported by each wave:
#   Wave 0 = 8:00 PM (first drop)
#   Wave 1 = 9:00 PM
#   Wave 2 = 11:00 PM
#   Wave 3 = Next Day (effectively final)
#
# DATA-DERIVED values (marked [D]) come from CalVoter Foundation Close Count
# Transparency Project 2024 General Election data. See ASSUMPTIONS section.
# ESTIMATED values (marked [E]) are interpolated — see ASSUMPTIONS for method.
#
# The 8pm/9pm/11pm fractions are inferred from the election-night trajectory;
# the 2024 data gives Nov 5 ~midnight and Nov 7 snapshots as anchors.
REPORTING_SPEED = {
    # ── Fast reporters: tiny rural/foothill counties ───────────────────────────
    # Alpine [D]: 95.7% by midnight election night in 2024. Tiny county (~750 votes).
    "Alpine":    [0.80, 0.92, 0.96, 0.99],
    # Mono [D]: 95.1% by midnight election night in 2024.
    "Mono":      [0.78, 0.91, 0.95, 0.99],
    # Mariposa [E]: Estimated from Mono (similar tiny rural character, ~9.6k votes cast).
    "Mariposa":  [0.70, 0.86, 0.93, 0.98],
    # Inyo [D]: 64.9% by midnight; 92.3% by day 2. Moderate pace.
    "Inyo":      [0.55, 0.75, 0.88, 0.97],

    # ── Medium reporters: mid-size rural foothill counties ─────────────────────
    # Amador [E]: Estimated from Inyo (similar small rural foothill, ~16k votes cast).
    "Amador":    [0.52, 0.72, 0.86, 0.97],
    # Calaveras [E]: Blend of Inyo + Madera (slightly larger, ~22k votes cast).
    "Calaveras": [0.45, 0.68, 0.82, 0.96],
    # Tuolumne [E]: Blend of Madera + Inyo (largest of estimated four, ~29k votes).
    "Tuolumne":  [0.40, 0.62, 0.78, 0.95],
    # Nevada [D]: OUTLIER — only 15.7% by midnight, 51.9% by day 7. Very slow.
    #             Likely large provisional/cured ballot operation. Treated as slow.
    "Nevada":    [0.12, 0.20, 0.35, 0.75],

    # ── Slow reporters: larger / Central Valley counties ───────────────────────
    # El Dorado [D]: 78.1% by midnight, then stalls — batch processor pattern.
    #               98.5% by day 2, barely moves after. Wave 3 = ~99%.
    "El Dorado": [0.65, 0.78, 0.90, 0.98],
    # Placer [D]: 72.0% by midnight, stalls to day 7 (87%), then ~96% by day 14.
    "Placer":    [0.60, 0.72, 0.82, 0.95],
    # Madera [D]: No election-night snapshot. 67.6% by day 2, 80.7% by day 7.
    #             Election-night fraction extrapolated as ~40% (pre-processed mail drop).
    "Madera":    [0.38, 0.55, 0.72, 0.92],
    # Merced [D]: No election-night snapshot. 46.3% by day 2 — slowest CV county.
    #             Election-night fraction extrapolated as ~25%.
    "Merced":    [0.22, 0.38, 0.55, 0.88],
    # Stanislaus [D]: No election-night snapshot. 65.1% by day 2, barely moves to
    #                 day 7 (65.1%), then jumps to 72.9% by day 14 — pronounced batch
    #                 processor. Election-night extrapolated as ~35%.
    "Stanislaus":[0.32, 0.48, 0.62, 0.88],
}

# ── Mail ballot percentages ────────────────────────────────────────────────────
# Estimated from statewide CA trends. Not county-specific empirical values.
# Sierra/foothill counties tend higher; Central Valley counties lower.
# Source: general CA election pattern knowledge; update from registrar data if available.
MAIL_DEM_BOOST = 0.04   # mail votes ~4pp more Dem than county average (CA research)
MAIL_PCT = {
    "Alpine": 0.72, "Amador": 0.70, "Calaveras": 0.73, "El Dorado": 0.68,
    "Inyo": 0.71, "Madera": 0.62, "Mariposa": 0.74, "Merced": 0.60,
    "Mono": 0.75, "Nevada": 0.76, "Placer": 0.69, "Stanislaus": 0.63,
    "Tuolumne": 0.72,
}

# ── Scenarios ──────────────────────────────────────────────────────────────────
# State environment shifts relative to the model's 59.8% base.
# See ASSUMPTIONS section for calibration details.
SCENARIOS = {
    "narrow_win":      {"env_shift": +0.081, "desc": "Narrow win (~52.7% district)"},
    "comfortable_win": {"env_shift": +0.115, "desc": "Comfortable win (~56.1% district)"},
    "narrow_loss":     {"env_shift": +0.042, "desc": "Narrow loss (~48.8% district)"},
}

WAVE_LABELS = ["8:00 PM", "9:00 PM", "11:00 PM", "Next Day (Final)"]

def generate_scenario(scenario_name: str, seed: int = 42) -> dict:
    """
    Returns a dict of wave_label -> DataFrame with county results.
    Each wave reflects partial reporting with realistic mail/ED mix.
    """
    rng = np.random.default_rng(seed)
    sc = SCENARIOS[scenario_name]
    base_env = 0.5980
    true_env = base_env + sc["env_shift"]

    # Draw final true vote shares for each county
    # Add small county-specific noise (lean_sd ~1.5pp)
    final_shares = {}
    final_total_votes = {}
    for cn, d in COUNTIES.items():
        county_noise = rng.normal(0, 0.015)   # ~1.5pp county-level noise
        share = true_env + d["lean_lin"] + county_noise
        share = max(0.10, min(0.90, share))
        final_shares[cn] = share
        total_v = int(d["reg"] * d["turnout"] * rng.uniform(0.95, 1.05))
        final_total_votes[cn] = total_v

    # Compute district-level share
    total_d = sum(final_total_votes[cn] * final_shares[cn] for cn in COUNTIES)
    total_v = sum(final_total_votes.values())
    print(f"\nScenario: {scenario_name} ({sc['desc']})")
    print(f"  True env: {true_env:.3f}, District Dem share: {total_d/total_v:.3f}")

    # For each county, split votes into mail and election-day
    # Mail ballots tend to come in earlier and lean more Dem
    county_mail_votes = {}
    county_ed_votes = {}
    for cn in COUNTIES:
        mp = MAIL_PCT[cn]
        n_mail = int(final_total_votes[cn] * mp)
        n_ed   = final_total_votes[cn] - n_mail
        # Mail Dem share = county share + MAIL_DEM_BOOST (capped at 0.9)
        mail_share = min(0.90, final_shares[cn] + MAIL_DEM_BOOST)
        # ED Dem share is lower (mail_boost absorbed by ED going other way)
        # weighted average: mp*mail_share + (1-mp)*ed_share = final_share
        ed_share = (final_shares[cn] - mp * mail_share) / (1 - mp) if (1 - mp) > 0 else final_shares[cn]
        ed_share = max(0.10, min(0.90, ed_share))
        county_mail_votes[cn] = {"n": n_mail, "dem_share": mail_share}
        county_ed_votes[cn]   = {"n": n_ed,   "dem_share": ed_share}

    waves = {}
    for wave_idx, wave_label in enumerate(WAVE_LABELS):
        rows = []
        for cn in COUNTIES:
            speed = REPORTING_SPEED[cn][wave_idx]
            prec_total = PRECINCTS[cn]

            # Add wave-level noise to reporting speed (±5pp)
            speed_jitter = rng.uniform(-0.05, 0.05)
            speed = max(0.0, min(1.0, speed + speed_jitter))

            total_v = final_total_votes[cn]
            votes_in = int(total_v * speed)

            if votes_in == 0:
                rows.append({
                    "county": cn,
                    "precincts_reporting": 0,
                    "precincts_total": prec_total,
                    "dem_votes": 0, "rep_votes": 0, "other_votes": 0,
                    "mail_dem": 0, "mail_rep": 0, "mail_other": 0,
                    "source": "mock", "last_updated": wave_label,
                })
                continue

            n_mail = county_mail_votes[cn]["n"]
            mail_share = county_mail_votes[cn]["dem_share"]
            n_ed = county_ed_votes[cn]["n"]
            ed_share = county_ed_votes[cn]["dem_share"]

            # Early waves: mail-heavy (mail processed first in CA)
            # Wave fractions: by wave_idx, what share of final is mail vs ED
            mail_reporting_boost = [0.85, 0.75, 0.60, 0.50][wave_idx]
            # What fraction of the reported votes are mail?
            mail_frac_in_wave = min(1.0, MAIL_PCT[cn] + (1 - MAIL_PCT[cn]) * mail_reporting_boost * (1 - speed) / max(0.01, 1 - MAIL_PCT[cn]))
            mail_frac_in_wave = min(MAIL_PCT[cn] + 0.25, max(MAIL_PCT[cn] - 0.10, MAIL_PCT[cn]))
            # Simpler: at wave 0, mail is ~90% of ballots in; by final, back to mail_pct
            mail_pct_in_wave = [
                min(0.95, MAIL_PCT[cn] + 0.18),
                min(0.95, MAIL_PCT[cn] + 0.10),
                min(0.95, MAIL_PCT[cn] + 0.04),
                MAIL_PCT[cn],
            ][wave_idx]

            mail_in  = int(votes_in * mail_pct_in_wave)
            ed_in    = votes_in - mail_in

            # Cap at final totals
            mail_in  = min(mail_in,  n_mail)
            ed_in    = min(ed_in,    n_ed)
            votes_in = mail_in + ed_in

            if votes_in == 0:
                rows.append({
                    "county": cn, "precincts_reporting": 0, "precincts_total": prec_total,
                    "dem_votes": 0, "rep_votes": 0, "other_votes": 0,
                    "mail_dem": 0, "mail_rep": 0, "mail_other": 0,
                    "source": "mock", "last_updated": wave_label,
                })
                continue

            mail_dem  = int(mail_in * mail_share + rng.normal(0, mail_in * 0.005))
            mail_dem  = max(0, min(mail_in, mail_dem))
            mail_rep  = mail_in - mail_dem
            mail_other = 0

            ed_dem    = int(ed_in * ed_share + rng.normal(0, ed_in * 0.005))
            ed_dem    = max(0, min(ed_in, ed_dem))
            ed_rep    = ed_in - ed_dem

            dem_total   = mail_dem + ed_dem
            rep_total   = mail_rep + ed_rep
            other_total = 0

            # Precincts in: scale by speed, with small variation
            prec_in = min(prec_total, max(0, int(prec_total * speed + rng.normal(0, 0.5))))

            rows.append({
                "county": cn,
                "precincts_reporting": prec_in,
                "precincts_total": prec_total,
                "dem_votes": dem_total,
                "rep_votes": rep_total,
                "other_votes": other_total,
                "mail_dem": mail_dem,
                "mail_rep": mail_rep,
                "mail_other": 0,
                "source": "mock",
                "last_updated": wave_label,
            })

        df = pd.DataFrame(rows)
        waves[wave_label] = df

        # Print district totals for this wave
        d_in = df.dem_votes.sum(); r_in = df.rep_votes.sum()
        tv_in = d_in + r_in
        share_in = d_in / tv_in if tv_in > 0 else 0
        prec_pct = df.precincts_reporting.sum() / df.precincts_total.sum()
        print(f"  {wave_label:20s}: {prec_pct*100:.0f}% prec, Dem {share_in*100:.1f}% ({d_in:,}D / {r_in:,}R)")

    return waves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all",
                        choices=["narrow_win","comfortable_win","narrow_loss","all"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]

    for sc_name in scenarios:
        out_dir = os.path.join(os.path.dirname(__file__), "mock_results", sc_name)
        os.makedirs(out_dir, exist_ok=True)

        waves = generate_scenario(sc_name, seed=args.seed)
        for wave_label, df in waves.items():
            safe_name = wave_label.replace(":", "").replace(" ", "_").replace("(", "").replace(")", "")
            path = os.path.join(out_dir, f"{safe_name}.csv")
            df.to_csv(path, index=False)
            print(f"    → {path}")

        # Also write a manifest
        manifest = {
            "scenario": sc_name,
            "description": SCENARIOS[sc_name]["desc"],
            "waves": [
                wave_label.replace(":", "").replace(" ", "_").replace("(", "").replace(")", "")
                for wave_label in waves.keys()
            ],
            "wave_labels": list(waves.keys()),
        }
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
