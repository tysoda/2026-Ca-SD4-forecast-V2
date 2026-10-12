"""Per-county BG-derived leans for SD4.

Computes block-group lean (d2p minus state d2p, mean over PRS/GOV/USS) per election,
projects to 2026 by linear trend or weighted mean, aggregates to county level weighted
by projected BG registration, and returns walk-forward lean SDs per county.

State 2-party shares are taken from district_series_sd4.csv (built by panel_sd4.py).

The output dict has one entry per county name (matching COUNTY_CENTROIDS):
  {
    "Alpine": {
      "lean_avg": float,   # 3-election weighted mean of BG leans, reg-weighted to county
      "lean_lin": float,   # linear trend projected to 2026
      "lean_sd":  float,   # walk-forward lean SD (generals from 2016 on)
    },
    ...
  }

Returns None if the BG panel is not present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

CONTESTS = ("PRS", "GOV", "USS")

FIPS_TO_COUNTY = {
    "06003": "Alpine",
    "06005": "Amador",
    "06009": "Calaveras",
    "06017": "El Dorado",
    "06027": "Inyo",
    "06039": "Madera",
    "06043": "Mariposa",
    "06047": "Merced",
    "06051": "Mono",
    "06057": "Nevada",
    "06061": "Placer",
    "06099": "Stanislaus",
    "06109": "Tuolumne",
}


def order_key(e: str) -> int:
    return int(e[1:]) * 10 + (e[0] == "g")


def _trend(x: np.ndarray, y: np.ndarray, at: float) -> float:
    if len(x) < 2:
        return float(np.mean(y))
    b = np.polyfit(x, y, 1)
    return float(np.polyval(b, at))


def compute_bg_county_leans(target_year: float = 2026.0) -> dict[str, dict] | None:
    """Return per-county lean dicts derived from the BG panel.

    Each county dict has keys lean_avg, lean_lin, lean_sd.
    Returns None if required files are missing.
    """
    panel_path  = PROC / "bg_panel_sd4.csv"
    series_path = PROC / "district_series_sd4.csv"
    if not panel_path.exists() or not series_path.exists():
        return None

    panel  = pd.read_csv(panel_path)
    series = pd.read_csv(series_path)

    # Zero-pad BG IDs (stored as integers in the CSV)
    panel["BG"] = panel["BG"].astype(str).str.zfill(12)
    panel["county_fips"] = panel["BG"].str[:5]
    panel["county"] = panel["county_fips"].map(FIPS_TO_COUNTY)

    # Drop BGs we can't map (guard against edge cases)
    panel = panel[panel["county"].notna()].copy()

    # Build state 2-party share per election/contest from district_series_sd4.csv
    st_map: dict[tuple[str, str], float] = {}
    for _, row in series.iterrows():
        e = row["election"]
        for c in CONTESTS:
            col = f"{c}_state_d2p"
            if col in row and pd.notna(row[col]) and 0 < row[col] < 1:
                st_map[(e, c)] = float(row[col])

    # Only use general elections for lean projection
    elections = sorted(panel["election"].unique(), key=order_key)
    generals  = [e for e in elections if e[0] == "g"]

    # ── Per-BG lean per general election ─────────────────────────────────────
    bg_leans_rows = []
    for e in generals:
        ep = panel[panel["election"] == e].copy()
        parts = []
        for c in CONTESTS:
            key = (e, c)
            if key not in st_map:
                continue
            s2p = st_map[key]
            d   = ep[f"{c}_dem"]
            r   = ep[f"{c}_rep"]
            tot = d + r
            mask = tot > 0
            bg_d2p = pd.Series(
                np.where(mask, d / tot.where(mask, 1), np.nan), index=ep.index
            )
            parts.append(bg_d2p - s2p)
        if not parts:
            continue
        lean = pd.concat(parts, axis=1).mean(axis=1)
        yr   = 2000 + int(e[1:])
        row  = ep[["BG", "county", "reg_sov", "ballots"]].copy()
        row["lean"]     = lean.values
        row["year"]     = yr
        row["election"] = e
        bg_leans_rows.append(row)

    if not bg_leans_rows:
        return None

    bg_leans = pd.concat(bg_leans_rows, ignore_index=True)
    bg_leans = bg_leans[bg_leans["lean"].notna()]

    # ── County-level lean per general (registration-weighted BG leans) ────────
    county_series: dict[str, list[tuple[int, float]]] = {cn: [] for cn in FIPS_TO_COUNTY.values()}

    for e in generals:
        yr = 2000 + int(e[1:])
        for cn in FIPS_TO_COUNTY.values():
            c_e = bg_leans[(bg_leans["election"] == e) & (bg_leans["county"] == cn)]
            if c_e.empty:
                continue
            w = c_e["reg_sov"].clip(lower=0)
            if w.sum() <= 0:
                continue
            county_lean = float((c_e["lean"] * w).sum() / w.sum())
            county_series[cn].append((yr, county_lean))

    # ── Walk-forward lean SD per county (generals from 2016 on) ──────────────
    county_wf_errs: dict[str, list[float]] = {cn: [] for cn in FIPS_TO_COUNTY.values()}

    for cn, pts in county_series.items():
        pts_sorted = sorted(pts, key=lambda t: t[0])
        for i in range(len(pts_sorted)):
            yr, actual = pts_sorted[i]
            if yr < 2016:
                continue
            hist = pts_sorted[:i]
            if len(hist) < 2:
                continue
            hx, hy = zip(*hist)
            pred = _trend(np.array(hx, dtype=float), np.array(hy), float(yr))
            county_wf_errs[cn].append(actual - pred)

    # ── Project to target_year per county ─────────────────────────────────────
    result: dict[str, dict] = {}

    for cn in FIPS_TO_COUNTY.values():
        pts = county_series.get(cn, [])
        if not pts:
            result[cn] = {"lean_avg": np.nan, "lean_lin": np.nan, "lean_sd": 0.05}
            continue

        pts_sorted = sorted(pts, key=lambda t: t[0])

        # Weighted mean: mean of last 3 generals
        recent_vals = [v for _, v in pts_sorted[-3:]]
        lean_avg    = float(np.mean(recent_vals))

        # Linear trend to target_year
        hx, hy = zip(*pts_sorted)
        lean_lin = _trend(np.array(hx, dtype=float), np.array(hy), target_year)

        # Walk-forward SD (fallback to 5pp for counties with too few elections)
        errs    = county_wf_errs.get(cn, [])
        lean_sd = float(np.std(errs, ddof=1)) if len(errs) >= 2 else 0.05

        result[cn] = {"lean_avg": lean_avg, "lean_lin": lean_lin, "lean_sd": lean_sd}

    return result


# Precompute at import time (cheap, results cached per process)
_COUNTY_BG_LEANS: dict[str, dict] | None = None


def get_county_bg_leans() -> dict[str, dict] | None:
    global _COUNTY_BG_LEANS
    if _COUNTY_BG_LEANS is None:
        _COUNTY_BG_LEANS = compute_bg_county_leans()
    return _COUNTY_BG_LEANS


if __name__ == "__main__":
    leans = compute_bg_county_leans()
    if leans is None:
        print("BG panel or district series not found.")
    else:
        print(f"{'County':12s}  {'lean_avg':>9s}  {'lean_lin':>9s}  {'lean_sd':>9s}  {'n_wf':>5s}")
        print("-" * 55)
        for cn, d in sorted(leans.items()):
            print(f"{cn:12s}  {d['lean_avg']:+9.4f}  {d['lean_lin']:+9.4f}  {d['lean_sd']:9.4f}")
