"""Walk-forward validation for SD4 block-group lean and turnout models.

Compares walk-forward errors against the existing county model.
Run after panel_sd4.py has produced bg_panel_sd4.csv and district_series_sd4.csv.

    python forecast/validate_sd4.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
CONTESTS = ("PRS", "GOV", "USS")


def order_key(e: str) -> int:
    return int(e[1:]) * 10 + (e[0] == "g")


def _ols(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    if w is not None:
        sw = np.sqrt(w)
        X, y = X * sw[:, None], y * sw
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _trend(x, y, at):
    if len(x) < 2:
        return float(np.mean(y))
    b = np.polyfit(x, y, 1)
    return float(np.polyval(b, at))


# ── lean validation ────────────────────────────────────────────────────────────
def lean_validation(series: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Walk-forward: predict district lean for each election from 2016 on."""
    s = series[series["lean"].notna()].copy()
    s["ord"] = s["election"].map(order_key)
    s = s.sort_values("ord").reset_index(drop=True)
    s["year_f"] = s["year"] + (s["general"] * 0.5)  # fractional year for trend

    results = {}
    for kind in ("g", "p"):
        sk = s[s["election"].str[0] == kind]
        rows = []
        for m in ("weighted", "linear"):
            errs = []
            for i, row in sk.iterrows():
                hist = sk[sk["year_f"] < row["year_f"]]
                if row["year"] < 2016 or len(hist) < 2:
                    continue
                if m == "weighted":
                    pred = float(hist["lean"].tail(3).mean())
                elif m == "linear":
                    pred = _trend(hist["year_f"].values, hist["lean"].values, row["year_f"])
                errs.append(row["lean"] - pred)
            rows.append((m, errs))
        results[kind] = {m: {"errors": [round(e, 4) for e in errs],
                              "mae": float(np.mean(np.abs(errs))) if errs else np.nan,
                              "sd": float(np.std(errs, ddof=1)) if len(errs) >= 2 else np.nan}
                         for m, errs in rows}
    return results


# ── turnout validation ─────────────────────────────────────────────────────────
def turnout_validation(panel: pd.DataFrame, series: pd.DataFrame,
                       min_reg: float = 50, wf_start: str = "p16") -> dict:
    """Walk-forward block-group turnout OLS: BG fixed effects + presidential + general + prev."""
    p = panel[["BG", "election", "reg_sov", "ballots"]].copy()
    keep = p.groupby("BG")["reg_sov"].min() >= min_reg
    p = p[p["BG"].isin(keep[keep].index)]
    p["turnout"] = (p["ballots"] / p["reg_sov"]).clip(0, 1)
    p = p.merge(series[["election", "presidential", "general"]], on="election")
    p["ord"] = p["election"].map(order_key)
    p = p.sort_values(["BG", "ord"])
    p["prev"] = p.groupby("BG")["turnout"].shift(1)
    fit_rows = p[p["prev"].notna()]

    bgs = sorted(p["BG"].unique())

    def design(d):
        dummies = pd.get_dummies(
            pd.Categorical(d["BG"], categories=bgs), drop_first=True, dtype=float
        ).values
        return np.column_stack([np.ones(len(d)), d[["presidential", "general", "prev"]].values, dummies])

    # Walk-forward from wf_start
    wf_ord = order_key(wf_start)
    elections_ordered = sorted(p["election"].unique(), key=order_key)
    errs, rows = [], []
    for e in elections_ordered:
        if order_key(e) < wf_ord:
            continue
        tr = fit_rows[fit_rows["ord"] < order_key(e)]
        te = p[(p["election"] == e) & p["prev"].notna()]
        if len(tr) < len(bgs) + 5 or len(te) == 0:
            continue
        c = _ols(design(tr), tr["turnout"].values, tr["reg_sov"].values)
        pred = np.clip(design(te) @ c, 0.02, 0.98)
        dp = float((pred * te["reg_sov"]).sum() / te["reg_sov"].sum())
        da = float(te["ballots"].sum() / te["reg_sov"].sum())
        errs.append(da - dp)
        rows.append({"election": e, "actual": round(da, 4), "predicted": round(dp, 4),
                     "error": round(da - dp, 4)})

    return {
        "district_sd": float(np.sqrt(np.mean(np.square(errs)))) if errs else np.nan,
        "district_mae": float(np.mean(np.abs(errs))) if errs else np.nan,
        "n_errors": len(errs),
        "walk_forward": rows,
    }


# ── main ───────────────────────────────────────────────────────────────────────
def run():
    panel = pd.read_csv(PROC / "bg_panel_sd4.csv")
    series = pd.read_csv(PROC / "district_series_sd4.csv")

    print("=" * 65)
    print("SD4 BLOCK-GROUP MODEL WALK-FORWARD VALIDATION")
    print("=" * 65)

    # ── lean ──────────────────────────────────────────────────────────────────
    lean_val = lean_validation(series, panel)
    print("\n── LEAN (district two-party share vs state) ──────────────────")
    for kind, label in (("g", "Generals"), ("p", "Primaries")):
        print(f"\n  {label}:")
        for m, res in lean_val[kind].items():
            print(f"    {m:10s}  MAE={res['mae']:.4f}  SD={res['sd']:.4f}"
                  f"  n={len(res['errors'])}")
            for i, e in enumerate(res["errors"]):
                # recover election from series
                sk = series[series["election"].str[0] == kind].sort_values("year")
                elections = sk[sk["year"] >= 2016]["election"].tolist()
                elabel = elections[i] if i < len(elections) else f"#{i}"
                print(f"             {elabel:6s} err={e:+.4f}")

    # ── turnout ───────────────────────────────────────────────────────────────
    print("\n── TURNOUT (district aggregate) ──────────────────────────────")
    tv = turnout_validation(panel, series)
    print(f"\n  Block-group model:")
    print(f"    district RMSE = {tv['district_sd']:.4f}  MAE = {tv['district_mae']:.4f}"
          f"  n={tv['n_errors']}")
    for r in tv["walk_forward"]:
        print(f"    {r['election']:6s}  actual={r['actual']:.4f}  pred={r['predicted']:.4f}"
              f"  err={r['error']:+.4f}")

    # ── county model comparison ────────────────────────────────────────────────
    import json
    params_path = ROOT / "forecast_params.json"
    if params_path.exists():
        p_json = json.load(open(params_path))
        sds = p_json.get("turnout_sds", {})
        print("\n── COUNTY MODEL (existing, for comparison) ───────────────────")
        print(f"\n  County fixed-effects model:")
        print(f"    district RMSE = {sds.get('district_turnout_sd', 'n/a'):.4f}"
              f"  n={sds.get('n_district_errors', '?')}")
        wfp = [r for r in sds.get("walk_fwd_predictions", []) if r["county"] == "Entire District"]
        for r in wfp:
            print(f"    {r['year']}/{r['month']:>3}  actual={r['actual']:.4f}"
                  f"  pred={r['predicted']:.4f}  err={r['error']:+.4f}")

        print("\n── SUMMARY ───────────────────────────────────────────────────")
        bg_rmse = tv["district_sd"]
        co_rmse = sds.get("district_turnout_sd", np.nan)
        improvement = (co_rmse - bg_rmse) / co_rmse * 100 if co_rmse and not np.isnan(bg_rmse) else np.nan
        print(f"\n  Turnout district RMSE:")
        print(f"    County model:       {co_rmse:.4f}")
        print(f"    Block-group model:  {bg_rmse:.4f}")
        if not np.isnan(improvement):
            print(f"    Improvement:        {improvement:+.1f}%")
        print(f"\n  Lean SD (generals, weighted):  {lean_val['g']['weighted']['sd']:.4f}")
        print(f"  Lean SD (generals, linear):    {lean_val['g']['linear']['sd']:.4f}")


if __name__ == "__main__":
    run()
