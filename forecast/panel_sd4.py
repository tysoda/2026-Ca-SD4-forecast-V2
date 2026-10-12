"""Block-group panel for SD4: every election's results and registration on fixed 2020 census block groups.

    python -m forecast.panel_sd4            (from the repo root)
    python forecast/panel_sd4.py            (from the repo root)

Source data: data/precinct/ on the precinct-data git branch.
Run `make fetch-precinct-data` (or the git commands below) to extract those files first:

    mkdir -p data/raw/sd4
    git show origin/precinct-data:data/precinct/<file> > data/raw/sd4/<file>

Or pass --data-dir to point at the directory where the CSVs live.

Outputs (data/processed/):
  bg_panel_sd4.csv        block group x election: registered, ballots, party reg, contest votes
  district_series_sd4.csv election totals for SD4, with state totals and lean vs state
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

ELECTIONS = ["p12", "g12", "p14", "g14", "p16", "g16", "p18", "g18",
             "p20", "g20", "p22", "g22", "p24", "g24"]
CONTESTS = ("PRS", "GOV", "USS")
REG_COLS = ["TOTREG_R", "DEM", "REP", "DCL", "AIP", "HISPDEM", "HISPREP", "HISPDCL", "HISPOTH"]


def is_2020_blocks(e: str) -> bool:
    return int(e[1:]) >= 22


def order_key(e: str) -> int:
    return int(e[1:]) * 10 + (e[0] == "g")


# ── geography ──────────────────────────────────────────────────────────────────
def block20_share(raw: Path) -> pd.Series:
    """Share of each 2020 block's p24 registrants that are in SD4 (SDDIST==4 in p24 sov)."""
    m = pd.read_csv(raw / "sd4c_p24_blk.csv", dtype=str)
    m["BLKREG"] = pd.to_numeric(m["BLKREG"])

    # p22+ use correct SDDIST==4; use p24 sov to identify in-district precincts
    sov = pd.read_csv(raw / "sd4c_p24_sov.csv", dtype=str)
    in_sd4 = set(sov.loc[sov["SDDIST"] == "4", "SRPREC_KEY"])

    m["inside"] = m["SRPREC_KEY"].isin(in_sd4)
    g = m.groupby("BLOCK_KEY")
    reg = g["BLKREG"].sum()
    share = (m["BLKREG"] * m["inside"]).groupby(m["BLOCK_KEY"]).sum() / reg
    share = share.where(reg > 0, g["inside"].mean())        # unpopulated blocks: precinct membership
    return share.rename("share20")


def block10_to_20(raw: Path, share20: pd.Series) -> pd.DataFrame:
    """2010 block -> 2020 block pieces with weights summing to 1 per 2010 block."""
    m = pd.read_csv(raw / "sd4c_p24_blk.csv", dtype=str)
    reg20 = pd.to_numeric(m["BLKREG"]).groupby(m["BLOCK_KEY"]).sum().rename("reg20")

    r = pd.read_csv(raw / "sd4c_tab2010_tab2020.csv", dtype={"BLOCK10": str, "BLOCK20": str})
    r = r.join(reg20, on="BLOCK20")
    r["reg20"] = r["reg20"].fillna(0.0)

    frac = (r["AREALAND_INT"] / r["AREALAND_2020"].where(r["AREALAND_2020"] > 0)).fillna(1.0)
    r["w_reg"] = r["reg20"] * frac
    r["w_area"] = r["AREALAND_INT"] + r["AREAWATER_INT"]

    tot_reg = r.groupby("BLOCK10")["w_reg"].transform("sum")
    tot_area = r.groupby("BLOCK10")["w_area"].transform("sum")
    r["w"] = np.where(tot_reg > 0,
                      r["w_reg"] / tot_reg.where(tot_reg > 0, 1),
                      r["w_area"] / tot_area.where(tot_area > 0, 1))
    return r[["BLOCK10", "BLOCK20", "w"]]


def precinct_bg_weights(e: str, raw: Path, share20: pd.Series, x10: pd.DataFrame) -> pd.DataFrame:
    """SRPREC_KEY, BG, w: share of the precinct's registrants in that block group and in SD4."""
    m = pd.read_csv(raw / f"sd4c_{e}_blk.csv", dtype=str)
    m["BLKREG"] = pd.to_numeric(m["BLKREG"])

    tot = m.groupby("SRPREC_KEY")["BLKREG"].transform("sum")
    n = m.groupby("SRPREC_KEY")["BLKREG"].transform("size")
    m["p"] = np.where(tot > 0, m["BLKREG"] / tot.where(tot > 0, 1), 1.0 / n)

    if is_2020_blocks(e):
        b = m[["SRPREC_KEY", "BLOCK_KEY", "p"]].rename(columns={"BLOCK_KEY": "BLOCK20"})
    else:
        b = m[["SRPREC_KEY", "BLOCK_KEY", "p"]].merge(x10, left_on="BLOCK_KEY", right_on="BLOCK10")
        b["p"] = b["p"] * b["w"]

    b = b.join(share20, on="BLOCK20")
    b["w"] = b["p"] * b["share20"].fillna(0.0)
    b["BG"] = b["BLOCK20"].str[:12]
    w = b.groupby(["SRPREC_KEY", "BG"], as_index=False)["w"].sum()
    return w[w["w"] > 1e-9]


# ── per-election values ────────────────────────────────────────────────────────
def _results(e: str, raw: Path) -> pd.DataFrame:
    """Precinct results; for p12–g20 uses all 13-county precincts (old lines)."""
    d = pd.read_csv(raw / f"sd4c_{e}_sov.csv", dtype={"SRPREC_KEY": str}, low_memory=False)

    # p22+: already scoped to correct SDDIST==4; p12-g20: all 13 counties (weights handle the fractions)
    out = pd.DataFrame({"SRPREC_KEY": d["SRPREC_KEY"],
                        "reg_sov": pd.to_numeric(d["TOTREG"], errors="coerce").fillna(0),
                        "ballots": pd.to_numeric(d["TOTVOTE"], errors="coerce").fillna(0)})

    for c in CONTESTS:
        dem_cols = [x for x in d.columns if x.startswith(c + "DEM")]
        rep_cols = [x for x in d.columns if x.startswith(c + "REP")]
        # Replace '***' masked values with NaN then 0 for summing
        def safe_sum(cols):
            if not cols:
                return pd.Series(0.0, index=d.index)
            return d[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        out[f"{c}_dem"] = safe_sum(dem_cols)
        out[f"{c}_rep"] = safe_sum(rep_cols)

    # Assembly votes
    for party in ("DEM", "REP", "AIP"):
        cols = [x for x in d.columns if x.startswith("ASS" + party)]
        def safe_sum_p(cols):
            if not cols:
                return pd.Series(0.0, index=d.index)
            return d[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        out[f"ASS_{party.lower()}"] = safe_sum_p(cols)

    return out


def _registration(e: str, raw: Path) -> pd.DataFrame:
    d = pd.read_csv(raw / f"sd4c_{e}_reg.csv", dtype={"SRPREC_KEY": str}, low_memory=False)
    out = d[["SRPREC_KEY"] + [c for c in REG_COLS if c in d.columns]].copy()
    hisp_cols = [c for c in ["HISPDEM", "HISPREP", "HISPDCL", "HISPOTH"] if c in out.columns]
    if hisp_cols:
        out["HISP"] = out[hisp_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        out = out.drop(columns=hisp_cols)
    # Numeric-ify remaining reg columns
    for c in out.columns:
        if c != "SRPREC_KEY":
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out


# ── main build ─────────────────────────────────────────────────────────────────
def build(raw: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw is None:
        raw = ROOT / "data" / "raw" / "sd4"

    share20 = block20_share(raw)
    x10 = block10_to_20(raw, share20)
    state = pd.read_csv(raw / "state_contest_totals.csv")

    panels, series = [], []
    for e in ELECTIONS:
        print(f"  {e}...", end=" ", flush=True)
        w = precinct_bg_weights(e, raw, share20, x10)
        vals = _results(e, raw).merge(_registration(e, raw), on="SRPREC_KEY", how="outer").fillna(0.0)
        num = [c for c in vals.columns if c != "SRPREC_KEY"]
        bg = (w.merge(vals, on="SRPREC_KEY")
               .assign(**{c: lambda d, c=c: d[c] * d["w"] for c in num})
               .groupby("BG")[num].sum().reset_index())
        bg.insert(1, "election", e)
        panels.append(bg)

        row = {"election": e, "year": 2000 + int(e[1:]), "general": int(e[0] == "g"),
               "presidential": int(int(e[1:]) % 4 == 0)}
        tot = bg[num].sum()
        row.update(registered=tot["reg_sov"], ballots=tot["ballots"],
                   turnout=tot["ballots"] / tot["reg_sov"] if tot["reg_sov"] > 0 else np.nan,
                   reg_dem=tot.get("DEM", 0) / tot.get("TOTREG_R", 1),
                   reg_rep=tot.get("REP", 0) / tot.get("TOTREG_R", 1),
                   reg_npp=tot.get("DCL", 0) / tot.get("TOTREG_R", 1),
                   reg_hisp=tot.get("HISP", 0) / tot.get("TOTREG_R", 1) if "HISP" in tot else np.nan)

        leans = []
        for c in CONTESTS:
            st = state[(state["election"] == e) & (state["contest"] == c)]
            if st.empty or (st["dem"].iloc[0] == 0 and st["rep"].iloc[0] == 0):
                continue
            d_dem = tot.get(f"{c}_dem", 0)
            d_rep = tot.get(f"{c}_rep", 0)
            if d_dem + d_rep == 0:
                continue
            d2p = d_dem / (d_dem + d_rep)
            s2p = st["dem"].iloc[0] / (st["dem"].iloc[0] + st["rep"].iloc[0])
            row[f"{c}_d2p"] = d2p
            row[f"{c}_state_d2p"] = s2p
            leans.append(d2p - s2p)

        row["lean"] = float(np.mean(leans)) if leans else np.nan
        row["n_contests"] = len(leans)
        series.append(row)
        print(f"BGs={bg.BG.nunique()}, reg={tot['reg_sov']:,.0f}, ballots={tot['ballots']:,.0f}",
              flush=True)

    panel = pd.concat(panels, ignore_index=True)
    series_df = pd.DataFrame(series)
    PROC.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PROC / "bg_panel_sd4.csv", index=False)
    series_df.to_csv(PROC / "district_series_sd4.csv", index=False)
    print(f"\nbg_panel_sd4.csv: {panel.BG.nunique()} block groups x {panel.election.nunique()} elections")
    return panel, series_df


if __name__ == "__main__":
    data_dir = None
    for arg in sys.argv[1:]:
        if arg.startswith("--data-dir="):
            data_dir = Path(arg.split("=", 1)[1])
    print("Building SD4 block-group panel...")
    panel, series = build(raw=data_dir)
    cols = ["election", "registered", "ballots", "turnout", "reg_dem", "reg_rep", "lean", "n_contests"]
    print(series[[c for c in cols if c in series.columns]].to_string(
        index=False, float_format=lambda x: f"{x:,.3f}" if abs(x) < 10 else f"{x:,.0f}"))
