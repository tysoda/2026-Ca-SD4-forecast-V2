"""Cut statewide SWDB SR-precinct files and the Census block crosswalk to SD4's 13 counties.

    python scripts/cut_swdb_sd4.py <statewide csv> [...]       # SOV, registration or block-map files
    python scripts/cut_swdb_sd4.py --crosswalk tab2010_tab2020_st06_ca.txt

Writes data/precinct/sd4c_{election}_{sov|reg|blk}.csv (every precinct in the 13 counties, all columns,
values exactly as published) and data/precinct/sd4c_tab2010_tab2020.csv.
"""
import re
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "precinct"
SD4_FIPS = {  # Alpine, Amador, Calaveras, El Dorado, Inyo, Madera, Mariposa, Merced, Mono, Nevada, Placer, Stanislaus, Tuolumne
    "06003", "06005", "06009", "06017", "06027", "06039", "06043", "06047", "06051", "06057", "06061", "06099", "06109"}


def cut(path: Path) -> None:
    name = path.name
    e = re.search(r"_([pg]\d\d)_", name).group(1)
    kind = "sov" if "sov_data" in name else "reg" if "registration" in name else "blk"
    d = pd.read_csv(path, dtype=str, low_memory=False)
    fips = next(c for c in d.columns if c.upper() == "FIPS")
    d = d[d[fips].str.strip().str.zfill(5).isin(SD4_FIPS)]
    d.to_csv(OUT / f"sd4c_{e}_{kind}.csv", index=False)
    print(f"sd4c_{e}_{kind}.csv: {len(d):,} rows")


def crosswalk(path: Path) -> None:
    r = pd.read_csv(path, sep="|", dtype=str, encoding="utf-8-sig")
    counties = {f[2:] for f in SD4_FIPS}
    r = r[r.COUNTY_2010.isin(counties) | r.COUNTY_2020.isin(counties)]
    r["BLOCK10"] = "06" + r.COUNTY_2010 + r.TRACT_2010 + r.BLK_2010
    r["BLOCK20"] = "06" + r.COUNTY_2020 + r.TRACT_2020 + r.BLK_2020
    r = r[["BLOCK10", "BLOCK20", "AREALAND_INT", "AREAWATER_INT", "AREALAND_2010", "AREALAND_2020"]]
    r.to_csv(OUT / "sd4c_tab2010_tab2020.csv", index=False)
    print(f"sd4c_tab2010_tab2020.csv: {len(r):,} rows")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    if args and args[0] == "--crosswalk":
        crosswalk(Path(args[1]))
    else:
        for a in args:
            cut(Path(a))
