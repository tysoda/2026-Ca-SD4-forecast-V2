# data/precinct — SWDB precinct data for SD4's 13 counties

Staged for porting the block-group lean/turnout method from `tysoda/Assembly-District-9`
(`forecast/panel.py`, `forecast/model.py`). **Not used by the live app yet.**

Source: Statewide Database (statewidedatabase.org) statewide files by SR precinct, cut to Alpine, Amador,
Calaveras, El Dorado, Inyo, Madera, Mariposa, Merced, Mono, Nevada, Placer, Stanislaus and Tuolumne with
`scripts/cut_swdb_sd4.py`. Every precinct in those counties and every column is kept, values exactly as published
(SWDB masks very small precincts' contest votes as `***`).

| File | What |
|---|---|
| `sd4c_{e}_sov.csv` | statement of vote by SR precinct (results; `SDDIST` = Senate district) |
| `sd4c_{e}_reg.csv` | registration by SR precinct (party, gender, age, surname groups) |
| `sd4c_{e}_blk.csv` | SR precinct → census block map (`BLKREG` = registrants in the block) |
| `sd4c_tab2010_tab2020.csv` | Census 2010→2020 tabulation block relationship file (overlap areas) |
| `state_contest_totals.csv` | statewide Dem/Rep totals per partisan contest, p12–g24 (for leans vs the state) |

Elections `{e}`: p12 g12 p14 g14 p16 g16 p18 g18 p20 g20 p22 g22 p24 g24.

**District lines.** p22 onward use the 2021 maps: SD4 is `SDDIST == 4` (≈608k registered in g22, 622k in g24).
p12–g20 use the **old** lines, where `SDDIST == 4` is a different area. Put those elections on current SD4 by
precinct → block (`sd4c_{e}_blk.csv`) → 2010→2020 crosswalk → the 2020 blocks of current SD4 precincts (from the
p24 block map), weighted by registrants, as in `Assembly-District-9/scripts/rebuild_old_lines.py`.

**Block vintages.** p12–g20 block maps use 2010 census blocks; p22–g24 use 2020 blocks.

**Precinct keys** are SR precincts (`SRPREC_KEY`) throughout, matching the block maps and registration files.
