# SD4 Ballot Return Tracker

Standalone Streamlit app for SD4 vote-by-mail returns. It lives in the forecast repo so it can read the
same CSVs the GitHub Actions already refresh, but it doesn't import the forecast model.

## Run

```bash
streamlit run ballot_tracker/app.py
```

**Streamlit Cloud:** New app → same repo/branch → main file path `ballot_tracker/app.py`.
Uses the root `requirements.txt` and `runtime.txt`.

## Files

| File | What it does |
|---|---|
| `app.py` | UI — Overview, Counties, Crosstabs, Trends, Outstanding, 2022 comparison, Method & data |
| `data.py` | Loaders for the repo-root CSVs and the live Optiq API (plain pandas) |
| `analysis.py` | Crosstabs, county pace, velocity, county × party, final-electorate scenario |

## Data

| Source | File | Refreshed by |
|---|---|---|
| SOS ballot status report | `ballot_returns.csv` | *Update Ballot Returns* Action |
| Optiq party × age × gender × ethnicity | `optiq_crosstabs.csv` | *Update Optiq Crosstabs* Action |
| Optiq county × party | `optiq_county_party.csv` | *Update Optiq Crosstabs* Action (new) |
| 2022 baseline | `vbm_2022_baseline.csv` | static |
| Party registration | `sd4_party_registration.csv` | manual |
| 2022 SD4 daily returns by party (read from PDI's tracker chart, avg of 3 readings) | `pdi_2022_daily_sd4.csv` | static |
| 2022 SD4 party registration (SOS 15-day report, Oct 24 2022; split counties estimated) | `sd4_party_registration_2022.csv` | static |
| **Optional** county-level 2022 return curve | `vbm_2022_daily.csv` | manual |

The 2022 pace baseline follows the PDI 2022 curve by default (sidebar can switch to an assumed shape).
A county-level `vbm_2022_daily.csv` overrides both when present. Schema:
`county, days_before_eday, vbm_returned` (SD4-portion counts, one row per county per day).

Snapshots whose numbers are identical to the previous one are dropped before computing trends, so
weekend re-publishes don't show up as zero-return days.
