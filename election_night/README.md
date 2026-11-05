# SD4 Election Night Tracker

Standalone Streamlit app for live SD4 results — the forecast app's 🌙 Election Night tab, lifted into its
own entry point. It lives in the forecast repo so it reads the same CSVs, but it doesn't import
`election_forecast.py`. The forecast app's tab is unchanged.

## Run

```bash
streamlit run election_night/app.py
```

**Streamlit Cloud:** New app → same repo/branch → main file path `election_night/app.py`.
Uses the root `requirements.txt` and `runtime.txt`.

## Files

| File | What it does |
|---|---|
| `app.py` | UI — district summary, live win probability, county results, manual entry, targets, key counties, governor check |
| `prior.py` | Pre-results prior: loads `forecast_params.json`, runs the same Monte Carlo as the forecast app |

## The prior

`prior.py` reproduces the forecast app's simulation with its **default** settings: model state environment,
Block Group leans (Linear/Average if the BG panel is missing), no turnout adjustments, no candidate
adjustment, no polling blend. The sidebar exposes simulation count, a manual margin override and the lean
method. The seed is fixed (`SEED = 2026`), so numbers don't jump between refreshes or viewers.

If you change the forecast app's simulation, copy the change into `prior.run_simulation`.

## Data

| Source | File | Refreshed by |
|---|---|---|
| SD4 State Senate by county (SOS) | `election_night.csv` | `fetch_election_results.py` (run by the app) |
| Governor by county, all 58 (SOS) | `governor_night.csv` | same |
| Optiq party composition (mail-ballot boost) | `optiq_crosstabs.csv` | *Update Optiq Crosstabs* Action |
| Mock waves | `mock_results/<scenario>/*.csv` | `generate_mock_results.py` |

## Differences from the forecast-app tab

- **Auto-refresh doesn't block the page.** A fragment ticks each minute and fetches when the last fetch
  (by any viewer on the same server) is 10+ minutes old; the first load fetches too if results are stale.
  Paused while a mock scenario is selected. Fetch timeout raised from 60 s to 120 s (71 county calls).
- **Governor totals come from `governor_night.csv`** (summed across counties). Manual entry still works and
  overrides them for that session; "Use SOS totals instead" clears the override.
- **Manual refresh busts the cache** (the tab's `load_night_csv.clear()` ran before the function existed).
- **Save results is disabled while a mock is loaded**, so mock numbers can't be written to the live CSV.
- Timestamps are Pacific.

On Streamlit Cloud the fetched CSVs live on the app's container disk, so manual edits and fetched results
reset on restart/redeploy, and the next SOS fetch overwrites manual entries.

## Late count (added 4 Nov)

- **History:** each fetch appends a row per county whose count changed to `election_night_history.csv`
  (seeded with the ~midnight 3–4 Nov count). `batches.py` turns it into per-update batches; the app's
  **Late Count — New Batches** section shows each batch's Dem share against the county's count before it.
- **Last Update** (County Results) is each county's latest SOS report time; 🆕 = within the past hour.
- **Unprocessed ballots:** enter the SOS *Unprocessed Ballots Status* figures (whole county) in
  `unprocessed_ballots.csv` (`county,unprocessed_ballots,as_of`). The app scales split counties by
  `partial_county_fractions.csv` and uses them for Est. Votes Remaining (marked *). The SOS page is behind
  bot protection, so it can't be fetched automatically.
- **Email:** the *County Update Email* Action (`scripts/notify_county_updates.py`) checks the SOS feed every
  ~10 min, commits new history rows and emails the new batches via the daily-email Gmail secrets
  (`NOTIFY_TO` secret overrides recipients). It stops after 11 Dec 2026.
- The fetcher no longer blanks a county if one request fails; it keeps the last saved numbers.

## Late-vote model (default win probability, added 4 Nov)

`late_model.py`: final = votes counted + remaining votes × late-vote Dem share, per county.
- **Late-vote lean:** the county's mail-ballot Dem share where it reports by vote type (`mail_splits.csv`,
  refreshed by `fetch_mail_splits.py` on every results fetch — Madera and Merced via Clarity, Stanislaus via its
  HTML report, Nevada via Enhanced Voting, Calaveras via its PDF report if `pypdf` is installed); otherwise its
  first election-night drop (`first_drop.csv`, almost all early mail). Post-election-day batches (posted after
  5 am on 4 Nov) are blended in, weighted by votes (prior weight 3,000 votes).
- **Uncertainty:** district-wide late-lean shift (SD 2 pt), county shift (3 pt, shrinking with late batches),
  remaining-vote size (20%, or 7% with SOS unprocessed figures). All adjustable under ⚙️ Late-vote model settings.
- The previous mail-adjusted method (Optiq outstanding-ballot boost) is kept in a collapsed panel for comparison.
- The late-vote lean assumptions are not yet checked against 2022/2024 county canvass data.
