# Optiq raw snapshots (archive)

Raw responses from Optiq's ballot-returns API (`abev.civiq.vote/api/ballot-returns/STATE%20SENATE%204`),
saved by `fetch_optiq_crosstabs.py`. One file per California day, overwritten on each hourly run, so
each file holds that day's latest Optiq load.

Each row is a party × age × gender × ethnicity × `returnDate` cell with a voter count; rows with no
`returnDate` are ballots issued but not returned. Unlike `optiq_crosstabs.csv`, this keeps the return
date, which separates ballots received by Election Day (checked in late by counties) from true late
mail (return dates after Election Day). Saved in case Optiq takes the district page down after the election.

- `2026-general/` — 2026 General, from Nov 5 2026 onward.
