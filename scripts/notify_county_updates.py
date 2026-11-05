"""
notify_county_updates.py — email when an SD4 county posts new results.

Fetches the SD4 State Senate results from the CA SOS API, compares each county
with its latest row in election_night_history.csv, appends any changes, and
emails a summary of the new batches. Run by the "County Update Email" GitHub
Action every ~10 minutes; the Action commits the updated history file.

Environment (same Gmail account as the daily snapshot email):
  GMAIL_USER, GMAIL_APP_PASSWORD   sender account + app password
  NOTIFY_TO                        recipients, comma-separated (optional;
                                   falls back to EMAIL_TO, then GMAIL_USER)

Usage:
  python3 scripts/notify_county_updates.py              # fetch, record, email
  python3 scripts/notify_county_updates.py --preview f.html   # no email, no history write
"""

from __future__ import annotations

import argparse
import os
import smtplib
import sys
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "election_night"))

import fetch_election_results as F  # noqa: E402
import batches as B                 # noqa: E402

STOP_AFTER = date(2026, 12, 11)     # SOS certification deadline
SOS_URL = "https://electionresults.sos.ca.gov/returns/state-senate/district/4"


def _pct(x) -> str:
    return "—" if pd.isna(x) else f"{x*100:.1f}%"


def _net(n: int) -> str:
    return f"{'D' if n >= 0 else 'R'} +{abs(int(n)):,}"


def build_email(new_b: pd.DataFrame, current: pd.DataFrame, hist_all: pd.DataFrame) -> tuple[str, str]:
    counties = list(new_b["county"])
    tot_d = int(pd.to_numeric(current["dem_votes"], errors="coerce").fillna(0).sum())
    tot_r = int(pd.to_numeric(current["rep_votes"], errors="coerce").fillna(0).sum())
    margin = tot_d - tot_r
    batch_net = int(new_b["net_dem"].sum())
    subject = (f"SD4 update: {', '.join(counties)} — batch {_net(batch_net)}, "
               f"district {_net(margin)}")

    rows = ""
    for _, r in new_b.iterrows():
        col = "#1a6b3c" if r.dem_share_new >= r.dem_share_before else "#b91c1c"
        rows += (f"<tr><td style='padding:4px 8px'><b>{r.county}</b><br>"
                 f"<span style='color:#888;font-size:12px'>{r.last_updated}</span></td>"
                 f"<td style='padding:4px 8px;text-align:right'>{int(r.new_votes):,}</td>"
                 f"<td style='padding:4px 8px;text-align:right;color:{col};font-weight:bold'>{_pct(r.dem_share_new)}</td>"
                 f"<td style='padding:4px 8px;text-align:right'>{_pct(r.dem_share_before)} → {_pct(r.dem_share_after)}</td>"
                 f"<td style='padding:4px 8px;text-align:right'>{_net(r.net_dem)}</td></tr>")
    sb = B.since_baseline(hist_all)
    late = ""
    if sb:
        late = (f"<p>All votes added since the first saved count: <b>{sb['new_votes']:,}</b>, "
                f"<b>{_pct(sb['dem_share_new'])} Dem</b> (count before them: {_pct(sb['dem_share_baseline'])}); "
                f"net {_net(sb['net_dem'])}.</p>")
    th = "padding:4px 8px;text-align:right;border-bottom:2px solid #ddd;font-size:12px;color:#666"
    html = f"""<div style="font-family:Helvetica,Arial,sans-serif;font-size:14px;color:#111;max-width:640px">
<h2 style="margin:0 0 4px 0">SD4: {', '.join(counties)} updated</h2>
<p style="margin:0 0 12px 0">District now: <b>{tot_d:,} D</b> · <b>{tot_r:,} R</b> ·
 <b>{tot_d/(tot_d+tot_r)*100:.1f}% Dem</b> · margin <b>{_net(margin)}</b></p>
<table style="border-collapse:collapse;font-size:14px">
<tr><th style="{th};text-align:left">County</th><th style="{th}">New votes</th><th style="{th}">Dem share of new</th>
<th style="{th}">Dem share before → after</th><th style="{th}">Net</th></tr>
{rows}</table>
{late}
<p style="font-size:12px;color:#888">Dem share is two-party. Green = batch more Democratic than the county's count before it.
<a href="{SOS_URL}">SOS SD4 results</a></p></div>"""
    return subject, html


def send(subject: str, html: str) -> None:
    user = os.environ["GMAIL_USER"]
    password = os.environ["GMAIL_APP_PASSWORD"].replace(" ", "")
    raw = os.environ.get("NOTIFY_TO") or os.environ.get("EMAIL_TO") or user
    to = [a.strip() for a in raw.split(",") if a.strip()]
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, user, ", ".join(to)
    msg.attach(MIMEText("A county in SD4 posted new results. Open in an HTML-capable client for details.", "plain"))
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
        s.login(user, password)
        s.sendmail(user, to, msg.as_string())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", metavar="HTML", help="Write the email to a file; don't send or save history")
    args = ap.parse_args()

    if date.today() > STOP_AFTER and not args.preview:
        print("Past certification — nothing to do.")
        return 0

    current = F.fetch_race("state-senate/district/4", F.SD4_COUNTIES, "SD4 State Senate")
    hist = F.load_history()
    new_rows = F.new_history_rows(current, hist)
    if new_rows.empty:
        print("No county changes.")
        return 0

    hist_all = pd.concat([hist, new_rows], ignore_index=True)
    # Only counties that already had a recorded count make a batch (first sighting = baseline)
    new_b = B.compute_batches(hist_all)
    new_b = new_b[new_b["fetched_at"] == new_rows["fetched_at"].iloc[0]]
    print(f"Changed: {', '.join(new_rows['county'])}; batches: {', '.join(new_b['county']) or 'none'}")

    if args.preview:
        if not new_b.empty:
            subject, html = build_email(new_b, current, hist_all)
            Path(args.preview).write_text(f"<p><b>Subject:</b> {subject}</p>{html}")
            print(f"Preview written to {args.preview}")
        return 0

    F.append_history(current)          # record before emailing so a send failure doesn't resend forever
    if not new_b.empty:
        subject, html = build_email(new_b, current, hist_all)
        try:
            send(subject, html)
            print(f"Emailed: {subject}")
        except KeyError:
            print("GMAIL_USER / GMAIL_APP_PASSWORD not set — email skipped.")
        except Exception as e:
            print(f"Email failed: {e}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
