#!/usr/bin/env python3
"""
daily_email.py — daily SD4 ballot-return snapshot by email.

Builds the same headline figures as the tracker's Overview tab (District and
By party), with change since the previous day, a county table and a link to
the tracker, and sends them as an HTML email through Gmail SMTP.

Run by .github/workflows/daily-email.yml. Environment:
  GMAIL_USER          sending Gmail / Google Workspace address
  GMAIL_APP_PASSWORD  16-character app password for that account
  EMAIL_TO            recipients, comma-separated (default: GMAIL_USER)
  TRACKER_URL         optional link to the Streamlit tracker

Local preview (no email sent):
  python3 ballot_tracker/daily_email.py --preview email.html
"""

from __future__ import annotations

import argparse
import os
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as A  # noqa: E402
import data as D      # noqa: E402

GREEN, GREY, AMBER, RED = "#1a6b3c", "#666666", "#d97706", "#b91c1c"


# ── Figures ──────────────────────────────────────────────────────────────────
def today_ca() -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/Los_Angeles").tz_localize(None).normalize()


def optiq_history(today: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    """Saved Optiq county snapshots, with today's live pull replacing any saved
    snapshot from today. Returns (history, source label)."""
    saved = D.load_optiq_county_party()
    live = D.fetch_optiq_county_party_live()
    if not live.empty:
        live = live.assign(snapshot_date=today)
        keep = saved[saved["snapshot_date"] != today] if not saved.empty else saved
        return pd.concat([keep, live], ignore_index=True), "live"
    return saved, "saved"


def party_block(cp_snap: pd.DataFrame) -> dict:
    """District party figures from one Optiq county-feed snapshot."""
    g = cp_snap.groupby("party")[["registered", "returned"]].sum()
    tot_ret, tot_reg = g["returned"].sum(), g["registered"].sum()
    out = {}
    for p in D.PARTIES:
        ret, reg = (int(g.loc[p, "returned"]), int(g.loc[p, "registered"])) if p in g.index else (0, 0)
        out[p] = {"returned": ret, "rate": ret / reg if reg else np.nan,
                  "share_ret": ret / tot_ret if tot_ret else np.nan,
                  "share_reg": reg / tot_reg if tot_reg else np.nan}
    out["dr_ret"] = (out["D"]["returned"] - out["R"]["returned"]) / tot_ret if tot_ret else np.nan
    out["dr_reg"] = out["D"]["share_reg"] - out["R"]["share_reg"]
    out["raw_dr"] = out["D"]["returned"] - out["R"]["returned"]
    return out


def build_figures() -> dict:
    today = today_ca()
    days_out = int((D.EDAY_2026 - today).days)
    sos = D.load_sos_returns()
    base = D.load_2022_baseline()
    curve = A.pdi_curve(D.load_pdi_2022())

    f = {"today": today, "days_out": days_out}

    # SOS headline count, this report and the previous one
    sos_dates = sorted(sos["snapshot_date"].unique())
    _, sm = A.county_table(sos, base, None, 1.0, curve, compare_date=today)
    f["sos"] = sm
    if len(sos_dates) >= 2:
        prev_sos = sos[sos["snapshot_date"] < sos_dates[-1]]
        f["sos_prev"] = A.county_table(prev_sos, base, None, 1.0, curve, compare_date=today)[1]

    # Optiq: today and the latest snapshot from an earlier day
    hist, src = optiq_history(today)
    f["optiq_source"] = src
    if hist.empty:
        f["optiq"] = None
        return f
    t, m = A.optiq_county_table(hist, base, None, 1.0, curve, sos=sos, compare_date=today)
    cur_date = hist["snapshot_date"].max()
    cur = hist[hist["snapshot_date"] == cur_date]
    f.update(optiq=m, counties=t, party=party_block(cur),
             county_dr=A.county_party_table(cur).set_index("county")["dr_margin_ret"])

    prev = hist[hist["snapshot_date"] < cur_date]
    if not prev.empty:
        prev_date = prev["snapshot_date"].max()
        _, pm = A.optiq_county_table(prev, base, None, 1.0, curve, sos=sos, compare_date=prev_date)
        f.update(optiq_prev=pm, prev_date=pd.Timestamp(prev_date),
                 party_prev=party_block(prev[prev["snapshot_date"] == prev_date]))
    return f


# ── HTML ─────────────────────────────────────────────────────────────────────
def pct(v, d=1):
    return "—" if v is None or not np.isfinite(v) else f"{v * 100:.{d}f}%"


def pp(v, d=1):
    return "—" if v is None or not np.isfinite(v) else f"{v * 100:+.{d}f}pp"


def delta(cur, prev, kind="int"):
    """Small grey 'change since yesterday' line under a value."""
    if prev is None or cur is None:
        return ""
    try:
        if not (np.isfinite(cur) and np.isfinite(prev)):
            return ""
    except TypeError:
        return ""
    d = cur - prev
    if kind == "int":
        txt = f"{d:+,.0f}"
    elif kind == "pp":
        txt = f"{d * 100:+.1f}pp"
    else:  # pace ratio, shown in percentage points of pace
        txt = f"{d * 100:+.0f} pts"
    return f'<div style="font-size:12px;color:#888;margin-top:2px">{txt} since yesterday</div>'


def card(label, value, sub="", change="", color=GREEN):
    return (
        f'<td style="vertical-align:top;padding:6px;width:25%">'
        f'<div style="background:#f7f7f5;border-left:3px solid {color};padding:10px 12px">'
        f'<div style="font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;'
        f'color:{color if color != GREEN else "#666"}">{escape(label)}</div>'
        f'<div style="font-family:Menlo,Consolas,monospace;font-size:24px;font-weight:600;color:#111;'
        f'line-height:1.2;margin-top:2px">{value}</div>'
        f'<div style="font-size:12px;color:#888;margin-top:2px">{sub}</div>{change}</div></td>'
    )


def section(title):
    return (f'<div style="font-family:Menlo,Consolas,monospace;font-size:11px;font-weight:600;'
            f'letter-spacing:.12em;text-transform:uppercase;color:#888;border-bottom:1px solid #e0e0e0;'
            f'padding-bottom:4px;margin:22px 0 8px">{escape(title)}</div>')


def pace_color(r):
    if r is None or not np.isfinite(r):
        return GREY
    return GREEN if r >= 1.10 else (AMBER if r >= 0.85 else RED)


def build_html(f: dict, tracker_url: str | None) -> tuple[str, str]:
    today, days_out = f["today"], f["days_out"]
    sm, sp = f["sos"], f.get("sos_prev")
    om, opm = f.get("optiq"), f.get("optiq_prev")
    subject = f"SD4 ballot returns — {today:%a %b %-d} (E−{days_out})"

    h = [f'<div style="font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#222;'
         f'max-width:720px;margin:0 auto">']
    h.append(f'<h2 style="font-family:Menlo,Consolas,monospace;margin:0 0 4px">📬 SD4 Ballot Returns</h2>')
    h.append(f'<div style="font-size:13px;color:#666">{today:%A %B %-d, %Y} · E−{days_out} · '
             f'SOS report {sm["snapshot"]:%b %-d}'
             + (f' · Optiq {"live" if f["optiq_source"] == "live" else "saved"}, '
                f'fetched {om["snapshot"]:%b %-d}' if om else " · Optiq unavailable")
             + '</div>')
    if tracker_url:
        h.append(f'<div style="margin:14px 0"><a href="{escape(tracker_url)}" style="background:{GREEN};'
                 f'color:#fff;text-decoration:none;padding:8px 14px;border-radius:3px;font-size:13px;'
                 f'font-weight:600">Open the tracker →</a></div>')

    # ── District ─────────────────────────────────────────────────────────────
    h.append(section("District"))
    cells = [card("Ballots returned · SOS", f"{sm['returned']:,}",
                  f"of {sm['issued']:,} issued ({sm['snapshot']:%b %-d})"
                  + (f" · Optiq: {om['returned']:,}" if om else ""),
                  delta(sm["returned"], sp["returned"] if sp else None))]
    if om:
        lrd = f["counties"]["latest_return_date"].max()
        n_lrd = int(f["counties"]["latest_return_date"].eq(lrd).sum()) if pd.notna(lrd) else 0
        cells += [
            card("Return rate · Optiq", pct(om["rate"]),
                 f"2022 last pre-election report: {pct(sm['rate_2022_baseline'])}",
                 delta(om["rate"], opm["rate"] if opm else None, "pp")),
            card("Latest return logged", f"{lrd:%b %-d}" if pd.notna(lrd) else "—",
                 f"{n_lrd} of 13 counties" if pd.notna(lrd) else ""),
            card("vs 2022 · Optiq", pct(om["pace_ratio"], 0),
                 f"2022 at E−{om['days_left']}: {pct(om['rate_2022_equiv'])} · 2026 now: {pct(om['rate'])}",
                 delta(om["pace_ratio"], opm["pace_ratio"] if opm else None, "pace"),
                 color=pace_color(om["pace_ratio"])),
        ]
    else:
        cells += [card("Return rate · SOS", pct(sm["rate"]), ""),
                  card("vs 2022 · SOS", pct(sm["pace_ratio"], 0),
                       f"2022 at E−{sm['days_left']}: {pct(sm['rate_2022_equiv'])}",
                       color=pace_color(sm["pace_ratio"])), "<td></td>"]
    h.append('<table style="width:100%;border-collapse:collapse"><tr>' + "".join(cells) + "</tr></table>")
    h.append('<div style="font-size:12px;color:#888">Ballots returned is the official SOS count. Rates and '
             f'comparisons use Optiq. 2022 is taken at today\'s point (E−{days_out}), from SD4\'s 2022 daily '
             'returns.</div>')

    # ── By party ─────────────────────────────────────────────────────────────
    if om:
        P, PP = f["party"], f.get("party_prev")
        h.append(section("By party — Optiq"))
        cells = []
        for p in D.PARTIES:
            q = P[p]
            sub = (f"{pct(q['rate'])} returned · {pct(q['share_ret'])} of returns vs "
                   f"{pct(q['share_reg'])} of reg ({pp(q['share_ret'] - q['share_reg'])})")
            cells.append(card(D.PARTY_LABELS[p], f"{q['returned']:,}", sub,
                              delta(q["returned"], PP[p]["returned"] if PP else None),
                              color=D.PARTY_COLORS[p]))
        cells.append(card("D−R among returns", pp(P["dr_ret"]),
                          f"registration {pp(P['dr_reg'])} · raw D−R {P['raw_dr']:+,}",
                          delta(P["dr_ret"], PP["dr_ret"] if PP else None, "pp"),
                          color=D.PARTY_COLORS["D"] if P["dr_ret"] - P["dr_reg"] >= 0 else D.PARTY_COLORS["R"]))
        h.append('<table style="width:100%;border-collapse:collapse"><tr>' + "".join(cells) + "</tr></table>")

    # ── Counties ─────────────────────────────────────────────────────────────
    if om:
        t = f["counties"].sort_values("tot_reg", ascending=False)
        h.append(section(f"Counties — Optiq, compared with 2022 at E−{om['days_left']}"))
        th = ('style="text-align:right;font-size:11px;color:#666;text-transform:uppercase;letter-spacing:.04em;'
              'padding:5px 8px;border-bottom:2px solid #ddd"')
        rows = [f'<tr><th {th.replace("right", "left")}>County</th><th {th}>Returned</th><th {th}>Rate</th>'
                f'<th {th}>2022 same day</th><th {th}>vs 2022</th><th {th}>D−R returns</th><th {th}>Latest</th></tr>']
        dr = f["county_dr"]

        def cell(v, left=False, color=None, bold=False):
            st = (f"text-align:{'left' if left else 'right'};padding:4px 8px;border-bottom:1px solid #f0f0f0;"
                  f"font-size:13px" + (f";color:{color}" if color else "") + (";font-weight:600" if bold else ""))
            return f'<td style="{st}">{v}</td>'

        for _, r in t.iterrows():
            name = r["county"] + ("*" if r.get("sos_filled") else "")
            lr = r["latest_return_date"]
            rows.append("<tr>" + cell(escape(name), left=True) + cell(f"{int(r['tot_ret']):,}")
                        + cell(pct(r["rate"])) + cell(pct(r["rate_2022_equiv"]))
                        + cell(pct(r["pace_ratio"], 0), color=pace_color(r["pace_ratio"]))
                        + cell(pp(dr.get(r["county"], np.nan)))
                        + cell(f"{lr:%b %-d}" if pd.notna(lr) else "—") + "</tr>")
        rows.append("<tr>" + cell("SD4 total", left=True, bold=True) + cell(f"{om['returned']:,}", bold=True)
                    + cell(pct(om["rate"]), bold=True) + cell(pct(om["rate_2022_equiv"]), bold=True)
                    + cell(pct(om["pace_ratio"], 0), color=pace_color(om["pace_ratio"]), bold=True)
                    + cell(pp(f["party"]["dr_ret"]), bold=True) + cell("") + "</tr>")
        h.append('<table style="width:100%;border-collapse:collapse">' + "".join(rows) + "</table>")
        if t["sos_filled"].any():
            h.append('<div style="font-size:12px;color:#888;margin-top:4px">* Optiq shows 0 returned for this '
                     'county, so its count and date come from the latest SOS report.</div>')

    if f.get("prev_date") is not None:
        h.append(f'<div style="font-size:12px;color:#888;margin-top:18px">"Since yesterday" compares with the '
                 f'Optiq snapshot from {f["prev_date"]:%b %-d}'
                 + (f' and the SOS report from {sp["snapshot"]:%b %-d}' if sp else "") + '.</div>')
    h.append('<div style="font-size:12px;color:#aaa;margin-top:10px">Sent by the SD4 tracker\'s daily-email '
             'GitHub Action.</div></div>')
    return subject, "".join(h)


# ── Send ─────────────────────────────────────────────────────────────────────
def send(subject: str, html: str) -> None:
    user = os.environ["GMAIL_USER"]
    password = os.environ["GMAIL_APP_PASSWORD"].replace(" ", "")
    to = [a.strip() for a in (os.environ.get("EMAIL_TO") or user).split(",") if a.strip()]
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, user, ", ".join(to)
    msg.attach(MIMEText("Open this email in an HTML-capable client to see the snapshot.", "plain"))
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
        s.login(user, password)
        s.sendmail(user, to, msg.as_string())
    print(f"Sent '{subject}' to {', '.join(to)}")


def main():
    ap = argparse.ArgumentParser(description="Daily SD4 ballot-return email.")
    ap.add_argument("--preview", metavar="PATH", help="Write the HTML to PATH instead of sending.")
    args = ap.parse_args()
    figures = build_figures()
    subject, html = build_html(figures, os.environ.get("TRACKER_URL") or None)
    if args.preview:
        Path(args.preview).write_text(html)
        print(f"Preview written to {args.preview} — subject: {subject}")
    else:
        send(subject, html)


if __name__ == "__main__":
    main()
