"""
fetch_mail_splits.py — SD4 results by vote type (mail vs in-person) from the
county results sites that publish them, written to mail_splits.csv.

Counties covered (the others publish totals only; the Election Night app uses
their first election-night drop, first_drop.csv, as a stand-in for mail):
  Madera, Merced   Clarity ENR JSON (vote-type file)
  Stanislaus       county HTML summary report (Vote Center / Vote By Mail)
  Nevada           Enhanced Voting results API
  Calaveras        county PDF summary (In-Person / Vote by Mail) — needs pypdf

A county that fails keeps its previous row. Run on its own or from
fetch_election_results.py.

Usage:
  python3 fetch_mail_splits.py            # fetch and save
  python3 fetch_mail_splits.py --dry-run  # print, don't save
"""

from __future__ import annotations

import argparse
import html
import io
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent
OUT_PATH = DATA_DIR / "mail_splits.csv"
COLS = ["county", "mail_dem", "mail_rep", "inperson_dem", "inperson_rep", "source", "fetched_at"]
UA = {"User-Agent": "Mozilla/5.0 (Macintosh) SD4-tracker"}
TIMEOUT = 25

CLARITY = {"Madera": "https://results.enr.clarityelections.com/CA/Madera/127402",
           "Merced": "https://results.enr.clarityelections.com/CA/Merced/127575"}
STANISLAUS_URL = "https://www.stanvote.gov/past-results/results.htm"
NEVADA_URL = "https://app.enhancedvoting.com/results/public/api/elections/nevada-county-ca/2026StatewideGeneral/data"
CALAVERAS_PAGE = "https://elections.calaverasgov.us/Results/Current-Results"


def _get(url: str) -> requests.Response:
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    return r


def _row(county, md, mr, id_, ir, source):
    return {"county": county, "mail_dem": int(md), "mail_rep": int(mr),
            "inperson_dem": int(id_), "inperson_rep": int(ir), "source": source}


def fetch_clarity(county: str) -> dict:
    base = CLARITY[county]
    ver = _get(f"{base}/current_ver.txt").text.strip()
    summary = _get(f"{base}/{ver}/json/en/summary.json").json()
    contest = next(c for c in summary if re.search(r"senat\w*,? district 4\b", c["C"], re.I))
    names = [n.upper() for n in contest["CH"]]
    i_d = next(i for i, n in enumerate(names) if "BRANDON" in n)
    i_r = next(i for i, n in enumerate(names) if "DUARTE" in n)
    vt = _get(f"{base}/{ver}/json/vt.json").json()
    votes = next(c for c in vt["Contests"] if c["K"] == contest["K"])["Votes"]
    md = mr = id_ = ir = 0
    for j, t in enumerate(vt["Types"]):
        if re.search(r"mail", t, re.I):
            md += votes[i_d][j]; mr += votes[i_r][j]
        elif not re.search(r"provisional|conditional", t, re.I):
            id_ += votes[i_d][j]; ir += votes[i_r][j]
    return _row(county, md, mr, id_, ir, "Clarity ENR")


def fetch_stanislaus() -> dict:
    raw = _get(STANISLAUS_URL).content.decode("latin-1")
    text = html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>", "", raw, flags=re.S)))
    text = re.sub(r"[\s\xa0Â]+", " ", text)
    block = text[text.upper().index("STATE SENATE DISTRICT 4"):][:1500]
    def nums(name):
        m = re.search(name + r"\s+([\d,]+)\s+[\d.]+%\s+([\d,]+)\s+[\d.]+%\s+([\d,]+)", block, re.I)
        return [int(x.replace(",", "")) for x in m.groups()]
    vc_d, vbm_d, _ = nums("JARON BRANDON")
    vc_r, vbm_r, _ = nums("ALEXANDRA DUARTE")
    return _row("Stanislaus", vbm_d, vbm_r, vc_d, vc_r, "county HTML report")


def fetch_nevada() -> dict:
    d = _get(NEVADA_URL).json()
    item = next(b for b in d["ballotItems"] if re.search(r"senate, district 4\b", b["name"][0]["text"], re.I))
    out = {}
    for o in item["summaryResults"]["ballotOptions"]:
        g = {x["groupName"][0]["text"].lower(): x["voteCount"] for x in o["groupResults"]}
        mail = sum(v for k, v in g.items() if "mail" in k)
        inp = sum(v for k, v in g.items() if "person" in k)
        out[o["name"][0]["text"].upper()] = (mail, inp)
    (md, id_), (mr, ir) = out["JARON BRANDON"], out["ALEXANDRA DUARTE"]
    return _row("Nevada", md, mr, id_, ir, "Enhanced Voting API")


def fetch_calaveras() -> dict:
    from pypdf import PdfReader  # optional dependency
    page = _get(CALAVERAS_PAGE).text
    href = re.search(r'href="([^"]*Election%?\s?Results[^"]*\.pdf[^"]*)"', page, re.I) or \
           re.search(r'href="([^"]*\.pdf[^"]*)"', page, re.I)
    url = requests.compat.urljoin(CALAVERAS_PAGE, html.unescape(href.group(1)).replace(" ", "%20"))
    reader = PdfReader(io.BytesIO(_get(url).content))
    text = " ".join(p.extract_text() or "" for p in reader.pages)
    text = re.sub(r"\s+", " ", text)
    block = text[re.search(r"STATE SENATOR,? DISTRICT 4", text, re.I).start():][:1200]
    def nums(name):
        m = re.search(name + r"\s+DEM|" + name + r"\s+REP", block, re.I)
        tail = block[m.end():]
        n = re.findall(r"([\d,]+)\s+[\d.]+%", tail)[:2]
        return [int(x.replace(",", "")) for x in n]
    ip_d, vbm_d = nums("JARON BRANDON")
    ip_r, vbm_r = nums("ALEXANDRA DUARTE")
    return _row("Calaveras", vbm_d, vbm_r, ip_d, ip_r, "county PDF report")


FETCHERS = {"Madera": lambda: fetch_clarity("Madera"), "Merced": lambda: fetch_clarity("Merced"),
            "Stanislaus": fetch_stanislaus, "Nevada": fetch_nevada, "Calaveras": fetch_calaveras}


def fetch_all() -> pd.DataFrame:
    try:
        old = pd.read_csv(OUT_PATH)
    except Exception:
        old = pd.DataFrame(columns=COLS)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for county, fn in FETCHERS.items():
        try:
            r = fn()
            r["fetched_at"] = now
            print(f"  {county}: mail D {r['mail_dem']:,} R {r['mail_rep']:,} · "
                  f"in-person D {r['inperson_dem']:,} R {r['inperson_rep']:,}")
            rows.append(r)
        except Exception as e:
            print(f"  ⚠  {county}: {type(e).__name__}: {e} — keeping previous", file=sys.stderr)
            prev = old[old["county"] == county]
            if not prev.empty:
                rows.append(prev.iloc[0].to_dict())
    return pd.DataFrame(rows, columns=COLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print("Fetching mail / in-person splits…")
    df = fetch_all()
    if args.dry_run:
        print(df.to_string(index=False))
        return
    if not df.empty:
        df.to_csv(OUT_PATH, index=False)
        print(f"✅ Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
