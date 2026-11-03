#!/usr/bin/env python3
"""One-off: download SOS ballot status reports (2026 General latest + final pre-election
report of past elections) and dump each workbook's sheets and headers.
Output: sos_archive/<election>/<file>, sos_archive/LAYOUT.txt, sos_archive/FOUND.txt"""
import datetime as dt, io, re, time, urllib.request, urllib.parse
from pathlib import Path
import openpyxl

OUT = Path("sos_archive"); OUT.mkdir(exist_ok=True)
CDN = "https://elections.cdn.sos.ca.gov/"
UA = {"User-Agent": "Mozilla/5.0 (SD4 ballot tracker; one-off archive fetch)"}

def get(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            return r.read(), r.geturl()
    except Exception as e:
        return None, str(getattr(e, "code", e))

ELECTIONS = {   # name: (election day, CDN folder guesses)
    "2022-general": (dt.date(2022, 11, 8), ["statewide-elections/2022-general"]),
    "2024-primary": (dt.date(2024, 3, 5),  ["statewide-elections/2024-primary"]),
    "2024-general": (dt.date(2024, 11, 5), ["statewide-elections/2024-general"]),
    "2025-special": (dt.date(2025, 11, 4), ["statewide-elections/2025-special", "statewide-elections/2025-statewide-special",
                                            "statewide-elections/2025-special-election", "special-elections/2025-statewide"]),
    "2026-primary": (dt.date(2026, 6, 2),  ["statewide-elections/2026-primary"]),
    "2026-general": (dt.date(2026, 11, 3), ["statewide-elections/2026-general"]),
}
found = []

# 1) Guess dated file names around election day (E-4 .. E-0, plus a few days after for late reports)
for name, (eday, folders) in ELECTIONS.items():
    days = range(-4, 1) if name != "2026-general" else range(-3, 0)
    for folder in folders:
        hit = False
        for off in days:
            d = eday + dt.timedelta(days=off)
            for fmt in ("%m-%d-%Y", "%m-%d-%y"):
                for ext in ("xlsx", "pdf"):
                    url = f"{CDN}{folder}/bsr-statistics-{d.strftime(fmt)}.{ext}"
                    data, info = get(url)
                    time.sleep(0.15)
                    if data and len(data) > 1000 and not data[:15].lower().startswith(b"<!doctype"):
                        p = OUT / name / url.rsplit("/", 1)[-1]; p.parent.mkdir(exist_ok=True); p.write_bytes(data)
                        found.append(f"{name}\tE{off:+d}\t{url}\t{len(data)}"); hit = True
        # undated file in the folder too
        for ext in ("xlsx", "pdf"):
            url = f"{CDN}{folder}/bsr-statistics.{ext}"
            data, info = get(url)
            if data and len(data) > 1000 and not data[:15].lower().startswith(b"<!doctype"):
                p = OUT / name / f"undated-{url.rsplit('/', 1)[-1]}"; p.parent.mkdir(exist_ok=True); p.write_bytes(data)
                found.append(f"{name}\tundated\t{url}\t{len(data)}")
        if hit:
            break

# 2) Crawl SOS pages for any ballot-status links (in case folder names differ)
seeds = ["https://www.sos.ca.gov/elections/prior-elections/statewide-election-results",
         "https://www.sos.ca.gov/elections/upcoming-elections",
         "https://www.sos.ca.gov/elections/ballot-status"]
seen, links = set(), set()
queue = [(u, 0) for u in seeds]
while queue and len(seen) < 120:
    url, depth = queue.pop(0)
    if url in seen: continue
    seen.add(url)
    data, info = get(url); time.sleep(0.2)
    if not data: continue
    html = data.decode("utf-8", "ignore")
    for href in re.findall(r'href="([^"#]+)"', html):
        full = urllib.parse.urljoin(url, href)
        if re.search(r"bsr|ballot-status|ballot status", full, re.I):
            links.add(full)
        if depth < 2 and "sos.ca.gov/elections" in full and re.search(r"20(22|24|25|26)|ballot|prior|upcoming|general|primary|special", full, re.I):
            queue.append((full, depth + 1))
(OUT / "CRAWL_LINKS.txt").write_text("\n".join(sorted(links)))
for l in sorted(links):
    if re.search(r"\.(xlsx|pdf)$", l, re.I) and not any(l in f for f in found):
        data, info = get(l); time.sleep(0.15)
        if data and len(data) > 1000:
            p = OUT / "crawl" / l.rsplit("/", 1)[-1]; p.parent.mkdir(exist_ok=True); p.write_bytes(data)
            found.append(f"crawl\t-\t{l}\t{len(data)}")
(OUT / "FOUND.txt").write_text("\n".join(found))

# 3) Layout dump for every workbook
lines = []
for p in sorted(OUT.rglob("*.xlsx")):
    lines.append(f"===== {p}")
    try:
        wb = openpyxl.load_workbook(io.BytesIO(p.read_bytes()), data_only=True)
        for ws in wb.worksheets:
            lines.append(f"--- sheet: {ws.title}  ({ws.max_row} rows x {ws.max_column} cols)")
            for row in ws.iter_rows(min_row=1, max_row=8, values_only=True):
                lines.append("    " + " | ".join("" if v is None else str(v)[:40] for v in row))
            for row in ws.iter_rows(values_only=True):
                if row and str(row[0]).strip().lower() in ("el dorado", "stanislaus", "state totals", "total", "totals"):
                    lines.append("  * " + " | ".join("" if v is None else str(v)[:20] for v in row))
    except Exception as e:
        lines.append(f"   could not open: {e}")
(OUT / "LAYOUT.txt").write_text("\n".join(lines))
print("\n".join(found)); print(f"{len(links)} crawl links")
