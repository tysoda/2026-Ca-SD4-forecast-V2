#!/usr/bin/env python3
"""One-off: find archived copies of SOS ballot status reports on the Wayback Machine and
download the last pre-election one for each past election."""
import json, time, urllib.request, datetime as dt
from pathlib import Path
OUT = Path("sos_archive"); UA = {"User-Agent": "Mozilla/5.0 (SD4 ballot tracker; one-off archive fetch)"}
def get(url, timeout=60):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
            return r.read()
    except Exception as e:
        print("  !", url[:120], e); return None
ELECTIONS = {"2022-general": dt.date(2022, 11, 8), "2024-primary": dt.date(2024, 3, 5),
             "2024-general": dt.date(2024, 11, 5), "2026-primary": dt.date(2026, 6, 2)}
log = []
for name, eday in ELECTIONS.items():
    prefix = f"elections.cdn.sos.ca.gov/statewide-elections/{name}/"
    cdx = get("https://web.archive.org/cdx/search/cdx?url=" + prefix + "*&output=json&fl=timestamp,original,statuscode,mimetype&filter=statuscode:200&collapse=urlkey&limit=2000")
    rows = json.loads(cdx)[1:] if cdx else []
    bsr = [r for r in rows if "bsr" in r[1].lower()]
    log.append(f"## {name}: {len(rows)} archived files, {len(bsr)} bsr")
    log += [f"   {r[0]} {r[1]}" for r in bsr]
    # every capture of bsr files (not collapsed) up to election day, to get the last pre-election one
    for path in sorted({r[1] for r in bsr}):
        caps = get("https://web.archive.org/cdx/search/cdx?url=" + path.split("://")[-1] + "&output=json&fl=timestamp,original,statuscode&filter=statuscode:200&limit=500")
        caps = json.loads(caps)[1:] if caps else []
        caps = [c for c in caps if c[0][:8] <= eday.strftime("%Y%m%d")]
        if not caps: continue
        ts, orig = caps[-1][0], caps[-1][1]
        data = get(f"https://web.archive.org/web/{ts}id_/{orig}")
        time.sleep(1)
        if data and len(data) > 1000:
            p = OUT / name / f"wayback-{ts}-{orig.rsplit('/', 1)[-1]}"; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
            log.append(f"   saved {p} ({len(data)} bytes, captured {ts})")
    time.sleep(1)
(OUT / "WAYBACK.txt").write_text("\n".join(log)); print("\n".join(log))
