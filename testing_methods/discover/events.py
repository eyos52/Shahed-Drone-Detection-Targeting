#!/usr/bin/env python3
"""Build the raid ledger from the Air Force Command Telegram channel (kpszsu).

The ledger is NOT a query generator -- discovery walks channels, never dates,
because that is what makes it lag-immune. The ledger does two other jobs:

  coverage   which incursions do we have footage for, and which nights are
             still uncovered? Gaps drive targeted search, everything else
             comes from channel sweeps.
  correlate  a clip posted at 21:40 showing a drone over Dnipro lines up with
             "21:36 jet UAV heading for Dnipro". That pins an undated repost to
             a real event, which is how lagged posts get their date back.

kpszsu turned out to post minute-resolution tracking rather than morning
totals, which is considerably more useful, and it tags each post with an emoji
that classifies the threat for free:

    shuttle   ударні БпЛА      strike UAV group -- Shahed-136 / Geran-2
    motorbike реактивний БпЛА  jet UAV          -- Shahed-238 / Geran-3
    comet     балістика        ballistic
    rocket    крилаті ракети   cruise missile

Only the two UAV classes matter for this project; the rest are recorded so the
ledger can distinguish "no footage found" from "nothing was flying".

Dependencies: stdlib only.

Usage:
    python events.py --out ../../manifests [--max-pages 40] [--delay 1.2]
"""
import argparse
import html
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import EVENT_FIELDS, upsert
from telegram import _POST, _TAG, _TEXT, _TIME, _WRAP, fetch

SOURCE = "kpszsu"

# Emoji -> threat class. The channel is consistent about these markers, but the
# Ukrainian text is checked too so a changed emoji does not silently reclassify.
EMOJI_CLASS = {
    "\U0001F6F5": "shahed_group",   # scooter  - ударні БпЛА
    "\U0001F3CD": "shahed_jet",     # motorbike - реактивний БпЛА
    "☄":     "ballistic",
    "\U0001F680": "cruise",
    "✈":     "aircraft",
}
TEXT_CLASS = [
    (re.compile(r"реактивн\w*\s+БпЛА", re.I), "shahed_jet"),
    (re.compile(r"ударн\w*\s+БпЛА", re.I), "shahed_group"),
    (re.compile(r"балістич", re.I), "ballistic"),
    (re.compile(r"крилат\w*\s+ракет", re.I), "cruise"),
]
# Ukrainian place names appear in several cases; keep the raw phrase rather than
# pretending to normalise it, and let the human reading the ledger judge.
_LOC = re.compile(r"(?:на|у напрямку|курсом на|над|вздовж)\s+([А-ЯЇІЄҐ][^,.!]{2,40})")


def classify(emoji, text):
    for pat, cls in TEXT_CLASS:
        if pat.search(text):
            return cls
    return EMOJI_CLASS.get(emoji, "other")


def parse_page(body):
    rows, ids = [], []
    for b in _WRAP.split(body)[1:]:
        m = _POST.search(b)
        if not m:
            continue
        mid = m.group(1).rsplit("/", 1)[-1]
        if mid.isdigit():
            ids.append(int(mid))
        txt, t = _TEXT.search(b), _TIME.search(b)
        if not (txt and t):
            continue
        s = " ".join(html.unescape(_TAG.sub(" ", txt.group(1))).split())
        if not s:
            continue
        emoji = next((c for c in s if c in EMOJI_CLASS), "")
        loc = _LOC.search(s)
        rows.append({
            "event_id": f"{SOURCE}/{mid}",
            "ts_utc": t.group(1),
            "source": SOURCE,
            "threat_type": classify(emoji, s),
            "emoji": emoji,
            "location": (loc.group(1).strip() if loc else ""),
            "text": s[:300],
            "n_clips": "",
        })
    return rows, (min(ids) if ids else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="manifests/ directory")
    ap.add_argument("--max-pages", type=int, default=40)
    ap.add_argument("--stop-at-id", default=None)
    ap.add_argument("--delay", type=float, default=1.2)
    a = ap.parse_args()

    url, before, all_rows = f"https://t.me/s/{SOURCE}", None, []
    for page in range(a.max_pages):
        u = url if before is None else f"{url}?before={before}"
        try:
            body, final = fetch(u)
        except Exception as e:                        # noqa: BLE001
            print(f"  stopped: {type(e).__name__}")
            break
        if "/s/" not in final:
            print("  kpszsu has no public preview -- ledger needs another source")
            break
        rows, oldest = parse_page(body)
        if oldest is None:
            break
        all_rows.extend(rows)
        print(f"  page {page + 1}: {len(rows)} events, oldest id {oldest} "
              f"(total {len(all_rows)})")
        if a.stop_at_id and oldest <= int(a.stop_at_id):
            break
        if before is not None and oldest >= before:
            break
        before = oldest
        time.sleep(a.delay)

    added, total = upsert(Path(a.out) / "events.csv", EVENT_FIELDS,
                          "event_id", all_rows)
    by = {}
    for r in all_rows:
        by[r["threat_type"]] = by.get(r["threat_type"], 0) + 1
    print(f"\n{added} new events ({total} in ledger)")
    for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"   {k:<16}{v}")


if __name__ == "__main__":
    main()
