#!/usr/bin/env python3
"""Channel registry and sweep driver -- the thing that makes discovery lag-immune.

Searching by raid date cannot find a clip posted six months after the raid.
Enumerating a channel's whole history can, because the clip is still in the
timeline. So the registry, not the calendar, is the unit of discovery, and
growing it is how coverage grows.

Three operations:

  --seed    write the starting registry (below). Classes matter: an `official`
            channel feeds the event ledger, `osint` channels carry the raw
            combat footage, `news` channels mostly carry studio re-cuts and
            need heavier screening.
  --probe   check which channels still render a public preview. This is opt-in
            per channel and changes over time -- measured 2026-10-29, 14 of 19
            reachable, and the misses include some of the most relevant names
            (militarnyi, DefenceU), so this is not a one-off check.
  --sweep   run the Telegram enumerator across every reachable channel,
            resuming each from its recorded last_seen_id.

Snowballing (adding the accounts that reposted a clip we already have) is the
other growth path; those rows get added_by=snowball so the seed set stays
distinguishable from what the process found on its own.

Dependencies: stdlib only.

Usage:
    python channels.py --seed  --out ../../manifests
    python channels.py --probe --out ../../manifests
    python channels.py --sweep --out ../../manifests [--max-pages 40]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import (CANDIDATE_FIELDS, CHANNEL_FIELDS, read_rows, upsert)
from telegram import enumerate_channel, fetch

# Reachability measured on M3, 2026-10-29. Unreachable ones are kept in the
# registry on purpose: a channel that blocks preview today may open later, and
# recording the miss stops us rediscovering the same dead end every sweep.
SEED = [
    # handle,                platform,   class
    ("kpszsu",               "telegram", "official"),   # Air Force Cmd -> event ledger
    ("operativnoZSU",        "telegram", "osint"),
    ("DeepStateUA",          "telegram", "osint"),
    ("ButusovPlus",          "telegram", "osint"),       # highest video density seen
    ("serhii_flash",         "telegram", "osint"),
    ("ssternenko",           "telegram", "osint"),
    ("war_monitor",          "telegram", "osint"),
    ("lachentyt",            "telegram", "osint"),
    ("monitorwar",           "telegram", "osint"),       # dormant since 2023-11
    ("insiderUKR",           "telegram", "news"),
    ("suspilnenews",         "telegram", "news"),
    ("nikvesti",             "telegram", "news"),        # Mykolaiv, local
    ("vanek_nikolaev",       "telegram", "news"),        # Mykolaiv, local
    ("UkraineNow",           "telegram", "news"),
    ("ukrainenowenglish",    "telegram", "news"),        # dormant since 2022-10
    # no public preview as of 2026-10-29 -- recorded so we stop retrying blind
    ("militarnyi",           "telegram", "news"),
    ("DefenceU",             "telegram", "official"),
    ("povitryanitrivogy",    "telegram", "official"),
    ("hueviyrozvedos",       "telegram", "osint"),
    ("Ukraine_Vitbot",       "telegram", "osint"),
    ("mykolaivskyi",         "telegram", "news"),
    ("odesa_infocenter",     "telegram", "news"),
    # X seeds, lifted from data/x_video_urls.txt -- enumerated by x.py
    ("Osinttechnical",       "x", "osint"),
    ("sentdefender",         "x", "osint"),
    ("front_ukrainian",      "x", "osint"),
    ("bayraktar_1love",      "x", "osint"),
    ("Drone_Wars_",          "x", "osint"),
    ("wilendhornets",        "x", "osint"),
]


def do_seed(out):
    rows = [{"handle": h, "platform": p, "channel_class": c,
             "preview_available": "", "enumerated_through": "", "last_seen_id": "",
             "n_found": "", "added_by": "seed", "notes": ""}
            for h, p, c in SEED]
    added, total = upsert(out / "channels.csv", CHANNEL_FIELDS, "handle", rows)
    print(f"seeded {added} new channels ({total} in registry)")


def do_probe(out, delay=0.8):
    reg = read_rows(out / "channels.csv", "handle")
    if not reg:
        print("registry is empty -- run --seed first")
        return
    rows = []
    for h, r in reg.items():
        if r.get("platform") != "telegram":
            continue
        try:
            _, final = fetch(f"https://t.me/s/{h}", timeout=20)
            ok = "/s/" in final
        except Exception as e:                            # noqa: BLE001
            print(f"  {h:<22} ERROR {type(e).__name__}")
            continue
        print(f"  {h:<22}{'reachable' if ok else 'no public preview'}")
        rows.append({"handle": h, "preview_available": "yes" if ok else "no"})
        time.sleep(delay)
    upsert(out / "channels.csv", CHANNEL_FIELDS, "handle", rows)
    n = sum(1 for r in rows if r["preview_available"] == "yes")
    print(f"\n{n}/{len(rows)} telegram channels reachable")


def do_sweep(out, max_pages, delay, only_class=None):
    reg = read_rows(out / "channels.csv", "handle")
    todo = [(h, r) for h, r in reg.items()
            if r.get("platform") == "telegram"
            and r.get("preview_available") != "no"
            and (not only_class or r.get("channel_class") == only_class)]
    if not todo:
        print("nothing to sweep -- run --probe first")
        return
    grand = 0
    for h, r in todo:
        print(f"[{h}]")
        rows, status, newest = enumerate_channel(
            h, max_pages=max_pages, stop_at_id=r.get("last_seen_id") or None,
            delay=delay, log=lambda s: None)
        added, _ = upsert(out / "candidates.csv", CANDIDATE_FIELDS,
                          "candidate_id", rows)
        upsert(out / "channels.csv", CHANNEL_FIELDS, "handle", [{
            "handle": h,
            "preview_available": "no" if status == "no_public_preview" else "yes",
            "enumerated_through": time.strftime("%Y-%m-%d"),
            "last_seen_id": newest or r.get("last_seen_id", ""),
            "n_found": len(rows), "notes": status}])
        grand += added
        print(f"   {status}: {len(rows)} video posts, {added} new")
    print(f"\nsweep added {grand} candidates")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="manifests/ directory")
    ap.add_argument("--seed", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--class", dest="only_class", default=None,
                    help="restrict --sweep to one channel class")
    ap.add_argument("--max-pages", type=int, default=40)
    ap.add_argument("--delay", type=float, default=1.2)
    a = ap.parse_args()
    out = Path(a.out)
    if a.seed:
        do_seed(out)
    if a.probe:
        do_probe(out, a.delay)
    if a.sweep:
        do_sweep(out, a.max_pages, a.delay, a.only_class)
    if not (a.seed or a.probe or a.sweep):
        ap.error("pick one of --seed / --probe / --sweep")


if __name__ == "__main__":
    main()
