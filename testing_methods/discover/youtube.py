#!/usr/bin/env python3
"""Enumerate YouTube channels and searches -- metadata only, nothing downloaded.

Complements the Telegram leg rather than duplicating it. Two reasons it earns
its place even though Telegram is where the footage originates:

  * Several of the most relevant outlets closed their Telegram preview but run
    an open YouTube channel. Militarnyi is the example: t.me/s/militarnyi 302s
    away with nothing, youtube.com/@Militarnyi enumerates fine.
  * Search reaches channels nobody has put in the registry yet, which is how
    the registry grows.

Discovery runs flat (`--flat-playlist`): one request per page, no per-video
extraction. That is fast but the metadata is thin -- measured on M3, flat mode
returns id, title and duration, while upload_date, timestamp and channel all
come back NA. So posted_date is deliberately left blank here and filled by
--hydrate, which does the expensive per-video pass over only the rows that
survived prefiltering.

Queries default to the terminology actually seen in the Telegram sweep, which
is not what an English search would guess: the jet variants (Герань-3/4/5,
"реактивний БпЛА", "дрон-камікадзе") now dominate the traffic and none of them
match a naive /shahed/ query.

Dependencies: yt-dlp on PATH or in the same venv.

Usage:
    python youtube.py --out ../../manifests --search-default [--n 50]
    python youtube.py --out ../../manifests --channel @Militarnyi [--n 200]
    python youtube.py --out ../../manifests --hydrate [--limit 100]
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import (CANDIDATE_FIELDS, CHANNEL_FIELDS, blank_candidate,
                      candidate_id, read_rows, upsert)

YTDLP = str(Path(sys.executable).parent / "yt-dlp")

# Built from the captions the Telegram sweep actually returned.
DEFAULT_QUERIES = [
    "збиття шахеда",
    "збиття Герань",
    "реактивний БпЛА збиття",
    "дрон-камікадзе збиття ППО",
    "мобільна вогнева група шахед",
    "нічна атака шахедів",
    "перехоплювач шахед",
    "shahed shot down thermal",
    "shahed intercept night",
]

SEP = "\x1f"
FMT = SEP.join(["%(id)s", "%(title)s", "%(duration)s", "%(channel)s",
                "%(uploader_id)s", "%(upload_date)s"])


def run(args, timeout=300):
    try:
        p = subprocess.run([YTDLP, *args], capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return []
    if p.returncode != 0 and not p.stdout.strip():
        sys.stderr.write((p.stderr or "")[-300:] + "\n")
        return []
    return [l for l in p.stdout.splitlines() if SEP in l]


def to_rows(lines, channel_hint=""):
    rows = []
    for l in lines:
        f = l.split(SEP)
        while len(f) < 6:
            f.append("NA")
        vid, title, dur, chan, uploader, up = f[:6]
        chan = next((c for c in (chan, uploader, channel_hint)
                     if c and c != "NA"), "")
        rows.append(blank_candidate(
            candidate_id=candidate_id("youtube", chan, vid),
            platform="youtube", channel=chan,
            url=f"https://www.youtube.com/watch?v={vid}",
            title=(title if title != "NA" else "")[:300],
            duration_s=(dur if dur.isdigit() else ""),
            # flat mode returns NA here; --hydrate fills it later
            posted_date=(f"{up[:4]}-{up[4:6]}-{up[6:8]}"
                         if up and up != "NA" and len(up) == 8 else ""),
        ))
    return rows


def enum_search(query, n):
    return to_rows(run(["--flat-playlist", "--no-warnings", "--ignore-errors",
                        "--print", FMT, f"ytsearch{n}:{query}"]))


def enum_channel(handle, n):
    h = handle if handle.startswith("http") else \
        f"https://www.youtube.com/{handle if handle.startswith('@') else '@' + handle}/videos"
    return to_rows(run(["--flat-playlist", "--no-warnings", "--ignore-errors",
                        "--playlist-end", str(n), "--print", FMT, h]),
                   channel_hint=handle.lstrip("@"))


def hydrate(out, limit):
    """Fill upload_date/resolution/fps for rows that survived prefiltering.

    Expensive (one extraction per video), so it runs only on keep/maybe.
    """
    path = Path(out) / "candidates.csv"
    rows = read_rows(path, "candidate_id")
    todo = [r for r in rows.values()
            if r["platform"] == "youtube" and not r["posted_date"]
            and r.get("verdict") in ("keep", "maybe")][:limit]
    if not todo:
        print("nothing to hydrate (run prefilter.py --apply first)")
        return
    print(f"hydrating {len(todo)} rows")
    upd = []
    for r in todo:
        got = run(["--no-warnings", "--skip-download", "--print",
                   SEP.join(["%(upload_date)s", "%(width)s", "%(height)s", "%(fps)s"]),
                   r["url"]], timeout=90)
        if not got:
            continue
        up, w, h, fps = (got[0].split(SEP) + ["", "", "", ""])[:4]
        upd.append({
            "candidate_id": r["candidate_id"],
            "posted_date": (f"{up[:4]}-{up[4:6]}-{up[6:8]}"
                            if up and up != "NA" and len(up) == 8 else ""),
            "native_res": (f"{w}x{h}" if w.isdigit() and h.isdigit() else ""),
            "native_fps": (fps if fps and fps != "NA" else ""),
        })
    added, total = upsert(path, CANDIDATE_FIELDS, "candidate_id", upd)
    print(f"hydrated {len(upd)} rows ({total} in manifest)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--search", action="append", default=[])
    ap.add_argument("--search-default", action="store_true")
    ap.add_argument("--channel", action="append", default=[])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--hydrate", action="store_true")
    ap.add_argument("--limit", type=int, default=100)
    a = ap.parse_args()
    out = Path(a.out)

    if a.hydrate:
        hydrate(out, a.limit)
        return

    queries = list(a.search) + (DEFAULT_QUERIES if a.search_default else [])
    allrows = []
    for q in queries:
        rows = enum_search(q, a.n)
        print(f"  search {q!r}: {len(rows)}")
        allrows += rows
    for c in a.channel:
        rows = enum_channel(c, a.n)
        print(f"  channel {c}: {len(rows)}")
        allrows += rows
        upsert(out / "channels.csv", CHANNEL_FIELDS, "handle", [{
            "handle": c.lstrip("@"), "platform": "youtube",
            "channel_class": "", "preview_available": "yes",
            "n_found": len(rows), "added_by": "seed"}])

    if not allrows:
        print("nothing found")
        return
    added, total = upsert(out / "candidates.csv", CANDIDATE_FIELDS,
                          "candidate_id", allrows)
    print(f"\n{len(allrows)} fetched, {added} new ({total} in manifest)")


if __name__ == "__main__":
    main()
