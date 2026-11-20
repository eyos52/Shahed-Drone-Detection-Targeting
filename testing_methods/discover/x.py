#!/usr/bin/env python3
"""Resolve X/Twitter status URLs to metadata. NOT a discovery surface.

Measured on M3, 2026-10-29:

  * A single status extracts fine with no credentials -- id, duration,
    upload_date, uploader and the full post text all come back.
  * Timeline enumeration does NOT. /media and /with_replies redirect to
    x.com/i/jf/onboarding/web?redirect_after_login=... and yt-dlp reports
    "Unsupported URL".

So X cannot grow the candidate pool on its own; it can only tell us about URLs
something else already found. That makes it useful for exactly three jobs:

  1. Hydrating the URLs already sitting in data/x_video_urls.txt.
  2. Resolving the 13 evaluation videos in result_videos/, whose provenance is
     undocumented -- none of their status ids appear in x_video_urls.txt, so
     nothing in the repo records who posted them or when. Their dates matter
     for the leakage question.
  3. Snowball: when a Telegram post links out to an X status, resolve it here.

If timeline enumeration is ever needed, it requires passing authenticated
cookies, which is a credentials decision for a human to make, not something
this script should do quietly.

Dependencies: yt-dlp on PATH or in the same venv.

Usage:
    python x.py --out ../../manifests --urls-from ../../../data/x_video_urls.txt
    python x.py --out ../../manifests --resolve-eval-videos /path/to/result_videos
    python x.py --out ../../manifests --url https://x.com/user/status/123
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import CANDIDATE_FIELDS, blank_candidate, candidate_id, upsert

YTDLP = str(Path(sys.executable).parent / "yt-dlp")
SEP = "\x1f"
FMT = SEP.join(["%(id)s", "%(uploader_id)s", "%(upload_date)s", "%(duration)s",
                "%(width)s", "%(height)s", "%(fps)s", "%(title)s"])
STATUS = re.compile(r"(?:x|twitter)\.com/(?:i/)?(?:\w+/)?status(?:es)?/(\d+)")


def resolve(url, timeout=90):
    try:
        p = subprocess.run(
            [YTDLP, "--no-warnings", "--skip-download", "--print", FMT, url],
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    line = next((l for l in p.stdout.splitlines() if SEP in l), None)
    if not line:
        return None
    f = (line.split(SEP) + [""] * 8)[:8]
    vid, up, date, dur, w, h, fps, title = f
    chan = up if up and up != "NA" else ""
    return blank_candidate(
        candidate_id=candidate_id("x", chan, vid),
        platform="x", channel=chan, url=url,
        posted_date=(f"{date[:4]}-{date[4:6]}-{date[6:8]}"
                     if date and date != "NA" and len(date) == 8 else ""),
        title=(title if title != "NA" else "")[:300],
        duration_s=(str(int(float(dur))) if dur.replace(".", "").isdigit() else ""),
        native_res=(f"{w}x{h}" if w.isdigit() and h.isdigit() else ""),
        native_fps=(fps if fps and fps != "NA" else ""),
    )


def urls_from_file(p):
    return [m.group(0) if m.group(0).startswith("http") else f"https://{m.group(0)}"
            for line in Path(p).read_text().splitlines()
            for m in [re.search(r"(?:https?://)?(?:x|twitter)\.com/\S+", line)] if m]


def eval_video_urls(result_dir):
    """The 13 annotated eval renders are named <status_id>.mp4 and nothing in
    the repo says where they came from. Rebuild the URL from the filename."""
    ids = sorted({p.stem for p in Path(result_dir).rglob("*.mp4")
                  if p.stem.isdigit() and len(p.stem) >= 15})
    return [f"https://x.com/i/status/{i}" for i in ids]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", action="append", default=[])
    ap.add_argument("--urls-from")
    ap.add_argument("--resolve-eval-videos", metavar="RESULT_VIDEOS_DIR")
    a = ap.parse_args()

    urls = list(a.url)
    if a.urls_from:
        urls += urls_from_file(a.urls_from)
    if a.resolve_eval_videos:
        urls += eval_video_urls(a.resolve_eval_videos)
    # de-dup on status id: the same clip is often listed under several handles
    seen, uniq = set(), []
    for u in urls:
        m = STATUS.search(u)
        k = m.group(1) if m else u
        if k not in seen:
            seen.add(k)
            uniq.append(u)
    if not uniq:
        ap.error("no URLs given")

    rows, dead = [], []
    for u in uniq:
        r = resolve(u)
        if r:
            rows.append(r)
            print(f"  ok   {r['posted_date']}  {r['duration_s']:>4}s  "
                  f"@{r['channel']:<18}{r['title'][:60]}")
        else:
            dead.append(u)
            print(f"  DEAD {u}")
    if rows:
        added, total = upsert(Path(a.out) / "candidates.csv", CANDIDATE_FIELDS,
                              "candidate_id", rows)
        print(f"\nresolved {len(rows)}/{len(uniq)}, {added} new ({total} in manifest)")
    if dead:
        print(f"unresolvable ({len(dead)}) -- deleted, protected, or geo-blocked:")
        for u in dead:
            print(f"   {u}")


if __name__ == "__main__":
    main()
