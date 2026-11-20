#!/usr/bin/env python3
"""Enumerate a public Telegram channel's whole post history -- metadata only.

Telegram is where this footage originates and Google barely indexes it, so this
is the highest-yield discovery surface. Public channels render server-side at
t.me/s/<channel> as plain HTML: no account, no API, no token.

Two things verified on M3 before this was written, both load-bearing:

  * The preview is OPT-IN per channel. `war_monitor` and `ukrainenowenglish`
    render 12-14 messages per page; `militarnyi`, `DefenceU` and
    `Pravda_Gerashchenko_Ua` 302 away to the non-preview page with zero
    messages. So coverage is channel-by-channel and the registry has to record
    which channels are reachable -- see preview_available in channels.csv.
  * `?before=<msg_id>` walks backwards reliably (20986 -> 20966 ...), which is
    what makes a full-history sweep possible and therefore lag-immune.

At 12 messages a page a 20k-message channel is ~1700 requests, so this rate
limits by default and is resumable: pass --stop-at-id with the channel's
last_seen_id to fetch only what is new.

Dependencies: stdlib only.

Usage:
    python telegram.py --channel ukrainenowenglish --out ../../manifests \\
        [--max-pages 50] [--stop-at-id 20966] [--delay 1.5]
"""
import argparse
import gzip
import html
import io
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import (CANDIDATE_FIELDS, CHANNEL_FIELDS, blank_candidate,
                      candidate_id, upsert)

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# One message block starts at the wrapper div; splitting on it is stable enough
# and avoids a parser dependency for what is a very regular template.
_WRAP = re.compile(r'tgme_widget_message_wrap')
_POST = re.compile(r'data-post="([^"]+)"')
_TIME = re.compile(r'datetime="([^"]+)"')
_DUR = re.compile(r'message_video_duration[^>]*>([\d:]+)<')
_VIDEO = re.compile(r'tgme_widget_message_video')
_TEXT = re.compile(r'tgme_widget_message_text[^>]*>(.*?)</div>', re.S)
_TAG = re.compile(r'<[^>]+>')


def fetch(url, timeout=30):
    """-> (body, final_url). final_url differs from url when the channel has no
    public preview, which is how we detect that case."""
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept-Encoding": "gzip", "Accept-Language": "en,uk,ru"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
        return raw.decode("utf-8", "replace"), r.geturl()


def dur_to_s(t):
    if not t:
        return ""
    p = [int(x) for x in t.split(":")]
    while len(p) < 3:
        p.insert(0, 0)
    return p[0] * 3600 + p[1] * 60 + p[2]


def parse_page(body, channel):
    """-> (rows, oldest_id). rows are only messages that carry a video."""
    rows, ids = [], []
    blocks = _WRAP.split(body)[1:]
    for b in blocks:
        m = _POST.search(b)
        if not m:
            continue
        post = m.group(1)                      # "channel/12345"
        mid = post.rsplit("/", 1)[-1]
        if mid.isdigit():
            ids.append(int(mid))
        if not _VIDEO.search(b):
            continue                            # photos and text are not candidates
        t = _TIME.search(b)
        txt = _TEXT.search(b)
        title = ""
        if txt:
            title = html.unescape(_TAG.sub(" ", txt.group(1)))
            title = " ".join(title.split())[:300]
        rows.append(blank_candidate(
            candidate_id=candidate_id("telegram", channel, mid),
            platform="telegram", channel=channel,
            url=f"https://t.me/{post}",
            posted_date=(t.group(1)[:10] if t else ""),
            title=title,
            duration_s=dur_to_s(_DUR.search(b).group(1) if _DUR.search(b) else ""),
        ))
    return rows, (min(ids) if ids else None)


def enumerate_channel(channel, max_pages=50, stop_at_id=None, delay=1.5, log=print):
    """Walk backwards from the newest post. Returns (rows, status, last_seen)."""
    url = f"https://t.me/s/{channel}"
    all_rows, newest, before = [], None, None
    for page in range(max_pages):
        u = url if before is None else f"{url}?before={before}"
        try:
            body, final = fetch(u)
        except urllib.error.HTTPError as e:
            return all_rows, f"http_{e.code}", newest
        except Exception as e:                       # noqa: BLE001 - network is messy
            return all_rows, f"error:{type(e).__name__}", newest
        if "/s/" not in final:
            return all_rows, "no_public_preview", newest
        rows, oldest = parse_page(body, channel)
        if oldest is None:
            return all_rows, ("no_public_preview" if page == 0 else "end"), newest
        if newest is None:
            newest = oldest
            m = re.findall(r'data-post="[^/]+/(\d+)"', body)
            if m:
                newest = max(int(x) for x in m)
        all_rows.extend(rows)
        log(f"  page {page + 1}: {len(rows)} video posts, oldest id {oldest} "
            f"(running total {len(all_rows)})")
        if stop_at_id and oldest <= int(stop_at_id):
            return all_rows, "caught_up", newest
        if before is not None and oldest >= before:
            return all_rows, "end", newest          # pagination stopped advancing
        before = oldest
        time.sleep(delay)
    return all_rows, "max_pages", newest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", required=True, action="append",
                    help="channel handle, repeatable")
    ap.add_argument("--out", required=True, help="manifests/ directory")
    ap.add_argument("--max-pages", type=int, default=50)
    ap.add_argument("--stop-at-id", default=None)
    ap.add_argument("--delay", type=float, default=1.5)
    a = ap.parse_args()

    out = Path(a.out)
    for ch in a.channel:
        print(f"[telegram] {ch}")
        rows, status, newest = enumerate_channel(
            ch, a.max_pages, a.stop_at_id, a.delay)
        added, total = upsert(out / "candidates.csv", CANDIDATE_FIELDS,
                              "candidate_id", rows)
        upsert(out / "channels.csv", CHANNEL_FIELDS, "handle", [{
            "handle": ch, "platform": "telegram", "channel_class": "",
            "preview_available": "no" if status == "no_public_preview" else "yes",
            "enumerated_through": time.strftime("%Y-%m-%d"),
            "last_seen_id": newest or "", "n_found": len(rows),
            "added_by": "seed", "notes": status,
        }])
        print(f"  -> {status}: {len(rows)} video posts, {added} new "
              f"({total} in manifest)")


if __name__ == "__main__":
    main()
