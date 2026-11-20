#!/usr/bin/env python3
"""Enumerate Ukrainian defence outlets and pull the videos their articles embed.

Measured availability, 2026-10-30 -- most of these outlets do not want to be
read programmatically, so the adapter table records what each one actually
serves rather than what it ought to:

    armyinform.com.ua/feed/        RSS, works, ~10 items per pull
    mil.in.ua/feed/                200 but serves HTML, not a feed
    mil.in.ua/wp-json/wp/v2/posts  endpoint answers, exposes 2 posts, search 0
    defence-ua.com/rss.xml         200, empty body
    united24media.com/rss          serves HTML
    ukrinform.net, suspilne.media  404

So this leg is thin by nature. It is also largely redundant: Militarnyi's
YouTube channel enumerates fine through youtube.py even though its own site
does not, and these outlets mostly republish footage that reaches us through
the Telegram and YouTube legs anyway. Build expectations accordingly.

What it does: enumerate articles, score the headline with the same prefilter
used everywhere else, then fetch only the surviving articles and extract the
media they embed. An article is not itself a candidate -- the embedded video
is, which is why the row that lands in candidates.csv carries the YouTube or
Telegram URL with the outlet recorded as the referrer.

Dependencies: stdlib only.

Usage:
    python news.py --out ../../manifests [--outlet armyinform] [--pages 3]
"""
import argparse
import gzip
import html
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import CANDIDATE_FIELDS, blank_candidate, candidate_id, upsert
from prefilter import score

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

OUTLETS = {
    "armyinform": {"type": "rss", "url": "https://armyinform.com.ua/feed/"},
    # Kept despite returning almost nothing: if the site ever opens up, the
    # adapter is already here and the failure is recorded rather than retried
    # from scratch every time someone wonders why Militarnyi is missing.
    "militarnyi": {"type": "wp", "url": "https://mil.in.ua/wp-json/wp/v2/posts"},
    "defence_ua": {"type": "rss", "url": "https://defence-ua.com/rss.xml"},
}

# Media an article can embed. Order matters: the first match on a page wins.
EMBED = [
    (re.compile(r'youtube\.com/embed/([A-Za-z0-9_-]{11})'), "youtube",
     "https://www.youtube.com/watch?v={}"),
    (re.compile(r'youtube\.com/watch\?v=([A-Za-z0-9_-]{11})'), "youtube",
     "https://www.youtube.com/watch?v={}"),
    (re.compile(r'youtu\.be/([A-Za-z0-9_-]{11})'), "youtube",
     "https://www.youtube.com/watch?v={}"),
    (re.compile(r'(?:t\.me|telegram\.me)/([A-Za-z0-9_]+/\d+)'), "telegram",
     "https://t.me/{}"),
    (re.compile(r'(?:x|twitter)\.com/\w+/status/(\d+)'), "x",
     "https://x.com/i/status/{}"),
]
_TAG = re.compile(r"<[^>]+>")


def get(url, timeout=25):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept-Encoding": "gzip", "Accept-Language": "uk,en"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
        return raw.decode("utf-8", "replace")


def clean(s):
    return " ".join(html.unescape(_TAG.sub(" ", s or "")).split())


def from_rss(url, pages):
    """-> [(title, link, date)]. Paginates WordPress-style ?paged=N."""
    out = []
    for p in range(1, pages + 1):
        u = url if p == 1 else f"{url}?paged={p}"
        try:
            body = get(u)
        except Exception as e:                                # noqa: BLE001
            print(f"    page {p}: {type(e).__name__}")
            break
        items = re.findall(r"<item>(.*?)</item>", body, re.S)
        if not items:
            break
        for it in items:
            def g(tag):
                m = re.search(rf"<{tag}[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</{tag}>",
                              it, re.S)
                return clean(m.group(1)) if m else ""
            out.append((g("title"), g("link"), g("pubDate")[:16]))
        time.sleep(0.6)
    return out


def from_wp(url, pages):
    out = []
    for p in range(1, pages + 1):
        try:
            body = get(f"{url}?per_page=50&page={p}&_fields=date,link,title")
            data = json.loads(body)
        except Exception as e:                                # noqa: BLE001
            print(f"    page {p}: {type(e).__name__}")
            break
        if not isinstance(data, list) or not data:
            break
        for d in data:
            # mil.in.ua answers the endpoint but not in the documented shape --
            # entries come back as bare lists, not post objects. Skip anything
            # that is not a dict rather than crashing the whole outlet sweep.
            if not isinstance(d, dict):
                continue
            t = d.get("title")
            t = t.get("rendered") if isinstance(t, dict) else t
            out.append((clean(t), d.get("link", ""), (d.get("date") or "")[:10]))
        time.sleep(0.6)
    return out


def embeds_in(article_url):
    try:
        body = get(article_url)
    except Exception:                                         # noqa: BLE001
        return []
    found = []
    for rx, platform, tmpl in EMBED:
        for m in rx.findall(body):
            found.append((platform, tmpl.format(m)))
    # de-dup, keep order
    seen, uniq = set(), []
    for p, u in found:
        if u not in seen:
            seen.add(u)
            uniq.append((p, u))
    return uniq[:3]       # an article rarely embeds more than a couple


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outlet", action="append", default=[],
                    help=f"one of {sorted(OUTLETS)}; default all")
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument("--max-articles", type=int, default=40)
    a = ap.parse_args()

    names = a.outlet or sorted(OUTLETS)
    rows, stats = [], {}
    for name in names:
        cfg = OUTLETS.get(name)
        if not cfg:
            print(f"unknown outlet {name}")
            continue
        print(f"[{name}]")
        arts = (from_rss if cfg["type"] == "rss" else from_wp)(cfg["url"], a.pages)
        print(f"  {len(arts)} articles")
        # score the headline before spending a request on the article body
        hits = []
        for title, link, date in arts:
            t, i, neg = score(title)
            if t or i:
                hits.append((title, link, date, 3 * t + 2 * i))
        hits.sort(key=lambda h: -h[3])
        hits = hits[:a.max_articles]
        print(f"  {len(hits)} match target/intercept terms")
        n_emb = 0
        for title, link, date, _ in hits:
            for platform, murl in embeds_in(link):
                n_emb += 1
                vid = murl.rstrip("/").rsplit("/", 1)[-1]
                rows.append(blank_candidate(
                    candidate_id=candidate_id(platform, name, vid),
                    platform=platform, channel=name, url=murl,
                    posted_date=date[:10], title=title[:300]))
            time.sleep(0.5)
        print(f"  {n_emb} embedded videos")
        stats[name] = (len(arts), len(hits), n_emb)

    if not rows:
        print("\nno embedded videos found")
        return
    added, total = upsert(Path(a.out) / "candidates.csv", CANDIDATE_FIELDS,
                          "candidate_id", rows)
    print(f"\n{len(rows)} embeds, {added} new ({total} in manifest)")
    for n, (arts, hits, emb) in stats.items():
        print(f"   {n:<12}{arts:>4} articles -> {hits:>3} on-topic -> {emb:>3} videos")


if __name__ == "__main__":
    main()
