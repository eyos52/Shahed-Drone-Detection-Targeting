#!/usr/bin/env python3
"""Stage 2 triage -- pull low-res proxies of the ranked candidates and sheet them.

A CSV of URLs is not reviewable. This turns the top of candidates.csv into two
things a human can actually use on M3:

  clips/   self-describing filenames, so the CSV is never needed to know what
           you are looking at: 003_keep_14s_ButusovPlus_34340.mp4
  sheets/  8 frames per clip, one clip per row, captioned. Scan the sheet,
           reject the obvious junk, and only open the videos still in play.

Proxies are Tier-1 scratch in the plan's storage model -- regenerable from the
URL, never uploaded, safe to delete. YouTube is capped at 360p; Telegram
exposes exactly one format (telegram:embed), but those posts are 6-60s so the
full pull is a couple of MB either way.

Resumable: a clip already on disk is skipped, so this can be re-run as the
candidate pool grows.

Dependencies: yt-dlp, imageio-ffmpeg, Pillow, numpy.

Usage:
    python triage.py --out ../../manifests --dest $SCRATCH/triage --limit 20
    python triage.py --out ../../manifests --dest $SCRATCH/triage --verdict keep
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import read_rows

from PIL import Image, ImageDraw

YTDLP = str(Path(sys.executable).parent / "yt-dlp")
SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
CELL = (240, 180)
NFRAMES = 8


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def fetch(row, rank, clips):
    chan = SAFE.sub("-", row.get("channel") or "unknown")[:22]
    vid = SAFE.sub("-", row["url"].rstrip("/").rsplit("/", 1)[-1])[:24]
    dur = row.get("duration_s") or "?"
    stem = f"{rank:03d}_{row['verdict']}_{dur}s_{chan}_{vid}"
    # Resume on the video id, not the whole stem. Rank is a position in the
    # current selection, so it shifts whenever the ruleset or filters change --
    # matching the full stem would re-download everything after any such change.
    # Rename instead, so ranks stay consistent across the set without refetching.
    existing = [p for p in clips.glob(f"*_{vid}.*")]
    if existing:
        p = existing[0]
        if p.stem != stem:
            p = p.rename(p.with_name(stem + p.suffix))
        return p
    # YouTube serves split video/audio, so a progressive "best[height<=360]"
    # matches nothing on most videos. Select by sort order instead and let
    # yt-dlp merge -- which needs ffmpeg, supplied by imageio-ffmpeg since M3
    # has no system one. telegram:embed exposes a single format and ignores all
    # of this.
    # No --no-warnings here. telegram:embed reports "unable to extract video
    # URL" as a WARNING, not an error, so suppressing warnings left stderr
    # empty and the failure unexplained. ~5% of Telegram posts that advertise a
    # video in the channel preview do not expose a fetchable file.
    cmd = [YTDLP, "--no-playlist",
           "-f", "bv*+ba/b", "-S", "res:360",
           "--ffmpeg-location", ffmpeg(),
           "-o", str(clips / f"{stem}.%(ext)s"), row["url"]]
    err = ""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        err = p.stderr or ""
    except subprocess.TimeoutExpired:
        err = "download timed out after 300s"
    hit = list(clips.glob(f"{stem}.*"))
    if not hit:
        # The venv's Python 3.10 deprecation notice is the LAST thing on stderr,
        # so a tail slice reports that instead of the actual failure. Pick the
        # real error line out of the stream.
        real = [l for l in err.splitlines()
                if any(k in l for k in ("ERROR", "WARNING", "Unsupported", "timed out"))]
        sys.stderr.write(f"  FAILED {row['url']}: "
                         f"{(real[0] if real else err.strip()[:140])[:160]}\n")
        return None
    return hit[0]


def duration_of(path):
    """Seconds, read from the container header. None if unreadable.

    Must NOT be `-f null -`: that decodes the entire file just to learn its
    length, which on a 79-minute download blew a 120s timeout and took the
    whole run down with it. Bare `ffmpeg -i FILE` prints the metadata and exits
    non-zero immediately without decoding a single frame.
    """
    try:
        p = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)],
                           capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return None
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", p.stderr or "")
    if not m:
        return None
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def frames(path, out_dir, n=NFRAMES):
    """n frames spread evenly across the whole clip.

    The obvious `thumbnail,fps=1/1` is wrong here: on a 14s clip it returns
    whatever ffmpeg considers representative out of the first few seconds, so
    an engagement in the back half is missed entirely and the clip gets
    rejected on frames that never showed the target. Sampling at n/duration
    guarantees the whole clip is covered.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    got = sorted(out_dir.glob("f*.jpg"))
    if len(got) >= n:
        return got[:n]
    dur = duration_of(path)
    vf = f"fps={n}/{dur:.3f}" if dur and dur > 0.5 else "fps=2"
    # One bad file must never kill the batch -- a 231-item run that dies on
    # item 170 wastes the 60 downloads that came after it.
    try:
        subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(path),
                        "-vf", vf, "-frames:v", str(n), "-q:v", "4",
                        str(out_dir / "f%02d.jpg")],
                       capture_output=True, timeout=240)
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"  frame-extract timed out: {path.name}\n")
    got = sorted(out_dir.glob("f*.jpg"))
    if not got:                      # unreadable or sub-second: take what exists
        try:
            subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(path),
                            "-frames:v", str(n), "-q:v", "4",
                            str(out_dir / "f%02d.jpg")],
                           capture_output=True, timeout=180)
        except subprocess.TimeoutExpired:
            pass
        got = sorted(out_dir.glob("f*.jpg"))
    return got[:n]


def thumb(p):
    try:
        im = Image.open(p).convert("RGB")
        im.thumbnail(CELL)
        c = Image.new("RGB", CELL, (22, 22, 22))
        c.paste(im, ((CELL[0] - im.width) // 2, (CELL[1] - im.height) // 2))
        return c
    except Exception:
        return Image.new("RGB", CELL, (60, 0, 0))


def sheet(entries, out_png, per_sheet_idx, total):
    GAP, CAP, HEAD = 5, 30, 34
    W = NFRAMES * CELL[0] + (NFRAMES + 1) * GAP
    RH = CELL[1] + CAP + GAP
    im = Image.new("RGB", (W, HEAD + RH * len(entries)), (12, 12, 12))
    dr = ImageDraw.Draw(im)
    dr.text((GAP, 9), f"triage sheet {per_sheet_idx}  |  {total} candidates total  "
                      f"|  filename carries rank, verdict, duration, channel",
            fill=(235, 235, 235))
    for i, (row, clip, fs) in enumerate(entries):
        y = HEAD + i * RH
        for k in range(NFRAMES):
            x = GAP + k * (CELL[0] + GAP)
            if k < len(fs):
                im.paste(thumb(fs[k]), (x, y))
            else:
                im.paste(Image.new("RGB", CELL, (18, 18, 18)), (x, y))
        col = (120, 230, 150) if row["verdict"] == "keep" else (235, 205, 120)
        dr.text((GAP, y + CELL[1] + 4), clip.name, fill=col)
        dr.text((GAP, y + CELL[1] + 16), (row.get("title") or "")[:200],
                fill=(165, 165, 165))
    im.save(out_png, quality=84)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="manifests/ directory")
    ap.add_argument("--dest", required=True, help="where proxies and sheets go")
    ap.add_argument("--verdict", default="keep,maybe")
    ap.add_argument("--max-duration", type=int, default=300,
                    help="skip anything longer. Long-form is podcasts and news "
                         "packages: 8 frames cannot represent 79 minutes, and "
                         "one of them is a ~100MB download. Mixed packages with "
                         "real footage inside belong in the scene-split stage, "
                         "not here.")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--per-sheet", type=int, default=5)
    a = ap.parse_args()

    want = set(a.verdict.split(","))

    def short_enough(r):
        d = r.get("duration_s")
        return not (str(d).isdigit() and int(d) > a.max_duration)

    rows = [r for r in read_rows(Path(a.out) / "candidates.csv",
                                 "candidate_id").values()
            if r.get("verdict") in want and short_enough(r)][:a.limit]
    if not rows:
        print("no candidates match -- run prefilter.py --apply first")
        return

    dest = Path(a.dest)
    clips, sheets, fdir = dest / "clips", dest / "sheets", dest / "frames"
    for d in (clips, sheets, fdir):
        d.mkdir(parents=True, exist_ok=True)

    entries, idx = [], []
    for n, r in enumerate(rows, 1):
        print(f"[{n}/{len(rows)}] {r['verdict']:<5} {r['url']}")
        clip = fetch(r, n, clips)
        if not clip:
            continue
        fs = frames(clip, fdir / clip.stem)
        entries.append((r, clip, fs))
        idx.append((clip.name, r.get("duration_s", ""), r["url"],
                    (r.get("title") or "")[:160]))

    for i in range(0, len(entries), a.per_sheet):
        chunk = entries[i:i + a.per_sheet]
        sheet(chunk, sheets / f"triage_{i // a.per_sheet:02d}.jpg",
              i // a.per_sheet, len(entries))

    with open(dest / "index.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["file", "duration_s", "url", "title"])
        w.writerows(idx)

    print(f"\n{len(entries)} clips -> {clips}")
    print(f"{(len(entries) + a.per_sheet - 1) // a.per_sheet} sheets -> {sheets}")
    print(f"index -> {dest / 'index.tsv'}")


if __name__ == "__main__":
    main()
