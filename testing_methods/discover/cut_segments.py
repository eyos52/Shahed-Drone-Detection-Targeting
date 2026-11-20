#!/usr/bin/env python3
"""Cut the flagged windows out of segments.csv into watchable files.

segments.csv is timestamps, not media -- nothing is extracted until this runs.
Filenames lead with the score so the directory sorts worst-last and you can
stop watching when the quality drops off.

Stream-copies by default: no re-encode, so nothing is lost and it is near
instant. Cuts land on the nearest keyframe, which is why the window is padded;
the precise trim happens later in the single encode that the harvest stage
does, per the "encode exactly once" rule in scraping-plan.md.

Dependencies: imageio-ffmpeg.

Usage:
    python cut_segments.py --segments $SCRATCH/segments/segments.csv \\
        --clips $SCRATCH/triage/clips --out $SCRATCH/segments/cuts \\
        --min-score 0.30 --best-per-video
"""
import argparse
import csv
import subprocess
from pathlib import Path


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-score", type=float, default=0.30)
    ap.add_argument("--best-per-video", action="store_true",
                    help="only the single strongest window from each video")
    ap.add_argument("--pad", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.segments))
            if float(r["score"]) >= a.min_score]
    if a.best_per_video:
        best = {}
        for r in rows:
            v = r["video"]
            if v not in best or float(r["score"]) > float(best[v]["score"]):
                best[v] = r
        rows = list(best.values())
    rows.sort(key=lambda r: -float(r["score"]))
    if a.limit:
        rows = rows[:a.limit]

    clips, out = Path(a.clips), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx, made = [], 0
    for r in rows:
        src = clips / r["video"]
        if not src.exists():
            continue
        s = max(0.0, float(r["start_s"]) - a.pad)
        e = float(r["end_s"]) + a.pad
        name = (f"{float(r['score']):.3f}_{Path(r['video']).stem[:46]}"
                f"_{s:.0f}-{e:.0f}s{src.suffix}")
        dst = out / name
        if dst.exists():
            idx.append((name, r)); made += 1; continue
        p = subprocess.run(
            [ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
             "-ss", f"{s:.2f}", "-to", f"{e:.2f}", "-i", str(src),
             "-c", "copy", str(dst)], capture_output=True, text=True, timeout=180)
        if not dst.exists() or dst.stat().st_size < 2048:
            # stream copy can produce an empty file when the window starts
            # between keyframes; fall back to a single re-encode for that cut
            dst.unlink(missing_ok=True)
            subprocess.run(
                [ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                 "-ss", f"{s:.2f}", "-to", f"{e:.2f}", "-i", str(src),
                 "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                 "-an", str(dst)], capture_output=True, timeout=300)
        if dst.exists():
            idx.append((name, r)); made += 1
        else:
            print(f"  FAILED {r['video']} @ {s:.1f}")

    with open(out / "index.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["file", "score", "source_video", "start_s", "end_s",
                    "persistence", "isolation", "blob_px", "sky_frac"])
        for name, r in idx:
            w.writerow([name, r["score"], r["video"], r["start_s"], r["end_s"],
                        r["persistence"], r["isolation"], r["blob_px"],
                        r["sky_frac"]])

    total = sum(float(r["end_s"]) - float(r["start_s"]) + 2 * a.pad for _, r in idx)
    print(f"{made} cuts -> {out}")
    print(f"{total/60:.1f} minutes total")
    print(f"index -> {out/'index.tsv'}")


if __name__ == "__main__":
    main()
