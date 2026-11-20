#!/usr/bin/env python3
"""Find the seconds that matter inside a clip, so nobody scrubs a 5-minute package.

The unit of work is a SHOT, not a video. A news report becomes a handful of
rows pointing at 02:14-02:26, and the studio material around it is dropped
without anyone watching it. That is also what makes the long-form candidates
usable again: they were excluded from frame triage at 300s precisely because
8 thumbnails cannot represent a mixed package.

Two passes per video:

  1. Shot boundaries via ffmpeg scene detection.
  2. Per shot, is there a SMALL, PERSISTENT, STRAIGHT-MOVING thing against sky?

The discriminating measurement is `moving_frac` -- what fraction of pixels
differ from the shot's temporal median. A cut, a pan or a talking head lights
up most of the frame. A drone at range lights up well under a percent. That one
number removes studio, interview and camera-move shots before anything more
expensive runs.

Everything else is scored, not gated, and written to segments.csv raw so
thresholds can be calibrated against hand-labelled truth rather than guessed:

  persistence   fraction of frames carrying a real peak
  straightness  residual from a line fit through the peak track. A Shahed
                tracks straight and level; birds wander, cuts jump.
  blob_px       size of the moving thing. Small is the regime that matters.
  sky_frac      smoothness of the background plate. Studio shots are full of
                edges, faces and text; sky is not.

The peak track is also a free first-pass box seed for the annotator, which is
the "frame-difference as a finding aid" idea in test-video-dataset.md 6.1.

Known limitation, deliberately not pre-solved: a panning camera moves every
pixel. `moving_frac` catches the gross cases. Global-motion compensation is
only worth adding if the measured false-positive rate says so.

Dependencies: imageio-ffmpeg, numpy. No scipy on M3, so the blur and the line
fit are done by hand.

Usage:
    python segment.py --clips $SCRATCH/triage/clips --out $SCRATCH/segments
    python segment.py --clips ... --out ... --sheets
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

W, H = 320, 180          # analysis size; letterboxed, so aspect is preserved
FPS = 4                  # sample rate inside a shot
MIN_SHOT = 1.5           # shorter than this cannot support a track
PEAK_K = 6.0             # peak must exceed mean + K*std of the diff
MOVE_K = 3.0             # pixel counts as moving above mean + K*std


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_duration(path):
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


def shot_bounds(path, dur, thresh=0.30):
    """-> [(start, end)] from ffmpeg scene detection, merged with the ends."""
    try:
        p = subprocess.run(
            [ffmpeg(), "-hide_banner", "-i", str(path),
             "-vf", f"select='gt(scene,{thresh})',showinfo",
             "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return [(0.0, dur)]
    cuts = sorted({float(t) for t in
                   re.findall(r"pts_time:([\d.]+)", p.stderr or "")})
    marks = [0.0] + [c for c in cuts if 0.2 < c < dur - 0.2] + [dur]
    return [(a, b) for a, b in zip(marks, marks[1:]) if b - a >= MIN_SHOT]


def read_gray(path, start, end):
    """Raw grayscale frames for one shot, letterboxed to a fixed size so the
    byte stream can be reshaped without parsing per-frame headers."""
    vf = (f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,format=gray")
    cmd = [ffmpeg(), "-hide_banner", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(path),
           "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None
    n = len(p.stdout) // (W * H)
    if n < 4:
        return None
    return np.frombuffer(p.stdout[:n * W * H], np.uint8).reshape(n, H, W).astype(np.float32)


def _box1d(a, r, axis):
    L = a.shape[axis]
    pw = [(0, 0)] * a.ndim
    pw[axis] = (r, r)
    cs = np.cumsum(np.pad(a, pw, mode="edge"), axis=axis)
    z = list(cs.shape)
    z[axis] = 1
    cs = np.concatenate([np.zeros(z, cs.dtype), cs], axis=axis)
    hi = [slice(None)] * a.ndim
    lo = [slice(None)] * a.ndim
    hi[axis] = slice(2 * r + 1, 2 * r + 1 + L)
    lo[axis] = slice(0, L)
    return (cs[tuple(hi)] - cs[tuple(lo)]) / (2 * r + 1)


def boxblur(a, r):
    """Separable box blur via integral images. numpy only -- no scipy on M3."""
    return _box1d(_box1d(a, r, 1), r, 2)


def saliency(frames, r_small=1, r_large=8):
    """Band-pass response: small object against a smooth background.

    This replaces temporal differencing as the primary signal, because most of
    this footage is CAMERA-TRACKED -- somebody following the drone with a phone
    or a gimbal. The target then barely moves in frame, so a temporal median
    absorbs it into the background and the detector goes blind exactly on the
    clearest positives. Measured: a clip with a Shahed visible in every sampled
    thumbnail scored moving_frac 0.0005 and was rejected outright.

    A difference of two box blurs keeps structures a few pixels across and
    suppresses both sky gradients and large objects, regardless of whether the
    camera or the target is the thing that moves.
    """
    return boxblur(frames, r_small) - boxblur(frames, r_large)


def line_residual(t, v):
    """RMS residual of a least-squares line fit, in pixels."""
    if len(t) < 3:
        return 0.0
    A = np.stack([t, np.ones_like(t)], axis=1)
    sol, *_ = np.linalg.lstsq(A, v, rcond=None)
    return float(np.sqrt(np.mean((A @ sol - v) ** 2)))


def sky_fraction(bg):
    """Fraction of the background plate that is smooth. Sky has almost no
    gradient; studio sets, chyrons and faces are full of edges."""
    gy = np.abs(np.diff(bg, axis=0))[:, :-1]
    gx = np.abs(np.diff(bg, axis=1))[:-1, :]
    g = gx + gy
    return float((g < 4.0).mean())


def score_shot(frames):
    bg = np.median(frames, axis=0)

    # kept only as a camera-motion / cut indicator, no longer the detector
    diff = np.abs(frames - bg)
    dmu, dsd = float(diff.mean()), float(diff.std()) + 1e-6
    moving = (diff > (dmu + MOVE_K * dsd)).mean(axis=(1, 2))

    sal = np.abs(saliency(frames))
    # Suppress pixel-locked overlays. A burnt-in watermark or channel logo is a
    # bright isolated peak at a fixed coordinate, so it wins argmax on every
    # frame and hides the real target behind it -- measured on a clip whose
    # Shahed was visible in every thumbnail but never became the peak. The
    # temporal median of the saliency map is exactly that static component;
    # anything that shifts by even a couple of pixels survives subtraction.
    sal = np.maximum(sal - np.median(sal, axis=0), 0.0)
    flat = sal.reshape(len(sal), -1)
    idx = flat.argmax(axis=1)
    peak = flat.max(axis=1)
    ys, xs = np.divmod(idx, W)

    # Isolation separates "one small thing against sky" from "a scene full of
    # edges". A lone drone towers over the rest of the response; a studio set,
    # a treeline or a chyron produces thousands of comparable peaks.
    bulk = np.percentile(flat, 99.5, axis=1) + 1e-6
    isolation = peak / bulk
    good = isolation > 2.5

    persistence = float(good.mean())
    jump = straightness = blob_px = drift = 0.0
    if good.sum() >= 3:
        gi = np.flatnonzero(good)
        gx, gy = xs[gi].astype(np.float64), ys[gi].astype(np.float64)
        # Continuity: a real target drifts a few pixels per sample. Noise
        # teleports across the frame, which is what produced drift of 100-270px
        # with straightness ~0.1 in the first version.
        if len(gi) >= 2:
            step = np.hypot(np.diff(gx), np.diff(gy))
            jump = float(np.median(step))
        t = gi.astype(np.float64)
        rx, ry = line_residual(t, gx), line_residual(t, gy)
        straightness = float(1.0 / (1.0 + (rx + ry) / 4.0))
        thr = sal.mean() + 3.0 * sal.std()
        blobs = []
        for i in gi:
            y0, y1 = max(0, ys[i] - 10), min(H, ys[i] + 11)
            x0, x1 = max(0, xs[i] - 10), min(W, xs[i] + 11)
            blobs.append(int((sal[i, y0:y1, x0:x1] > thr).sum()))
        blob_px = float(np.median(blobs))
        drift = float(np.hypot(np.ptp(gx), np.ptp(gy)))

    return {
        "moving_frac": round(float(np.median(moving)), 5),
        "persistence": round(persistence, 3),
        "isolation": round(float(np.median(isolation)), 2),
        "jump_px": round(jump, 1),
        "straightness": round(straightness, 3),
        "blob_px": round(blob_px, 1),
        "drift_px": round(drift, 1),
        "sky_frac": round(sky_fraction(bg), 3),
    }


def combine(m):
    """Gates first, then a product of the surviving signals.

    Deliberately NOT gated on moving_frac any more: camera-tracked footage has
    a near-static target and a moving background, and gating on motion threw
    away the clearest positives.
    """
    if m["persistence"] < 0.30:
        return 0.0, "no_isolated_target"
    if m["jump_px"] > 25:
        return 0.0, "track_discontinuous"     # peak teleporting = noise
    # A perfectly immobile peak is a channel logo or a burnt-in watermark, not
    # a drone. Measured: a clip scored top of the list on isolation 4.0 with
    # drift 0.0 and straightness 1.00 -- a textbook static overlay. Even
    # camera-tracked targets jitter by a couple of pixels.
    if m["drift_px"] < 2:
        return 0.0, "static_peak_overlay"
    if m["blob_px"] > 300:
        return 0.0, "blob_too_large"
    if m["sky_frac"] < 0.25:
        return 0.0, "cluttered_background"
    size = 1.0 if 1 <= m["blob_px"] <= 200 else 0.4
    iso = min(m["isolation"] / 5.0, 1.0)
    return round(m["persistence"] * iso * size *
                 max(m["straightness"], 0.25), 4), ""


FIELDS = ["video", "seg", "start_s", "end_s", "dur_s", "score", "reject",
          "persistence", "isolation", "jump_px", "straightness", "blob_px",
          "drift_px", "sky_frac", "moving_frac", "cut_cmd"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, help="directory of videos")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--scene-thresh", type=float, default=0.30)
    ap.add_argument("--window", type=float, default=6.0,
                    help="seconds per scoring window inside a shot")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    clips = sorted(p for p in Path(a.clips).iterdir()
                   if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov"})
    if a.limit:
        clips = clips[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for n, c in enumerate(clips, 1):
        dur = probe_duration(c)
        if not dur:
            print(f"[{n}/{len(clips)}] {c.name}: unreadable")
            continue
        shots = shot_bounds(c, dur, a.scene_thresh)
        kept = 0
        for si, (s, e) in enumerate(shots):
            fr = read_gray(c, s, min(e, s + 120))     # bound memory on long shots
            if fr is None:
                continue
            # Score in short windows, not whole shots. A target may be present
            # for 10s of a 47s shot; averaging persistence over the whole shot
            # buries it (measured: isolation 1.6 on a shot with a clearly
            # visible Shahed). Windows also happen to be the length we want to
            # cut anyway.
            win = int(a.window * FPS)
            step = max(1, win // 2)
            spans = ([(0, len(fr))] if len(fr) <= win
                     else [(i, i + win) for i in range(0, len(fr) - win + 1, step)])
            for wi, (f0, f1) in enumerate(spans):
                m = score_shot(fr[f0:f1])
                sc, rej = combine(m)
                kept += sc > 0
                ws, we = s + f0 / FPS, s + f1 / FPS
                rows.append({
                    "video": c.name, "seg": f"{si}.{wi}",
                    "start_s": round(ws, 2), "end_s": round(we, 2),
                    "dur_s": round(we - ws, 2), "score": sc, "reject": rej,
                    **m,
                    "cut_cmd": f'-ss {max(0, ws - 0.5):.2f} -to {min(dur, we + 0.5):.2f}',
                })
        print(f"[{n}/{len(clips)}] {c.name[:52]}  {len(shots)} shots, {kept} windows scored")

    rows.sort(key=lambda r: -r["score"])
    with open(out / "segments.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    live = [r for r in rows if r["score"] > 0]
    print(f"\n{len(rows)} shots from {len(clips)} videos -> {len(live)} with a target signal")
    print(f"wrote {out / 'segments.csv'}")
    if live:
        print(f"\n{'score':>7}  {'secs':>13}  video")
        for r in live[:15]:
            print(f"{r['score']:>7.3f}  {r['start_s']:>6.1f}-{r['end_s']:<6.1f} {r['video'][:58]}")


if __name__ == "__main__":
    main()
