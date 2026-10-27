#!/usr/bin/env python3
"""Stage C -- per-image features for separating real photographs from renders
and composites, and for judging whether a real image is actually usable.

Two independent axes, because they are genuinely different questions:

  IS IT REAL?      a CAD render, a procedurally-generated sky, or a dark delta
                   pasted onto a stock photo, versus a real camera frame.
  IS IT USEFUL?    a real close-up of an airframe held against a white wall is
                   real and still worthless for a detector that must find a
                   20-pixel target in the sky. Box size and position capture
                   this, and they are reported separately so neither axis
                   silently overrides the other.

Nothing here decides anything. It emits numbers; a classifier fitted on
hand-labelled sheets does the deciding, so the thresholds are learned from
Andy's judgement rather than guessed here.

Features, and what each one is listening for:

  bnd_ratio      gradient magnitude on the box boundary ring over gradient in
                 the background. A pasted cutout has a harder edge than the
                 scene around it; a real telephoto target is softened by
                 atmosphere and motion.
  bg_edge_den    fraction of background pixels with real gradient. Plain
                 gradient sky (render or studio wall) is near zero; a real
                 scene has horizon, cloud structure, clutter.
  noise_ratio    high-frequency residual variance inside the box over outside.
                 A spliced region carries different sensor noise than its host.
  col_dist       colour distance between box interior and a surrounding
                 annulus. Composites often mismatch white balance.
  blockiness     8x8 JPEG blocking energy. Renders saved as PNG have none;
                 camera/video frames almost always do. Weak but cheap, and it
                 survives Roboflow's re-encode better than full DCT forensics.
  sat_mean/std   renders tend to flat or oversaturated colour.
  lap_var        global sharpness.
  sqrt_in / cx / cy / ar     the usefulness axis.

Dependencies: Pillow + numpy only.

Usage (array-friendly):
    python score_images.py --pool POOL --cluster CLUSTER_DIR --out OUT.csv \\
        [--input-size 640] [--shard 0 --of 8]
"""
import argparse
import csv
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import parse_file

import numpy as np
from PIL import Image

MAXSIDE = 512          # analysis resolution; keeps cost flat across a mixed pool


def grad_mag(g):
    gy = np.zeros_like(g); gx = np.zeros_like(g)
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    return np.abs(gx) + np.abs(gy)


def highfreq(g):
    """Residual after a cheap 3x3 box blur -- stands in for sensor noise."""
    k = np.zeros_like(g)
    k[1:-1, 1:-1] = (g[:-2, :-2] + g[:-2, 1:-1] + g[:-2, 2:] +
                     g[1:-1, :-2] + g[1:-1, 1:-1] + g[1:-1, 2:] +
                     g[2:, :-2] + g[2:, 1:-1] + g[2:, 2:]) / 9.0
    return g - k


def blockiness(g):
    """Energy at 8-pixel boundaries relative to energy just off them. JPEG
    compression leaves a grid here; a PNG render does not."""
    h, w = g.shape
    if h < 24 or w < 24:
        return 0.0
    cols = np.arange(8, w - 1, 8)
    rows = np.arange(8, h - 1, 8)
    if cols.size < 2 or rows.size < 2:
        return 0.0
    on = (np.abs(g[:, cols] - g[:, cols - 1]).mean()
          + np.abs(g[rows, :] - g[rows - 1, :]).mean())
    off_c = np.clip(cols + 3, 0, w - 1)
    off_r = np.clip(rows + 3, 0, h - 1)
    off = (np.abs(g[:, off_c] - g[:, off_c - 1]).mean()
           + np.abs(g[off_r, :] - g[off_r - 1, :]).mean())
    return float(on / (off + 1e-6))


def boxes_of(label_path):
    """(cx, cy, w, h) per annotation. Goes through labels.parse_file so that
    the ten polygon-exporting datasets are measured on their true extent --
    slicing p[1:5] off a polygon gives a sliver and corrupts every size
    statistic downstream, including the review size band."""
    if label_path is None or not label_path.exists():
        return []
    return [(cx, cy, w, h) for _c, cx, cy, w, h in parse_file(label_path)]


def label_for(ip):
    parts = list(ip.parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        lp = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
        if lp.exists():
            return lp
    alt = ip.with_suffix(".txt")
    return alt if alt.exists() else None


def score(ip, input_size):
    with Image.open(ip) as im:
        im = im.convert("RGB")
        W0, H0 = im.size
        im.thumbnail((MAXSIDE, MAXSIDE), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32)
    H, W = a.shape[:2]
    g = a.mean(axis=2)
    gm = grad_mag(g)
    hf = highfreq(g)
    mx = a.max(axis=2); mn = a.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0.0)

    boxes = boxes_of(label_for(ip))
    mask = np.zeros((H, W), dtype=bool)
    ring = np.zeros((H, W), dtype=bool)
    biggest = None
    for cx, cy, bw, bh in boxes:
        x1 = max(int((cx - bw / 2) * W), 0); x2 = min(int((cx + bw / 2) * W), W)
        y1 = max(int((cy - bh / 2) * H), 0); y2 = min(int((cy + bh / 2) * H), H)
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        mask[y1:y2, x1:x2] = True
        pad = max(2, int(0.12 * max(x2 - x1, y2 - y1)))
        rx1, rx2 = max(x1 - pad, 0), min(x2 + pad, W)
        ry1, ry2 = max(y1 - pad, 0), min(y2 + pad, H)
        ring[ry1:ry2, rx1:rx2] = True
        if biggest is None or (x2 - x1) * (y2 - y1) > biggest[4]:
            biggest = (cx, cy, bw, bh, (x2 - x1) * (y2 - y1))
    ring &= ~mask
    bg = ~ring & ~mask

    def m(arr, sel):
        return float(arr[sel].mean()) if sel.any() else float("nan")

    bg_g = m(gm, bg)
    rec = {
        "bnd_ratio":   (m(gm, ring) / (bg_g + 1e-6)) if ring.any() else float("nan"),
        "bg_edge_den": float((gm[bg] > 8).mean()) if bg.any() else float("nan"),
        "noise_ratio": ((float(hf[mask].var()) + 1e-6) / (float(hf[bg].var()) + 1e-6))
                       if mask.any() and bg.any() else float("nan"),
        "col_dist":    (float(np.linalg.norm(a[mask].mean(0) - a[ring].mean(0)))
                        if mask.any() and ring.any() else float("nan")),
        "blockiness":  blockiness(g),
        "sat_mean":    float(sat.mean()),
        "sat_std":     float(sat.std()),
        "lap_var":     float(gm.var()),
        "n_boxes":     len(boxes),
        "img_w":       W0,
        "img_h":       H0,
    }
    if biggest:
        cx, cy, bw, bh, _ = biggest
        scale = input_size / max(W0, H0)
        rec.update(sqrt_in=round(math.sqrt(bw * W0 * bh * H0) * scale, 2),
                   cx=round(cx, 4), cy=round(cy, 4),
                   ar=round((bw * W0) / (bh * H0 + 1e-6), 3))
    else:
        rec.update(sqrt_in=float("nan"), cx=float("nan"),
                   cy=float("nan"), ar=float("nan"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--cluster", required=True,
                    help="dir holding images.csv from cluster_images.py; its rows "
                         "are already one-per-stem, so augmented copies are skipped")
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-size", type=int, default=640)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    args = ap.parse_args()

    pool = Path(args.pool)
    rows = [r for r in csv.DictReader(open(Path(args.cluster) / "images.csv"))
            if not r.get("err")]
    rows = rows[args.shard::args.of]
    print(f"shard {args.shard}/{args.of}: {len(rows):,} images", flush=True)

    out_rows, errs = [], 0
    for i, r in enumerate(rows):
        ip = pool / r["path"]
        base = {"source": r["source"], "path": r["path"],
                "cluster_bg": r["cluster_bg"],
                "cluster_bg_size": r["cluster_bg_size"],
                "max_crop_cluster": r.get("max_crop_cluster", 0)}
        try:
            base.update(score(ip, args.input_size))
        except Exception as e:
            errs += 1
            base["err"] = type(e).__name__
        out_rows.append(base)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1:,}/{len(rows):,}", flush=True)

    keys = sorted({k for r in out_rows for k in r})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {args.out}  rows={len(out_rows):,}  errors={errs}")


if __name__ == "__main__":
    main()
