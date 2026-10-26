#!/usr/bin/env python3
"""Stage E -- rank every image by how likely it is to be a USABLE REAL frame,
then render review sheets in that order from both ends.

Ranking is deliberately transparent rather than learned, because we do not yet
have labels: every input is a feature whose behaviour was checked against
datasets verified by eye, and the weights say which of those was trustworthy.

  bg_edge_den   PRIMARY. Measured 0.000-0.036 across every dataset visually
                confirmed as a render (plain gradient sky, studio wall,
                procedural cloud) and 0.267-0.511 across every dataset with
                real footage. No overlap, so it carries most of the weight.
  blockiness    JPEG 8x8 grid energy. 1.23-1.45 on the video-derived sources
                (screen recordings, YouTube rips) against ~1.01 elsewhere.
  noise_ratio   sensor noise inside the box relative to outside.

bnd_ratio is deliberately NOT used. It separated a hard paste from a soft blend
40.96 to 1.24 on synthetic fixtures, but on this pool it sits at 0.9-1.3 for
renders and real alike -- Roboflow's resize-and-re-encode softens exactly the
edge it listens for. Keeping it would add noise, not signal.

Usability is a separate gate, not folded into the score: a real close-up of an
airframe against a white wall is real and still useless for finding a 20-pixel
target. Images outside the size band are ranked separately so a high real-score
cannot smuggle a hero shot into the keep pile.

Outputs, both ordered so review can stop when it stops being informative:
  ranked_candidates.csv + sheets   best first  -- what we can probably use
  ranked_rejects.csv    + sheets   worst first -- spot-check we are not
                                   discarding anything good

Dependencies: Pillow + numpy only.
"""
import argparse
import csv
import glob
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


def pct_ranks(vals):
    """Percentile rank in [0,1]; NaN -> 0.0 so missing features never promote."""
    idx = [(v, i) for i, v in enumerate(vals) if v is not None and v == v]
    out = [0.0] * len(vals)
    n = len(idx)
    for rank, (_, i) in enumerate(sorted(idx)):
        out[i] = rank / max(n - 1, 1)
    return out


def fnum(r, k):
    try:
        v = float(r[k])
        return None if v != v else v
    except Exception:
        return None


def draw_sheet(items, pool, out_png, cols, cell):
    rows_n = (len(items) + cols - 1) // cols
    pad = 2
    sheet = Image.new("RGB", (cols * (cell + pad), rows_n * (cell + pad)), (22, 22, 24))
    d = ImageDraw.Draw(sheet)
    for i, it in enumerate(items):
        cx, cy = (i % cols) * (cell + pad), (i // cols) * (cell + pad)
        try:
            with Image.open(Path(pool) / it["path"]) as im:
                th = im.convert("RGB")
                th.thumbnail((cell, cell), Image.LANCZOS)
                sheet.paste(th, (cx + (cell - th.width) // 2,
                                 cy + (cell - th.height) // 2))
        except Exception:
            d.rectangle([cx, cy, cx + cell, cy + cell], fill=(80, 20, 20))
        tag = "%d" % i
        d.rectangle([cx, cy, cx + 8 + 7 * len(tag), cy + 13], fill=(0, 0, 0))
        d.text((cx + 3, cy + 2), tag, fill=(255, 220, 90))
    sheet.save(out_png, quality=88)


def write_set(rows, pool, out_dir, name, per, cols, cell, max_sheets):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / ("%s.csv" % name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "sheet", "cell", "score", "source",
                                          "path", "sqrt_in", "bg_edge_den",
                                          "blockiness", "cluster_bg_size"])
        w.writeheader()
        n_sheets = (len(rows) + per - 1) // per
        if max_sheets:
            n_sheets = min(n_sheets, max_sheets)
        for s in range(n_sheets):
            chunk = rows[s * per:(s + 1) * per]
            sheet_name = "sheet_%03d.jpg" % s
            draw_sheet(chunk, pool, out_dir / sheet_name, cols, cell)
            for c, it in enumerate(chunk):
                w.writerow({"rank": s * per + c, "sheet": sheet_name, "cell": c,
                            "score": round(it["score"], 4), "source": it["source"],
                            "path": it["path"], "sqrt_in": it.get("sqrt_in"),
                            "bg_edge_den": it.get("bg_edge_den"),
                            "blockiness": it.get("blockiness"),
                            "cluster_bg_size": it.get("cluster_bg_size")})
        return n_sheets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--scores", required=True, help="dir of features_*.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-sqrt", type=float, default=16.0)
    ap.add_argument("--max-sqrt", type=float, default=256.0)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--cell", type=int, default=160)
    ap.add_argument("--cand-sheets", type=int, default=0, help="0 = all")
    ap.add_argument("--reject-sheets", type=int, default=100)
    ap.add_argument("--class-map", required=True,
                    help="class_map.json -- WITHOUT this the pool contains "
                         "datasets that matched a name search but label "
                         "'person', 'intercepter' or generic drones, and they "
                         "score HIGHEST because busy real backgrounds raise "
                         "bg_edge_den")
    args = ap.parse_args()

    cmap = json.loads(Path(args.class_map).read_text())
    dropped = {k for k, v in cmap.items()
               if isinstance(v, dict) and v.get("_all") == "drop"}

    def shahed_idxs(source):
        v = cmap.get(source)
        if not isinstance(v, dict):
            return None                      # unmapped -> treat as unknown
        return {int(k) for k, t in v.items() if k.isdigit() and t == "shahed"}

    def has_shahed_box(pool, rel, source):
        idxs = shahed_idxs(source)
        if not idxs:
            return False
        parts = list(Path(rel).parts)
        if "images" not in parts:
            return False
        i = len(parts) - 1 - parts[::-1].index("images")
        lp = Path(pool, *parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
        if not lp.exists():
            return False
        for line in lp.read_text(errors="replace").splitlines():
            p = line.split()
            if p:
                try:
                    if int(float(p[0])) in idxs:
                        return True
                except ValueError:
                    pass
        return False

    rows = []
    for f in sorted(glob.glob(str(Path(args.scores) / "features_*.csv"))):
        rows += [r for r in csv.DictReader(open(f)) if not r.get("err")]
    print("loaded %d scored images" % len(rows))

    n0 = len(rows)
    rows = [r for r in rows if r["source"] not in dropped]
    print("  dropped %d images from non-Shahed datasets (%s)"
          % (n0 - len(rows), ", ".join(sorted(dropped))))
    n1 = len(rows)
    rows = [r for r in rows if has_shahed_box(args.pool, r["path"], r["source"])]
    print("  kept %d of %d with at least one shahed-class box" % (len(rows), n1))

    bge = pct_ranks([fnum(r, "bg_edge_den") for r in rows])
    blk = pct_ranks([fnum(r, "blockiness") for r in rows])
    noi = pct_ranks([fnum(r, "noise_ratio") for r in rows])
    for i, r in enumerate(rows):
        r["score"] = 0.60 * bge[i] + 0.25 * blk[i] + 0.15 * noi[i]
        r["_sq"] = fnum(r, "sqrt_in")

    def banded(r):
        return r["_sq"] is not None and args.min_sqrt <= r["_sq"] <= args.max_sqrt

    in_band = [r for r in rows if banded(r)]
    out_band = [r for r in rows if not banded(r)]
    in_band.sort(key=lambda r: -r["score"])
    rejects = list(reversed(in_band))          # same pool, worst-scoring first

    print("in size band [%g,%g]: %d    outside band: %d"
          % (args.min_sqrt, args.max_sqrt, len(in_band), len(out_band)))

    out = Path(args.out)
    per = args.cols * args.rows
    nc = write_set(in_band, args.pool, out / "candidates", "ranked_candidates",
                   per, args.cols, args.cell, args.cand_sheets)
    nr = write_set(rejects, args.pool, out / "rejects", "ranked_rejects",
                   per, args.cols, args.cell, args.reject_sheets)
    print("candidates: %d images -> %d sheets" % (len(in_band), nc))
    print("rejects   : %d sheets rendered from the worst-scoring end" % nr)
    print("outside size band (hero shots / sub-floor specks): %d, not sheeted"
          % len(out_band))


if __name__ == "__main__":
    main()
