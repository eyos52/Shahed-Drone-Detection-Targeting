#!/usr/bin/env python3
"""Stage D -- build numbered contact sheets for human (or model) adjudication.

Reviewing thousands of images one at a time is hopeless; a 10x10 grid shows 100
at once and is one file to open. Sheets are rendered to disk so they can be
viewed over OnDemand/JupyterLab on the cluster without downloading the dataset.

Two modes, matching the "check 2-3 per group" rule:

  --mode reps        N representatives per background cluster, clusters ordered
                     LARGEST FIRST. One verdict per cell covers the whole
                     cluster, so the first few sheets decide the most images.
                     Each cell is labelled `cell:clustersize`.

  --mode singletons  images whose background cluster is <= --max-size, i.e. the
                     un-duplicated tail where genuine photos live. Labelled
                     `cell`.

Every sheet gets a CSV manifest mapping cell number -> image, so rejections can
be recorded as plain text (`sheet_007: 3,17,42`) with no UI involved.

Dependencies: Pillow + numpy only.

Usage:
    python contact_sheets.py --pool ./pool --cluster ./cluster_out \\
        --out ./sheets --mode reps --per-cluster 3
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def load_rows(cluster_dir):
    with open(Path(cluster_dir) / "images.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("cluster_bg", "cluster_bg_size", "cluster_full_size",
                  "max_crop_cluster", "n_boxes"):
            try:
                r[k] = int(r[k])
            except (ValueError, KeyError, TypeError):
                r[k] = 0
    return rows


def draw_sheet(items, pool, out_png, cols, cell, label_cluster):
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
        tag = f"{i}:{it['cluster_bg_size']}" if label_cluster else str(i)
        d.rectangle([cx, cy, cx + 8 + 7 * len(tag), cy + 13], fill=(0, 0, 0))
        d.text((cx + 3, cy + 2), tag, fill=(255, 220, 90))
    sheet.save(out_png, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["reps", "singletons"], default="reps")
    ap.add_argument("--per-cluster", type=int, default=3)
    ap.add_argument("--max-size", type=int, default=3,
                    help="singletons mode: max cluster size to include")
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--cell", type=int, default=160)
    ap.add_argument("--max-sheets", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    rows = [r for r in load_rows(args.cluster) if not r.get("err")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.mode == "reps":
        groups = defaultdict(list)
        for r in rows:
            groups[r["cluster_bg"]].append(r)
        ordered = sorted(groups.values(), key=lambda g: -len(g))
        items = []
        for g in ordered:
            step = max(1, len(g) // args.per_cluster)
            items.extend(g[::step][:args.per_cluster])
        label_cluster = True
    else:
        items = [r for r in rows if r["cluster_bg_size"] <= args.max_size]
        label_cluster = False

    per = args.cols * args.rows
    n_sheets = (len(items) + per - 1) // per
    if args.max_sheets:
        n_sheets = min(n_sheets, args.max_sheets)
    print(f"mode={args.mode}  candidates={len(items):,}  "
          f"sheets={n_sheets} ({per}/sheet)")

    man_path = out / f"manifest_{args.mode}.csv"
    with open(man_path, "w", newline="") as mf:
        wr = csv.writer(mf)
        wr.writerow(["sheet", "cell", "source", "path", "cluster_bg",
                     "cluster_bg_size", "n_boxes"])
        for s in range(n_sheets):
            chunk = items[s * per:(s + 1) * per]
            name = f"sheet_{s:03d}.jpg"
            draw_sheet(chunk, args.pool, out / name, args.cols, args.cell,
                       label_cluster)
            for c, it in enumerate(chunk):
                wr.writerow([name, c, it["source"], it["path"], it["cluster_bg"],
                             it["cluster_bg_size"], it["n_boxes"]])
            covered = sum(it["cluster_bg_size"] for it in chunk) if label_cluster else len(chunk)
            print(f"  {name}  {len(chunk):>3} cells  covers ~{covered:,} images")

    print(f"\nwrote {n_sheets} sheets + {man_path}")
    print("review by noting cell numbers per sheet, e.g.  sheet_007: 3,17,42")


if __name__ == "__main__":
    main()
