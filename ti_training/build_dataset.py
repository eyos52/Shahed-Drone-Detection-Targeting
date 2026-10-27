#!/usr/bin/env python3
# Index + combine every Shahed/AOD source into one 5-class detection dataset,
# and emit it in BOTH formats we need:
#
#   /content/combined5/      YOLO layout (Ultralytics baseline)
#   /content/tidl_dataset/   COCO layout (edgeai-mmdetection / YOLOv7-lite)
#   /content/index.csv       one row per box, for EDA
#
# Re-runnable: it rebuilds from scratch every time, so edit the config block
# and run it again. Images are symlinked, never copied.

import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Paths come from the environment so the same script runs on Colab and on HPC,
# where /content does not exist and 8.6 GB must not live in $HOME.
#   SHAHED_ROOT     working root            (default /content)
#   SHAHED_DATASETS extracted DataSets dir  (default $SHAHED_ROOT/ds/DataSets)
_ROOT = Path(os.environ.get("SHAHED_ROOT", "/content"))
BASE = Path(os.environ.get("SHAHED_DATASETS", _ROOT / "ds" / "DataSets"))
YOLO_OUT = Path(os.environ.get("SHAHED_YOLO_OUT", _ROOT / "combined5"))
COCO_OUT = Path(os.environ.get("SHAHED_COCO_OUT", _ROOT / "tidl_dataset"))
INDEX_CSV = Path(os.environ.get("SHAHED_INDEX", _ROOT / "index.csv"))

CLASSES = ["airplane", "bird", "drone", "helicopter", "shahed"]
SEED = 42
IMG_EXT = (".jpg", ".jpeg", ".png")

A = BASE / "AOD 4"
_AOD_LBL = A / "Annotations" / "YOLOv8 format"

# remap: source class id -> final class id. Ids absent from remap are dropped
# (counted and reported). split: "native" keeps the source's own split, or a
# (train, valid, test) fraction triple.
SOURCES = [
    {
        "name": "aod4",
        # airplane/bird/drone/helicopter keep their identity in the 5-class space
        "remap": {0: 0, 1: 1, 2: 2, 3: 3},
        "split": "native",
        "pairs": [
            (A / "Images/train", _AOD_LBL / "train/labels", "train"),
            (A / "Images/valid", _AOD_LBL / "valid/labels", "valid"),
            (A / "Images/test", _AOD_LBL / "test/labels", "test"),
        ],
    },
    {
        "name": "shahed_roboflow",
        # nc=2 ['other','shashed']; 'shashed' (typo in source) -> shahed.
        # class 0 'other' is an unspecified aerial object with no 5-class
        # equivalent, so its boxes are dropped -- see DROPPED report below.
        "remap": {1: 4},
        "split": (0.60, 0.20, 0.20),
        "pairs": [(BASE / "shahed/train/images", BASE / "shahed/train/labels", None)],
    },
    {
        "name": "shahed136",
        # nc=1 ['shahed-136'] -> shahed. Real footage, so it is re-split across
        # train/valid/test rather than left test-only.
        "remap": {0: 4},
        "split": (0.60, 0.20, 0.20),
        "pairs": [(BASE / "Shahed-136_youtube_batch2/test/images",
                   BASE / "Shahed-136_youtube_batch2/test/labels", None)],
    },
    {
        "name": "shahed_synth",
        # synthetic; labels already live at index 1 in a 2-class space.
        # Kept out of test so the test split stays real-imagery only.
        "remap": {1: 4},
        "split": (0.80, 0.20, 0.0),
        "pairs": [(BASE / "shahed_dataset_1/prepped/images",
                   BASE / "shahed_dataset_1/prepped/labels", None)],
    },
]

SPLITS = ("train", "valid", "test")


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def parse_label(path, remap):
    """Return (boxes, dropped_ids, bad_rows). boxes are (cls, x, y, w, h)."""
    boxes, dropped, bad = [], Counter(), 0
    for row in path.read_text().splitlines():
        row = row.strip()
        if not row:
            continue
        parts = row.split()
        if len(parts) < 5:
            bad += 1
            continue
        try:
            cid = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            bad += 1
            continue
        if cid not in remap:
            dropped[cid] += 1
            continue
        if not (w > 0 and h > 0):
            bad += 1
            continue
        # clip the box to the image in corner space; AOD 4 ships at least one
        # label whose right edge sits 40% past the frame
        x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        x1, y1 = max(x1, 0.0), max(y1, 0.0)
        x2, y2 = min(x2, 1.0), min(y2, 1.0)
        if x2 <= x1 or y2 <= y1:
            bad += 1
            continue
        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        boxes.append((remap[cid], x, y, w, h))
    return boxes, dropped, bad


def build_index():
    records, reports = [], []
    for src in SOURCES:
        rep = Counter()
        dropped_ids = Counter()
        for img_dir, lbl_dir, native_split in src["pairs"]:
            if not img_dir.exists() or not lbl_dir.exists():
                print(f"  !! missing {img_dir if not img_dir.exists() else lbl_dir}")
                continue
            imgs = {p.stem: p for p in img_dir.iterdir()
                    if p.suffix.lower() in IMG_EXT}
            lbls = {p.stem: p for p in lbl_dir.glob("*.txt")}
            rep["images_seen"] += len(imgs)
            rep["labels_seen"] += len(lbls)
            rep["image_only"] += len(set(imgs) - set(lbls))
            rep["label_only"] += len(set(lbls) - set(imgs))
            for stem in sorted(set(imgs) & set(lbls)):
                boxes, drops, bad = parse_label(lbls[stem], src["remap"])
                dropped_ids.update(drops)
                rep["bad_rows"] += bad
                if not boxes:
                    rep["no_usable_boxes"] += 1
                    continue
                try:
                    with Image.open(imgs[stem]) as im:
                        wpx, hpx = im.size
                except Exception:
                    rep["unreadable_image"] += 1
                    continue
                records.append({
                    "source": src["name"], "stem": stem, "path": imgs[stem],
                    "native_split": native_split, "w": wpx, "h": hpx,
                    "boxes": boxes,
                })
                rep["kept"] += 1
        reports.append((src["name"], rep, dropped_ids))
    return records, reports


# ---------------------------------------------------------------------------
# Dedup + split
# ---------------------------------------------------------------------------
def dedup(records):
    seen, out, dups = {}, [], []
    for r in records:
        h = file_md5(r["path"])
        if h in seen:
            dups.append((r["source"], r["stem"], seen[h]))
            continue
        seen[h] = f'{r["source"]}/{r["stem"]}'
        out.append(r)
    return out, dups


def assign_splits(records):
    rng = random.Random(SEED)
    policy = {s["name"]: s["split"] for s in SOURCES}
    by_source = defaultdict(list)
    for r in records:
        by_source[r["source"]].append(r)

    for name, group in by_source.items():
        pol = policy[name]
        if pol == "native":
            for r in group:
                r["split"] = r["native_split"]
            continue
        group = sorted(group, key=lambda r: r["stem"])
        rng.shuffle(group)
        n = len(group)
        n_tr = int(round(n * pol[0]))
        n_va = int(round(n * pol[1]))
        for i, r in enumerate(group):
            r["split"] = "train" if i < n_tr else ("valid" if i < n_tr + n_va else "test")
    return records


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
def emit_yolo(records):
    if YOLO_OUT.exists():
        shutil.rmtree(YOLO_OUT)
    for split in SPLITS:
        (YOLO_OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (YOLO_OUT / split / "labels").mkdir(parents=True, exist_ok=True)

    for r in records:
        name = f'{r["source"]}__{r["stem"]}'
        img_link = YOLO_OUT / r["split"] / "images" / (name + r["path"].suffix.lower())
        if not img_link.exists():
            img_link.symlink_to(r["path"])
        lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in r["boxes"]]
        (YOLO_OUT / r["split"] / "labels" / (name + ".txt")).write_text("\n".join(lines) + "\n")
        r["out_name"] = img_link.name

    (YOLO_OUT / "data.yaml").write_text(
        f"path: {YOLO_OUT}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")


def emit_coco(records):
    # mmdetection CocoDataset: data_root + ann_file + data_prefix(img=...)
    if COCO_OUT.exists():
        shutil.rmtree(COCO_OUT)
    (COCO_OUT / "annotations").mkdir(parents=True, exist_ok=True)
    # reuse the YOLO symlink trees as the image dirs
    for split in SPLITS:
        (COCO_OUT / split).symlink_to(YOLO_OUT / split / "images",
                                      target_is_directory=True)

    cats = [{"id": i + 1, "name": c, "supercategory": "none"}
            for i, c in enumerate(CLASSES)]
    per_split = defaultdict(list)
    for r in records:
        per_split[r["split"]].append(r)

    for split in SPLITS:
        images, annotations = [], []
        ann_id = 1
        for img_id, r in enumerate(sorted(per_split[split], key=lambda z: z["out_name"]), 1):
            images.append({"id": img_id, "file_name": r["out_name"],
                           "width": r["w"], "height": r["h"]})
            for c, x, y, w, h in r["boxes"]:
                # YOLO cx,cy,w,h normalised -> COCO x,y,w,h absolute
                bw, bh = w * r["w"], h * r["h"]
                bx, by = (x * r["w"]) - bw / 2, (y * r["h"]) - bh / 2
                bx, by = max(bx, 0.0), max(by, 0.0)
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": c + 1,
                    "bbox": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2), "iscrowd": 0, "segmentation": [],
                })
                ann_id += 1
        (COCO_OUT / "annotations" / f"instances_{split}.json").write_text(json.dumps({
            "info": {"description": "AOD4 + Shahed, 5-class", "version": "1"},
            "licenses": [], "categories": cats,
            "images": images, "annotations": annotations}))
        print(f"  instances_{split}.json: {len(images):,} images, {len(annotations):,} boxes")


def emit_index(records):
    with open(INDEX_CSV, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["source", "split", "out_name", "img_w", "img_h",
                     "cls_id", "cls_name", "cx", "cy", "bw", "bh",
                     "box_w_px", "box_h_px", "area_px"])
        for r in records:
            for c, x, y, w, h in r["boxes"]:
                wr.writerow([r["source"], r["split"], r["out_name"], r["w"], r["h"],
                             c, CLASSES[c], f"{x:.6f}", f"{y:.6f}",
                             f"{w:.6f}", f"{h:.6f}",
                             round(w * r["w"], 2), round(h * r["h"], 2),
                             round(w * r["w"] * h * r["h"], 2)])


# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("STEP 1  index every source")
    print("=" * 72)
    records, reports = build_index()
    for name, rep, dropped in reports:
        print(f"  {name:18} {dict(rep)}")
        if dropped:
            print(f"  {'':18} DROPPED source class ids (no 5-class equivalent): "
                  f"{dict(dropped)}")
    print(f"\n  images with usable boxes: {len(records):,}")

    print("\n" + "=" * 72)
    print("STEP 2  md5 dedup across the merged pool")
    print("=" * 72)
    records, dups = dedup(records)
    print(f"  exact duplicate images removed: {len(dups)}")
    for d in dups[:5]:
        print(f"    {d[0]}/{d[1]}  ==  {d[2]}")
    print(f"  remaining: {len(records):,}")

    print("\n" + "=" * 72)
    print("STEP 3  assign splits")
    print("=" * 72)
    records = assign_splits(records)
    for s in SOURCES:
        g = [r for r in records if r["source"] == s["name"]]
        c = Counter(r["split"] for r in g)
        print(f"  {s['name']:18} policy={str(s['split']):22} "
              f"train={c['train']:<6} valid={c['valid']:<6} test={c['test']}")

    print("\n" + "=" * 72)
    print("STEP 4  emit YOLO layout (symlinked images)")
    print("=" * 72)
    emit_yolo(records)
    print(f"  wrote {YOLO_OUT}")

    print("\n" + "=" * 72)
    print("STEP 5  emit COCO layout for edgeai-mmdetection")
    print("=" * 72)
    emit_coco(records)
    print(f"  wrote {COCO_OUT}")

    emit_index(records)
    print(f"\n  wrote {INDEX_CSV}")

    print("\n" + "=" * 72)
    print("FINAL  class balance")
    print("=" * 72)
    print(f"  {'split':8}{'images':>9}{'boxes':>9}   " +
          "".join(f"{c:>12}" for c in CLASSES))
    for split in SPLITS:
        g = [r for r in records if r["split"] == split]
        cc = Counter(c for r in g for c, *_ in r["boxes"])
        print(f"  {split:8}{len(g):>9,}{sum(cc.values()):>9,}   " +
              "".join(f"{cc.get(i, 0):>12,}" for i in range(len(CLASSES))))
    imgs_per_src = Counter(r["source"] for r in records)
    print(f"\n  per-source contribution: {dict(imgs_per_src)}")


if __name__ == "__main__":
    main()
