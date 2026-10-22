#!/usr/bin/env python3
"""Stage B -- find near-duplicate groups so a whole group can be judged from 2-3 samples.

Implements the "if 200 images are near-identical, check 3 and the verdict covers
all 200" rule. Three hashes per image, each catching a different fake:

  full     whole frame                -> straight re-exports and duplicates
  bg       frame with boxes BLANKED   -> REUSED SKY PLATES. This is the one that
                                        catches "pasted a drone onto a sky
                                        background": the backgrounds collapse
                                        onto a handful of hashes even though the
                                        composites all look different.
  crop     each box, size-normalised  -> the same cutout pasted at different
                                        scales/positions

Uses dHash (robust to the re-encode Roboflow applies on export, unlike md5) and
clusters by Hamming distance with union-find over the *unique* hashes, which is
what keeps it tractable: heavy duplication means few unique hashes.

A background cluster spanning several source datasets is also strong evidence
those datasets are re-exports of each other.

Dependencies: Pillow + numpy only.

Usage:
    python cluster_images.py --pool ./pool --out ./cluster_out [--hamming 6]
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dhash_u64(img, size=8):
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits).view(">u8")[0].item() if bits.size == 64 else 0


def label_for(ip):
    """YOLO label path for an image, tolerating images/ + labels/ trees."""
    parts = list(ip.parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        lp = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
        if lp.exists():
            return lp
    alt = ip.with_suffix(".txt")
    return alt if alt.exists() else None


def read_boxes(lp):
    out = []
    if lp is None:
        return out
    for line in lp.read_text(errors="replace").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            out.append((int(float(p[0])), *(float(v) for v in p[1:5])))
        except ValueError:
            pass
    return out


class Union:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def join(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cluster(hashes, hamming, chunk=2048):
    """hashes: list of ints (may repeat). Returns list of cluster ids, aligned.

    Exact duplicates collapse first, then unique hashes are merged pairwise by
    popcount of XOR -- vectorised in chunks so 10k+ uniques stay cheap."""
    uniq = sorted(set(h for h in hashes if h is not None))
    idx = {h: i for i, h in enumerate(uniq)}
    n = len(uniq)
    u = Union(n)
    if n > 1 and hamming > 0:
        arr = np.array(uniq, dtype=np.uint64)
        for s in range(0, n, chunk):
            blk = arr[s:s + chunk]
            x = np.bitwise_xor(blk[:, None], arr[None, :])
            # popcount on uint64 via byte view
            pc = np.unpackbits(x.view(np.uint8).reshape(x.shape[0], x.shape[1], 8),
                               axis=2).sum(axis=2)
            ii, jj = np.where(pc <= hamming)
            for a, b in zip(ii, jj):
                ga, gb = s + int(a), int(b)
                if ga != gb:
                    u.join(ga, gb)
    roots = {}
    out = []
    for h in hashes:
        if h is None:
            out.append(-1)
            continue
        r = u.find(idx[h])
        out.append(roots.setdefault(r, len(roots)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hamming", type=int, default=6)
    ap.add_argument("--crop-norm", type=int, default=64)
    args = ap.parse_args()

    pool, out = Path(args.pool), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    crop_hashes, crop_owner = [], []
    for ds in sorted(p for p in pool.iterdir() if p.is_dir()):
        src = ds.name
        imgs = [p for p in sorted(ds.rglob("*")) if p.suffix.lower() in IMG_EXT]
        print(f"{src}: {len(imgs):,} images", flush=True)
        for ip in imgs:
            try:
                with Image.open(ip) as im:
                    im.load()
                    W, H = im.size
                    h_full = dhash_u64(im)
                    boxes = read_boxes(label_for(ip))
                    # blank the annotated regions, then hash what remains
                    bg = im.convert("RGB").copy()
                    px = np.asarray(bg).copy()
                    for _, cx, cy, bw, bh in boxes:
                        x1 = max(int((cx - bw / 2) * W), 0)
                        y1 = max(int((cy - bh / 2) * H), 0)
                        x2 = min(int((cx + bw / 2) * W), W)
                        y2 = min(int((cy + bh / 2) * H), H)
                        if x2 > x1 and y2 > y1:
                            px[y1:y2, x1:x2] = 0
                    h_bg = dhash_u64(Image.fromarray(px))
                    for bi, (cls, cx, cy, bw, bh) in enumerate(boxes):
                        x1 = max(int((cx - bw / 2) * W), 0)
                        y1 = max(int((cy - bh / 2) * H), 0)
                        x2 = min(int((cx + bw / 2) * W), W)
                        y2 = min(int((cy + bh / 2) * H), H)
                        if x2 - x1 < 4 or y2 - y1 < 4:
                            continue
                        crop = im.convert("RGB").crop((x1, y1, x2, y2)).resize(
                            (args.crop_norm, args.crop_norm), Image.LANCZOS)
                        crop_hashes.append(dhash_u64(crop))
                        crop_owner.append(len(rows))
            except Exception as e:
                rows.append({"source": src, "path": str(ip.relative_to(pool)),
                             "err": type(e).__name__, "w": 0, "h": 0,
                             "n_boxes": 0, "hash_full": None, "hash_bg": None})
                continue
            rows.append({"source": src, "path": str(ip.relative_to(pool)), "err": "",
                         "w": W, "h": H, "n_boxes": len(boxes),
                         "hash_full": h_full, "hash_bg": h_bg})

    print(f"\nhashing done: {len(rows):,} images, {len(crop_hashes):,} crops")
    cl_full = cluster([r["hash_full"] for r in rows], args.hamming)
    cl_bg = cluster([r["hash_bg"] for r in rows], args.hamming)
    cl_crop = cluster(crop_hashes, args.hamming) if crop_hashes else []

    nf, nb = Counter(cl_full), Counter(cl_bg)
    crop_cl_size = Counter(cl_crop)
    biggest_crop = defaultdict(int)
    for owner, c in zip(crop_owner, cl_crop):
        biggest_crop[owner] = max(biggest_crop[owner], crop_cl_size[c])

    for i, r in enumerate(rows):
        r["cluster_full"] = cl_full[i]
        r["cluster_full_size"] = nf[cl_full[i]]
        r["cluster_bg"] = cl_bg[i]
        r["cluster_bg_size"] = nb[cl_bg[i]]
        r["max_crop_cluster"] = biggest_crop.get(i, 0)

    man = out / "images.csv"
    with open(man, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    # background clusters, largest first -- these are the composite factories
    bg_groups = defaultdict(list)
    for r in rows:
        bg_groups[r["cluster_bg"]].append(r)
    summary = []
    for cid, members in sorted(bg_groups.items(), key=lambda kv: -len(kv[1])):
        srcs = Counter(m["source"] for m in members)
        summary.append({"cluster_bg": cid, "size": len(members),
                        "sources": dict(srcs), "n_sources": len(srcs),
                        "example": members[0]["path"]})
    (out / "bg_clusters.json").write_text(json.dumps(summary, indent=1))

    print(f"\n{'size':>7}  {'srcs':>4}  example")
    for s in summary[:25]:
        print(f"{s['size']:>7}  {s['n_sources']:>4}  {s['example'][:76]}")
    singles = sum(1 for s in summary if s["size"] == 1)
    print(f"\nbackground clusters: {len(summary):,}  (singletons: {singles:,})")
    print(f"images in clusters of 50+: "
          f"{sum(s['size'] for s in summary if s['size'] >= 50):,}")
    print(f"cross-source clusters (re-exports): "
          f"{sum(1 for s in summary if s['n_sources'] > 1)}")
    print(f"\nwrote {man}, {out/'bg_clusters.json'}")


if __name__ == "__main__":
    main()
