#!/usr/bin/env python3
"""Stage F -- pull a human-verified selection out of a review set into the
training dataset, carrying labels and full provenance.

Input is the shorthand used during review:  "22:21,39,66,67; 29:38,83,92; ..."
meaning sheet 22 cells 21/39/66/67, and so on. A missing semicolon between
groups is tolerated, because the list is typed by hand.

Label handling, and why it is not shahed-only:
  Each source dataset numbers its classes differently, so labels are rewritten
  into one shared space via class_map.json. Every box is kept and remapped --
  NOT just the shahed ones. Dropping a bird box while keeping the image would
  leave a visible, unlabelled bird in a training frame, which actively teaches
  the model that birds are background. Birds are the dominant false positive
  for anything pointed at the sky, so that is the last lesson we want to teach.
  Anything whose class cannot be mapped is reported, never silently dropped.

Output is a plain YOLO tree plus a manifest recording, for every image, where
it came from and which sheet and cell it was approved on -- so a later
disagreement about an image can be traced back to the decision that admitted it.

Usage:
    python collect_selection.py --pool POOL --review REVIEW_CSV \\
        --class-map class_map.json --out DATASET_DIR \\
        --selection "22:21,39,66,67; 29:38,83,92" [--note "rejects pass 1"]
"""
import argparse
import csv
import json
import re
import shutil
from collections import Counter, OrderedDict
from pathlib import Path

# shared label space for the assembled dataset
TAXONOMY = ["shahed", "bird", "helicopter", "other"]


def parse_selection(s):
    """'22:21,39,66,67; 29:38,83' -> {22: [21,39,66,67], 29: [38,83]}.
    Tolerates missing separators between groups and stray whitespace."""
    out = OrderedDict()
    # cells must be comma-separated, so a bare space before the next "NN:"
    # ends the list instead of swallowing that sheet number as a cell --
    # "82:3,73,96,99 83:3,24" has to yield sheet 83, not cell 83 of sheet 82
    for m in re.finditer(r"(\d+)\s*:\s*(\d+(?:\s*,\s*\d+)*)", s):
        sheet = int(m.group(1))
        cells = [int(c) for c in re.findall(r"\d+", m.group(2))]
        out.setdefault(sheet, [])
        out[sheet].extend(cells)
    return out


def label_path_for(pool, rel):
    parts = list(Path(rel).parts)
    if "images" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("images")
    return Path(pool, *parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--review", required=True,
                    help="ranked_rejects.csv or ranked_candidates.csv")
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    sel = parse_selection(args.selection)
    total = sum(len(v) for v in sel.values())
    print("selection: %d sheets, %d cells" % (len(sel), total))

    cmap = json.loads(Path(args.class_map).read_text())
    index = {}
    for r in csv.DictReader(open(args.review)):
        m = re.search(r"(\d+)", r["sheet"])
        if m:
            index[(int(m.group(1)), int(r["cell"]))] = r

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    man = []
    missing, nolabel, unmapped = [], [], Counter()
    cls_count = Counter()

    for sheet, cells in sel.items():
        for cell in cells:
            r = index.get((sheet, cell))
            if r is None:
                missing.append("%d:%d" % (sheet, cell))
                continue
            src, rel = r["source"], r["path"]
            ip = Path(args.pool) / rel
            if not ip.exists():
                missing.append("%d:%d (%s)" % (sheet, cell, rel))
                continue

            stem = "%s__%s" % (src, Path(rel).stem)
            new_img = out / "images" / (stem + ip.suffix.lower())
            shutil.copy2(ip, new_img)

            lp = label_path_for(args.pool, rel)
            lines = []
            if lp and lp.exists():
                m = cmap.get(src, {})
                for line in lp.read_text(errors="replace").splitlines():
                    p = line.split()
                    if len(p) < 5:
                        continue
                    try:
                        idx = int(float(p[0]))
                    except ValueError:
                        continue
                    target = m.get(str(idx))
                    if target in TAXONOMY:
                        new_idx = TAXONOMY.index(target)
                        cls_count[target] += 1
                        lines.append(" ".join([str(new_idx)] + p[1:5]))
                    else:
                        unmapped["%s:%d" % (src, idx)] += 1
            else:
                nolabel.append(stem)
            (out / "labels" / (stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))

            man.append({"new_name": new_img.name, "source": src, "orig_path": rel,
                        "review_sheet": sheet, "review_cell": cell,
                        "score": r.get("score"), "sqrt_in": r.get("sqrt_in"),
                        "cluster_bg_size": r.get("cluster_bg_size"),
                        "n_boxes": len(lines), "note": args.note})

    mf = out / "manifest.csv"
    write_header = not mf.exists()
    with open(mf, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0].keys()) if man else ["new_name"])
        if write_header and man:
            w.writeheader()
        w.writerows(man)

    (out / "data.yaml").write_text(
        "path: %s\ntrain: images\nval: images\nnc: %d\nnames: %s\n"
        % (out.resolve(), len(TAXONOMY), TAXONOMY))

    print("copied %d images" % len(man))
    print("boxes by class: %s" % dict(cls_count))
    if nolabel:
        print("NO LABEL FILE (need hand-labelling): %d -> %s"
              % (len(nolabel), ", ".join(nolabel[:5])))
    if unmapped:
        print("UNMAPPED class indices (boxes NOT written): %s" % dict(unmapped))
    if missing:
        print("NOT FOUND in review csv: %s" % ", ".join(missing))
    print("manifest -> %s" % mf)


if __name__ == "__main__":
    main()
