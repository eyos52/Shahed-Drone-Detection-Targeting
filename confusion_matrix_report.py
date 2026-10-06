#!/usr/bin/env python3
"""Print a detection confusion matrix as text (no display needed).

Usage:
    python confusion_matrix_report.py CONFIG PREDS.pkl [--score-thr 0.3] [--iou 0.5]

PREDS.pkl comes from:
    python tools/test.py CONFIG CHECKPOINT --out PREDS.pkl

Reads it as: row = ground truth, column = what the model predicted.
The 'background' COLUMN counts ground-truth objects that were never detected
(false negatives). The 'background' ROW counts predictions that matched no
ground-truth object (false positives).
"""
import argparse
import pickle
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("preds")
    ap.add_argument("--score-thr", type=float, default=0.3,
                    help="drop predictions below this confidence (default 0.3)")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="IoU for counting a prediction as matching a GT box")
    ap.add_argument("--repo", default=None,
                    help="path to edgeai-mmdetection (defaults to cwd)")
    args = ap.parse_args()

    repo = Path(args.repo or Path.cwd()).resolve()
    sys.path[:0] = [str(repo), str(repo / "tools" / "analysis_tools")]

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet.registry import DATASETS
    from confusion_matrix import calculate_confusion_matrix

    init_default_scope("mmdet")
    cfg = Config.fromfile(args.config)
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    with open(args.preds, "rb") as f:
        results = pickle.load(f)

    print(f"dataset: {len(dataset)} images   predictions: {len(results)}")
    print(f"score_thr={args.score_thr}  tp_iou_thr={args.iou}\n")

    cm = calculate_confusion_matrix(dataset, results,
                                   score_thr=args.score_thr,
                                   tp_iou_thr=args.iou)
    names = list(dataset.metainfo["classes"]) + ["background"]
    w = max(11, max(len(n) for n in names) + 1)

    print("COUNTS   (row = ground truth, col = predicted)")
    print(" " * w + "".join(f"{n[:w-1]:>{w}}" for n in names) + f"{'row tot':>{w}}")
    for i, n in enumerate(names):
        row = cm[i]
        print(f"{n:<{w}}" + "".join(f"{row[j]:>{w}.0f}" for j in range(len(names)))
              + f"{row.sum():>{w}.0f}")

    print("\nROW-NORMALISED  (% of each true class)")
    print(" " * w + "".join(f"{n[:w-1]:>{w}}" for n in names))
    for i, n in enumerate(names):
        tot = cm[i].sum()
        if tot == 0:
            print(f"{n:<{w}}" + "".join(f"{'-':>{w}}" for _ in names))
            continue
        print(f"{n:<{w}}" + "".join(f"{100*cm[i][j]/tot:>{w}.1f}" for j in range(len(names))))

    # focused read-out per real class: detected-correct vs mislabelled vs missed
    print("\nPER-CLASS BREAKDOWN")
    print(f"{'class':<14}{'GT':>7}{'correct':>9}{'mislabelled':>13}{'missed':>8}"
          f"{'recall':>9}")
    for i, n in enumerate(names[:-1]):
        tot = cm[i].sum()
        if tot == 0:
            print(f"{n:<14}{0:>7}")
            continue
        correct = cm[i][i]
        missed = cm[i][-1]
        mislab = tot - correct - missed
        print(f"{n:<14}{tot:>7.0f}{correct:>9.0f}{mislab:>13.0f}{missed:>8.0f}"
              f"{100*correct/tot:>8.1f}%")
        if mislab > 0:
            worst = sorted(((cm[i][j], names[j]) for j in range(len(names) - 1) if j != i),
                           reverse=True)
            got = ", ".join(f"{nm}={int(c)}" for c, nm in worst if c > 0)
            print(f"{'':<14}  -> called: {got}")


if __name__ == "__main__":
    main()
