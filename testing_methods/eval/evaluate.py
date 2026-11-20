#!/usr/bin/env python3
"""First genuinely held-out evaluation of final_v7i_1280_e60 / last.pt.

Everything in the repo until now was measured on the Roboflow validation split,
which is also what the "test_*" directories contain -- identical ground-truth
counts across all four confusion matrices. This set shares no frames, no source
clips and no platform with the training pool: 11 hand-annotated clips scraped
from Telegram and YouTube after the model was trained.

Reports three things, because the aggregate alone hides what matters:
  1. Ultralytics' own metrics, so the number is comparable to the 0.845 mAP50
     the project has been quoting.
  2. Per-video recall -- an aggregate over 11 clips can be carried by one easy
     one.
  3. False alarms on the 1,647 frames annotated as EMPTY. Those are real
     negatives the annotator asserted, and they are the thing a detector
     deployed against night sky actually has to survive.
"""
import csv, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

S = Path('/lustre/scratch/client/users/muxing')
DS = S/'testset/ds'
W = S/'shahed_repo/models/final_v7i_1280_e60/weights/last.pt'
CONF_REPORT = 0.25          # what a deployment would actually use
IOU_MATCH = 0.5

model = YOLO(str(W))
print("model classes:", model.names)
shahed_idx = [i for i, n in model.names.items() if 'shahed' in str(n).lower()]
shahed_idx = shahed_idx[0] if shahed_idx else 1
print("shahed index:", shahed_idx)

import os
if os.environ.get('SKIP_VAL'):
    print('(skipping ultralytics val -- already measured)')
else:
  print("\n" + "="*62 + "\n1. ULTRALYTICS METRICS (held-out set)\n" + "="*62)
  m = model.val(data=str(DS/'data.yaml'), split='test', imgsz=1280,
                conf=0.001, iou=0.6, verbose=False, plots=False,
                project=str(S/'testset'), name='val', exist_ok=True)
  ap50 = dict(zip(m.box.ap_class_index, m.box.ap50))
  ap = dict(zip(m.box.ap_class_index, m.box.ap))
  print(f"  all   P {m.box.mp:.3f}  R {m.box.mr:.3f}  mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}")
  print(f"  shahed AP50 {ap50.get(shahed_idx, float('nan')):.3f}   "
        f"AP50-95 {ap.get(shahed_idx, float('nan')):.3f}")

print("\n" + "="*62 + f"\n2. PER-VIDEO, at conf={CONF_REPORT}\n" + "="*62)
def iou(a, b):
    x1,y1 = max(a[0],b[0]), max(a[1],b[1]); x2,y2 = min(a[2],b[2]), min(a[3],b[3])
    iw, ih = max(0,x2-x1), max(0,y2-y1); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

imgs = sorted((DS/'images/test').glob('*.jpg'))
stats = defaultdict(lambda: dict(gt=0, tp=0, fn=0, fp_pos=0, neg=0, fp_neg=0))
def chunked(seq, n=100):
    for i in range(0, len(seq), n): yield seq[i:i+n]

pairs = []
for batch in chunked(imgs, 100):
    # predict a batch at a time and drop the Results objects immediately --
    # holding 3399 of them at 1280px OOM-killed a 32GB job
    for p, r in zip(batch, model.predict([str(x) for x in batch], imgsz=1280,
                                         conf=CONF_REPORT, iou=0.45, verbose=False)):
        pairs.append((p, r.orig_shape,
                      [b.xyxy[0].tolist() for b in r.boxes if int(b.cls[0]) == shahed_idx]
                      if r.boxes is not None else []))
    del batch

for p, shape, pred in pairs:
    tag = p.stem.split('_')[0]
    H, Wd = shape
    lp = DS/'labels/test'/(p.stem + '.txt')
    gt = []
    if lp.exists() and lp.stat().st_size:
        for ln in lp.read_text().splitlines():
            c, cx, cy, w, h = (float(x) for x in ln.split()[:5])
            gt.append([(cx-w/2)*Wd, (cy-h/2)*H, (cx+w/2)*Wd, (cy+h/2)*H])
    s = stats[tag]
    if gt:
        s['gt'] += len(gt)
        used = set()
        for g in gt:
            best, bi = 0.0, -1
            for i, q in enumerate(pred):
                if i in used: continue
                v = iou(g, q)
                if v > best: best, bi = v, i
            if best >= IOU_MATCH: s['tp'] += 1; used.add(bi)
            else: s['fn'] += 1
        s['fp_pos'] += max(0, len(pred) - len(used))
    else:
        s['neg'] += 1
        s['fp_neg'] += len(pred)

print(f"{'clip':<6}{'gt':>6}{'TP':>6}{'FN':>6}{'recall':>8}{'neg frames':>12}{'FP on neg':>11}")
tg = ttp = tfn = tneg = tfp = 0
for tag in sorted(stats):
    s = stats[tag]
    rec = s['tp']/s['gt'] if s['gt'] else float('nan')
    print(f"{tag:<6}{s['gt']:>6}{s['tp']:>6}{s['fn']:>6}{rec:>8.3f}{s['neg']:>12}{s['fp_neg']:>11}")
    tg+=s['gt']; ttp+=s['tp']; tfn+=s['fn']; tneg+=s['neg']; tfp+=s['fp_neg']
print(f"{'ALL':<6}{tg:>6}{ttp:>6}{tfn:>6}{(ttp/tg if tg else 0):>8.3f}{tneg:>12}{tfp:>11}")

print("\n" + "="*62 + "\n3. FALSE ALARMS ON TRUE-NEGATIVE FRAMES\n" + "="*62)
print(f"  frames annotated empty : {tneg}")
print(f"  frames with a false box: {tfp}")
print(f"  false-alarm rate       : {100*tfp/max(tneg,1):.1f}% of empty frames")
print(f"  at 25 fps that is      : {tfp/max(tneg,1)*25*60:.0f} false alarms per minute of empty sky")
