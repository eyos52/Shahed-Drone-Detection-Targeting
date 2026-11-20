#!/usr/bin/env python3
"""Burn ground truth and model predictions into the same frame, and re-score at
several IoU thresholds.

Aggregate metrics cannot distinguish "the model missed the target" from "the
label was in the wrong place". Watching both boxes on the same frame can.

  GREEN  = annotation (ground truth)
  ORANGE = model prediction, with confidence
  IoU printed when both are present on a frame

Predictions are cached to preds.jsonl so re-rendering never re-runs inference.
"""
import json, subprocess
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO
import imageio_ffmpeg

S  = Path('/lustre/scratch/client/users/muxing')
DS = S/'testset/ds'
OUT = S/'testset/overlay'; OUT.mkdir(parents=True, exist_ok=True)
W  = S/'shahed_repo/models/final_v7i_1280_e60/weights/last.pt'
CONF = 0.25
FF = imageio_ffmpeg.get_ffmpeg_exe()

model = YOLO(str(W))
SH = [i for i,n in model.names.items() if 'shahed' in str(n).lower()][0]

imgs = sorted((DS/'images/test').glob('*.jpg'))
cache = S/'testset/preds.jsonl'
if not cache.exists():
    with open(cache,'w') as fh:
        for i in range(0, len(imgs), 100):
            batch = imgs[i:i+100]
            for p, r in zip(batch, model.predict([str(x) for x in batch], imgsz=1280,
                                                 conf=CONF, iou=0.45, verbose=False)):
                fh.write(json.dumps({'n': p.name, 'shape': list(r.orig_shape),
                    'b': [[*map(float, b.xyxy[0].tolist()), float(b.conf[0])]
                          for b in (r.boxes or []) if int(b.cls[0]) == SH]})+'\n')
            print(f"  predicted {min(i+100,len(imgs))}/{len(imgs)}", flush=True)
preds = {json.loads(l)['n']: json.loads(l) for l in open(cache)}
print(f"predictions cached for {len(preds)} frames")

def iou(a,b):
    x1,y1 = max(a[0],b[0]), max(a[1],b[1]); x2,y2 = min(a[2],b[2]), min(a[3],b[3])
    iw,ih = max(0,x2-x1), max(0,y2-y1); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua>0 else 0.0

def gt_of(name, Wd, H):
    lp = DS/'labels/test'/(Path(name).stem + '.txt')
    out = []
    if lp.exists() and lp.stat().st_size:
        for ln in lp.read_text().splitlines():
            c,cx,cy,w,h = (float(x) for x in ln.split()[:5])
            out.append([(cx-w/2)*Wd,(cy-h/2)*H,(cx+w/2)*Wd,(cy+h/2)*H])
    return out

# ---- render one overlay video per clip -------------------------------------
byclip = defaultdict(list)
for p in imgs: byclip[p.stem.split('_')[0]].append(p)

for tag, frames in sorted(byclip.items()):
    tmp = OUT/f'_{tag}'; tmp.mkdir(exist_ok=True)
    for k, p in enumerate(sorted(frames)):
        rec = preds.get(p.name, {'shape':[0,0],'b':[]})
        im = Image.open(p).convert('RGB'); Wd,H = im.size
        dr = ImageDraw.Draw(im)
        g = gt_of(p.name, Wd, H)
        for b in g:
            dr.rectangle(b, outline=(0,255,80), width=3)
        for *b,c in rec['b']:
            dr.rectangle(b, outline=(255,150,0), width=3)
            dr.text((b[0], max(0,b[1]-12)), f"{c:.2f}", fill=(255,150,0))
        best = max([iou(a,b[:4]) for a in g for b in rec['b']], default=None)
        txt = f"{tag}  f{k:04d}  GT:{len(g)}  pred:{len(rec['b'])}"
        if best is not None: txt += f"  IoU:{best:.2f}"
        elif g and not rec['b']: txt += "  MISS"
        elif rec['b'] and not g: txt += "  FALSE ALARM"
        dr.rectangle([0,0,460,20], fill=(0,0,0))
        dr.text((4,4), txt, fill=(255,255,255))
        im.save(tmp/f'f{k:06d}.jpg', quality=88)
    subprocess.run([FF,'-y','-hide_banner','-loglevel','error','-framerate','25',
                    '-i',str(tmp/'f%06d.jpg'),'-c:v','libx264','-crf','20',
                    '-pix_fmt','yuv420p',str(OUT/f'{tag}_overlay.mp4')],
                   capture_output=True, timeout=600)
    for f in tmp.iterdir(): f.unlink()
    tmp.rmdir()
    print(f"  {tag}_overlay.mp4", flush=True)

# ---- recall vs IoU threshold ----------------------------------------------
print("\n" + "="*58 + "\nRECALL vs IoU THRESHOLD (does box convention explain it?)\n" + "="*58)
print(f"{'IoU':>6}{'TP':>8}{'recall':>9}")
for thr in (0.1,0.2,0.3,0.4,0.5,0.7):
    tp = gt_tot = 0
    for p in imgs:
        rec = preds.get(p.name, {'b':[]})
        im_w, im_h = Image.open(p).size
        g = gt_of(p.name, im_w, im_h)
        gt_tot += len(g); used=set()
        for a in g:
            best, bi = 0.0, -1
            for i,b in enumerate(rec['b']):
                if i in used: continue
                v = iou(a, b[:4])
                if v > best: best, bi = v, i
            if best >= thr: tp += 1; used.add(bi)
    print(f"{thr:>6.1f}{tp:>8}{tp/max(gt_tot,1):>9.3f}")
