#!/usr/bin/env python3
"""Pure-negative test: a fighter-jet clip containing no Shahed at all.

No positives means no recall to argue about and no label quality to blame --
every detection the model makes here is wrong by construction. That isolates
exactly the failure Andy observed by eye: distant aircraft and dark blobs being
called Shahed.

Also breaks results out by predicted class. The model has two: 'not' and
'shahed'. Calling a fighter jet 'not' is correct behaviour; calling it 'shahed'
is the failure. The distinction matters and the aggregate hides it.
"""
import json, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import imageio_ffmpeg

S = Path('/lustre/scratch/client/users/muxing')
VID = next((S/'keepers').glob('k48_*.mp4'))
FR  = S/'testset/frames_neg'; FR.mkdir(parents=True, exist_ok=True)
OUT = S/'testset/overlay'; OUT.mkdir(parents=True, exist_ok=True)
FF  = imageio_ffmpeg.get_ffmpeg_exe()
CONF = 0.25

if not any(FR.iterdir()):
    subprocess.run([FF,'-y','-hide_banner','-loglevel','error','-i',str(VID),
                    '-fps_mode','passthrough','-q:v','2',str(FR/'frame_%06d.jpg')],
                   capture_output=True, timeout=900)
frames = sorted(FR.glob('*.jpg'))
if not frames:
    raise SystemExit("no frames extracted -- check ffmpeg invocation")
print(f"clip: {VID.name}\nframes: {len(frames)}  (ground truth: ZERO shahed)\n")

model = YOLO(str(S/'shahed_repo/models/final_v7i_1280_e60/weights/last.pt'))
SH = [i for i,n in model.names.items() if 'shahed' in str(n).lower()][0]
print("classes:", model.names)

rows=[]
for i in range(0, len(frames), 100):
    b = frames[i:i+100]
    for p, r in zip(b, model.predict([str(x) for x in b], imgsz=1280, conf=CONF,
                                     iou=0.45, verbose=False)):
        dets=[(int(x.cls[0]), float(x.conf[0]), [float(v) for v in x.xyxy[0]])
              for x in (r.boxes or [])]
        rows.append((p, dets))

sh = [(p,d) for p,d in rows if any(c==SH for c,_,_ in d)]
notc = [(p,d) for p,d in rows if any(c!=SH for c,_,_ in d)]
n_sh_boxes = sum(1 for _,d in rows for c,_,_ in d if c==SH)
n_not_boxes = sum(1 for _,d in rows for c,_,_ in d if c!=SH)
confs = np.array([cf for _,d in rows for c,cf,_ in d if c==SH]) if n_sh_boxes else np.array([])

print("\n"+"="*58+"\nFALSE ALARMS ON A CLIP WITH NO SHAHED\n"+"="*58)
print(f"  frames                      : {len(rows)}")
print(f"  frames w/ a SHAHED box      : {len(sh)}  ({100*len(sh)/len(rows):.1f}%)")
print(f"  total SHAHED boxes          : {n_sh_boxes}")
print(f"  frames w/ a 'not' box       : {len(notc)}  ({100*len(notc)/len(rows):.1f}%)  <- correct class")
print(f"  total 'not' boxes           : {n_not_boxes}")
if confs.size:
    print(f"\n  false-positive confidence: min {confs.min():.2f}  p50 {np.median(confs):.2f}  "
          f"p90 {np.percentile(confs,90):.2f}  max {confs.max():.2f}")
    for t in (0.25,0.4,0.5,0.6,0.7,0.8):
        print(f"     conf>={t:.2f}: {int((confs>=t).sum()):4} boxes on "
              f"{len({p for p,d in rows for c,cf,_ in d if c==SH and cf>=t}):4} frames")
fps = 25.0
print(f"\n  false alarms per minute (at {fps:.0f} fps): {len(sh)/len(rows)*fps*60:.0f}")

# overlay
tmp = OUT/'_k48neg'; tmp.mkdir(exist_ok=True)
for k,(p,dets) in enumerate(rows):
    im = Image.open(p).convert('RGB'); dr = ImageDraw.Draw(im)
    for c,cf,b in dets:
        col = (255,40,40) if c==SH else (90,170,255)
        dr.rectangle(b, outline=col, width=3)
        dr.text((b[0], max(0,b[1]-12)), f"{model.names[c]} {cf:.2f}", fill=col)
    dr.rectangle([0,0,470,20], fill=(0,0,0))
    nsh = sum(1 for c,_,_ in dets if c==SH)
    dr.text((4,4), f"k48 FIGHTER JET - no shahed present   f{k:04d}   "
                   f"shahed-FP:{nsh}", fill=(255,255,255))
    im.save(tmp/f'f{k:06d}.jpg', quality=88)
subprocess.run([FF,'-y','-hide_banner','-loglevel','error','-framerate','25',
                '-i',str(tmp/'f%06d.jpg'),'-c:v','libx264','-crf','20',
                '-pix_fmt','yuv420p',str(OUT/'k48_negative_overlay.mp4')],
               capture_output=True, timeout=600)
for f in tmp.iterdir(): f.unlink()
tmp.rmdir()
print(f"\noverlay -> {OUT/'k48_negative_overlay.mp4'}  (RED=shahed FP, BLUE='not')")
