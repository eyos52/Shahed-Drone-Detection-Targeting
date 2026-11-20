# annotations

`labels.csv` — all CVAT annotations in one file, 3,662 rows across 12 clips.

    clip,frame,cls,cx,cy,w,h

`cls` is empty for frames annotated as containing no target. Those are real
negatives asserted by the annotator, not unlabelled frames: boxed frames form
one contiguous run per clip (Track mode with interpolation), so a gap means
"target not present".

Coordinates are YOLO normalised (cx, cy, w, h) against the source clip's own
resolution. Clip tags map to `keepers/` in Box.

| clip | frames | boxes | notes |
|---|---|---|---|
| k01 | 161 | 76 | |
| k11 | 420 | 1 | near-pure negative |
| k14 | 350 | 324 | |
| k15 | 421 | 84 | |
| k24 | 280 | 265 | |
| k29 | 300 | 300 | target present throughout |
| k33 | 300 | 137 | |
| k42 | 350 | 255 | |
| k44 | 218 | 27 | |
| k46 | 344 | 28 | |
| k48 | 263 | **0** | **negative control** — fighter jet, no shahed |
| k84 | 255 | 255 | target present throughout |

Annotation quality is uneven — recall against this set more than doubles when
IoU is loosened from 0.5 to 0.1, so some boxes are loose. Treat per-frame IoU
with caution; frame-level presence/absence is reliable.

The original CVAT exports (`job_*_yolo 1.1.zip`) are not committed — the repo
root `.gitignore` excludes `*.zip`. They are in Box alongside the clips.
