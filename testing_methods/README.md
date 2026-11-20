# testing_methods

Test-set collection and model evaluation. Self-contained: everything here is
code and manifests, no media. The media it refers to is regenerable from these
manifests (see **Regenerating**), or lives in Box.

---

## Why this exists

The repository's reported metrics (`models/final_v7i_1280_e60/`, shahed
mAP50 0.845) are **validation** metrics, not test metrics. Three findings,
each independently checkable from files in this repo:

**1. There is no held-out test set.** The directories named `test_best_1280/`,
`test_last_1280/` and `test_best_shahed_1280/` contain the same ground truth as
the training-time validation run. Summing the ground-truth columns of all four
committed confusion matrices:

| run | bird | not | shahed | total | shahed misses |
|---|---|---|---|---|---|
| v5i val | 614 | 561 | 774 | 1,949 | 121 |
| v5i "test" | 614 | 561 | 774 | 1,949 | 121 |
| v7i val | — | 1,175 | 774 | 1,949 | 121 |
| v7i "test" | — | 1,175 | 774 | 1,949 | 121 |

Identical. Likely cause: the training `data.yaml` has no `test:` key, and
Ultralytics' `split='test'` silently falls back to `val` when it is absent.
`best.pt` and `best_shahed.txt` ("epoch 58") were also *selected* on that set,
so those numbers are a best-of-60, not an unbiased estimate.

**2. The shahed class is ~99% synthetic.** Counting by source in the Roboflow
export `e-yjnj4/shahed-y4fsd` v5:

| split | shahed-labelled | synthetic | real | unique real sources |
|---|---|---|---|---|
| train | 5,453 | 5,409 (99.2%) | 44 | **15** |
| valid | 530 | 528 (99.6%) | 2 | 2 |
| test | 258 | 257 (99.6%) | 1 | 1 |

Renders carry a generated filename (`<20-28 random letters>0000_png.rf.<hash>`)
matching the `shahed_dataset_1_synthetic` source in the original Colab
pipeline. Verified visually: the 24 darkest shahed training images are all CGI
on procedural cloud backdrops, several showing the backdrop's quadrilateral
edge in frame. So the quoted 0.845 measures CGI detection, on a validation
split that is itself 99.6% CGI.

**3. Train/validation leakage in the distractor classes.** 41.6% of validation
and 44.7% of test images belong to a source clip that also has frames in train
(83 named clip groups, e.g. `BirdsFlying` 1755/167/92). Confirmed duplicate
source images across splits, flip-invariant dHash on pixel content: 148
train<->valid, 76 train<->test, 18 valid<->test.

---

## First held-out evaluation

11 clips collected from Telegram/YouTube *after* the model was trained, hand
annotated in CVAT. No shared frames, source clips or platform with training.
3,399 frames, 1,752 shahed instances, 1,647 frames annotated empty.

| metric | held-out | repo quotes |
|---|---|---|
| shahed mAP50 | **0.091** | 0.845 |
| shahed mAP50-95 | **0.030** | 0.700 |
| precision | 0.275 | 0.993 |
| recall @ IoU 0.5 | 0.235 | 0.844 |
| recall @ IoU 0.1 | 0.506 | — |

**Caveat, stated deliberately:** annotation quality on this set is imperfect.
Recall more than doubles (0.235 -> 0.506) when IoU is loosened from 0.5 to 0.1,
which means many detections land near but not tightly on the target — some of
that is loose labelling, not model error. The honest range is 0.09-0.5, not a
point estimate.

False alarms are not affected by label tightness: **604 false boxes on 1,647
frames annotated empty (36.7%)**, roughly 550 per minute of empty sky at 25fps.

**Negative control** (`k48`, a fighter-jet clip with zero shahed, 263 frames):
the model classified the aircraft as `not` in 82% of frames and produced only
**3** shahed false positives (1.1%), two of them below 0.4 confidence. It
rejects a large resolved aircraft well; the 550/min figure came from frames
containing nothing at all.

---

## Pipeline

### discover/ — find candidate footage (metadata only, no downloads)

Discovery walks *channels*, never dates: enumerating a channel's whole history
is lag-immune, so footage of a March raid posted in September is still found.
Dates are a coverage ledger, not a query.

| module | role |
|---|---|
| `manifest.py` | CSV schema + atomic upserts (existing values win, so re-runs never clobber scores) |
| `channels.py` | channel registry, reachability probe, sweep driver |
| `telegram.py` | `t.me/s/<channel>` HTML enumeration. Preview is opt-in per channel: 15/22 reachable |
| `youtube.py` | yt-dlp channel sweeps + search; `--hydrate` fills the thin flat-playlist metadata |
| `x.py` | **resolver only** — single statuses work without auth, timeline enumeration is auth-walled |
| `news.py` | RSS / WordPress REST adapters. Low yield: 1 of 3 outlets serves a readable feed |
| `events.py` | raid ledger from the Air Force Command channel (`kpszsu`), minute resolution |
| `prefilter.py` | free text scoring. `--rescore` re-evaluates rows scored under older rules |
| `triage.py` | fetch 360p proxies + contact sheets |
| `segment.py` | **shot detection + per-window target scoring** |
| `cut_segments.py` | cut the flagged windows into watchable clips |

### segment.py — the part that saves the most time

Turns "a 5-minute news package with 10 good seconds" into rows pointing at
`02:14-02:26`. Three things it does that the obvious approach does not:

- **Spatial saliency, not frame differencing.** Most of this footage is
  camera-*tracked*; the target barely moves in frame, so a temporal median
  absorbs it into the background. Measured: a clip with a visible Shahed in
  every sampled thumbnail scored `moving_frac` 0.0005 and was rejected.
- **Short windows, not whole shots.** Persistence averaged over a 47s shot
  buries a signal present for 10s of it.
- **Static-overlay suppression.** A burnt-in watermark is a bright isolated
  pixel-locked peak that wins argmax on every frame and masks the real target.
  Subtracting the temporal median of the saliency map removes it.

Known limit: the score **localizes** well but **ranks** poorly. On the first
review pass, 11 of 12 human-selected clips scored 0.20-0.25 and only one came
from the top of the ranking. Review the whole range; do not trust the ordering.

### eval/ — measure the model

| module | role |
|---|---|
| `evaluate.py` | ultralytics metrics + per-video recall + false alarms on empty frames |
| `overlay.py` | burns ground truth (green) and predictions (orange) into the same frame; sweeps recall vs IoU |
| `negtest.py` | pure-negative clip: every detection is wrong by construction |

---

## Current state

| | |
|---|---|
| candidates discovered | 4,379 |
| after prefilter | 94 keep + 391 maybe |
| clips downloaded | 371 |
| windows scored | 14,211 across 361 videos |
| cut for review | 237 (31 min) |
| annotated so far | 12 |

Yield notes for planning: the Telegram leg loses ~30% of posts to
`telegram:embed` extraction failures — **not** link rot, the rate is flat
across every month from 2025-06 to 2026-11. Human pick rate on cut segments
was ~14%.

---

## Regenerating what is not in git

No media is committed. `triage/clips` alone is 2.9 GB and every file in it is
reproducible from `manifests/candidates.csv`:

```bash
# re-fetch proxies for everything the prefilter kept
python discover/triage.py --out manifests --dest $SCRATCH/triage \
    --verdict keep,maybe --limit 500 --max-duration 1200

# score shots and windows
python discover/segment.py --clips $SCRATCH/triage/clips --out $SCRATCH/segments

# cut the flagged windows into watchable clips
python discover/cut_segments.py --segments $SCRATCH/segments/segments.csv \
    --clips $SCRATCH/triage/clips --out $SCRATCH/segments/cuts \
    --min-score 0.15 --best-per-video
```

Curated clips, review cuts, overlay videos and evidence sheets are in Box.

**Why media is not in git:** this repository is public, and the footage is
third-party news and OSINT video. `docs/scraping-plan.md` 7.4 already says to
ship a manifest plus a fetch script. The current repo shows why: the 13
original evaluation videos in `result_videos/` were committed *without* a
manifest, and all 13 source tweets now return HTTP 404. The media survived;
the provenance did not, and it cannot be reconstructed.

---

## Dependencies

`yt-dlp`, `imageio-ffmpeg`, `numpy`, `Pillow` for discovery and segmentation;
`ultralytics` for evaluation. No scipy (not available on M3), so the blur,
connected-component and line-fit code is numpy by hand.
