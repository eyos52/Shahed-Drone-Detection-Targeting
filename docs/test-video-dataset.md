# Test Video Dataset — Design & Collection Notes

Working notes for the held-out **video** evaluation set. Nothing here is built yet; this
records decisions made and traps identified so far so we don't re-derive them.

Collection pipeline lives in [scraping-plan.md](scraping-plan.md).

Last updated: 2026-09-22

---

## 1. What's decided

| | |
|---|---|
| **Training data** | AOD4 (22k) + Roboflow Shahed set (8k), combined, trained in one pass — no separate fine-tuning stage |
| **Class scheme** | 2 classes: `other`, `shahed` |
| **Balance** | ~22k `other` : ~8k `shahed` ≈ 2.75:1. Acceptable; no resampling planned |
| **Test data** | **Video only.** Still-image metrics do not measure what this model is for |
| **Base model** | TIDL YOLOv5-lite → ONNX export for TI Edge AI Studio / TIDL compilation |

> Note: `Dataset_Overview.html` in the repo root describes a *different* combined dataset
> (10,031 images, 4 sources, heavily synthetic). That notebook does not reflect the current
> plan. Do not use its composition numbers.

---

## 2. Operational scenario (drives everything below)

The model cues an interceptor against incoming one-way attack drones. The economics are the
point: currently ~$2M interceptors are spent on ~$50k Shaheds, and the goal is to enable
much cheaper kill mechanisms. That implies:

- **Engagement geometry** — look-up-at-sky from ground, or air-to-air tail-chase. Not
  downward-looking, not ground-clutter.
- **Range matters** — detection range buys the interceptor its firing solution. Small
  targets (sub-30px) are the operationally important case, not near passes.
- **Night is the norm.** Shahed raids are predominantly night operations. Ukrainian mobile
  fire groups stand night combat duty with thermal-equipped Browning M2s; interceptor drones
  (e.g. Sting) publish thermal-camera kill footage.

**Consequence: night / IR / thermal is the primary positive bucket, not a secondary one.**

---

## 3. Test set composition

Target ~40 clips, 10–20s each, ≈10 minutes total. Annotate at 2–5 fps with keyframe
interpolation (CVAT), not every frame.

### Positives — Shahed

| Bucket | Clips | Measures |
|---|---|---|
| Night / IR / thermal, ground-up | 8 | The primary operational case |
| Night, visible (engine glow, tracer, searchlight) | 5 | Most common civilian/AD footage condition |
| Air-to-air (interceptor, F-16, Yak-52, helicopter) | 4 | Seeker-POV analog |
| Small & distant (<30 px) | 4 | Detection range |
| Day EO | 3 | Baseline recall |
| Cluttered background (treeline, urban, cloud) | 2 | Background bias |

### Negatives — distractors

| Bucket | Clips | Measures |
|---|---|---|
| Birds, single + flocks, at Shahed-like apparent size | 6 | **The hard negative.** A 20px Shahed and a 20px bird are the same blob |
| Fixed-wing aircraft / airliners | 4 | Delta-wing confusion |
| Helicopters | 3 | AOD4 folds these into `other` — verify it held |
| Quadcopter / FPV drones | 4 | Explicit SOW requirement (status report slide 4) |
| Empty sky / clouds / moving camera | 3 | False alarms per minute with no target present |

The empty-sky and bird buckets are the ones usually skipped and the ones that actually
reveal failure. A detector biased by sky-heavy training typically either fires on every bird
or misses everything small — aggregate mAP hides both.

---

## 4. Sources

| Class | Source | Status |
|---|---|---|
| Shahed positives | News (CNN, Reuters, AP) + Ukrainian mil/OSINT originals | **To scrape** |
| Birds / airplanes / helicopters | Halmstad multi-sensor drone detection dataset (Svanström) | To request |
| IR / thermal negatives | Anti-UAV410 (IR-only) or Anti-UAV300/RGBT (RGB+IR) | Open download |
| Quad / FPV negatives | Halmstad + Anti-UAV | To request |

### Why news footage for positives

Anti-UAV and Halmstad are commercial quadcopters on controlled test flights at Swedish
airports — wrong airframe, wrong range profile, wrong lighting. News and OSINT footage is
real Shaheds under real engagement conditions, much of it the night/IR imagery that matches
the use case. For the positive class it is the closest available data, not a compromise.

### Halmstad — highest leverage item

650 videos (365 IR + 285 visible), 10s each, 203,328 annotated frames, classes **drone /
bird / airplane / helicopter**, split by DRI range category (Close / Medium / Distant).
Published via *Data in Brief*, **CC0-1.0** (free to download, use and edit; citation
requested). Covers the entire negative side, video-native and already annotated — no scraping,
no labeling.
<https://www.sciencedirect.com/science/article/pii/S2352340921007976>

### Existing Shahed video datasets: none (checked 2026-09-22)

Searched arXiv, Hugging Face, Roboflow Universe, and counter-UAS benchmark literature.
**No annotated Shahed video dataset is publicly available.** Everything Shahed-specific is
images or trained models:

| What | Why it doesn't serve |
|---|---|
| Roboflow sets (`shahed136-detect`, MyWS `shahed-136`, others) | Still images — one of these is likely our 8k |
| Rendered/synthetic Shahed sets (YOLO boxes + masks) | Synthetic, images |
| EDTH-Warsaw YOLO12 detector, takzen YOLO11, alexandre196 YOLOv8 | Trained models, no eval video released |
| CEUR-WS counter-UAS paper (Vol-4048/paper25) | Built ~1000 Shahed images *extracted from video* for training; video never released |

**Conclusion: scraping is required for the positive class.** Proceed with the scraping plan.

**Mine the above projects first.** Several were built from frame-grabs of the same news/OSINT
footage we intend to scrape. Their repos and papers often list source video URLs — that list is
simultaneously (a) a seed list of candidate clips and (b) a blocklist of footage that may
already be inside our Roboflow 8k. See §6.1.

### Better negatives — fixed-wing (found 2026-09-22)

Shahed is a delta/fixed-wing airframe, but nearly every anti-UAV dataset is quadcopters.
Fixed-wing distractors are the genuinely hard discrimination — harder than bird vs Shahed.

- **UAV-Anti-UAV** (arXiv 2512.07385) — 1,810 video sequences, 1.05M frames, 9.85 hours;
  categories include fixed-wing, VTOL, FPV, multirotor, helicopter. RGB only. Release stated as
  "will be available" at `github.com/983632847/Awesome-Multimodal-Object-Tracking` — **verify**.
- **Anti-UAV RGB-T (300-series)** — 318 RGB-T video pairs spanning **day and night, IR and
  visible**. Better for night negatives than the 410 set (which is IR-only).

### Drone-vs-Bird (WOSDETC) — correction

DvB **is** video. Its annotation format is `framenum num_objs obj1_x obj1_y obj1_w obj1_h
obj1_class ...`, one line per frame of the matching video. It looks empty because there is
no download button: you must email `wosdetc@googlegroups.com` and sign a data usage
agreement. Human in the loop → request early.
<https://github.com/wosdetc/challenge>

---

## 5. Finding night footage

English-language search surfaces daytime studio packages. Night footage is reachable three
other ways:

**Search in Ukrainian/Russian.** `Шахед`, `збиття шахеда`, `нічна атака шахедів`,
`мобільна вогнева група`, `Герань-2` / `Geran-2`.

**Search by content type, not by keyword** — these categories are inherently night:
- Mobile fire groups (`мобільні вогневі групи`) — thermal-sighted M2s, night combat duty
- Interceptor drone kill footage — thermal seeker POV, closest analog to our use case
- Air defense EO/IR feeds and gun-camera footage
- Civilian dashcam / phone footage of the engine glow during overnight barrages

**Search by event.** Large raids are dated, nameable events. Find the date in news archives,
then find that night's footage. Gives clustered, verifiable, dateable clips.

**Outlets that host the raw clip** (rather than a studio re-cut): Militarnyi, Defense Express
(en.defence-ua.com), United24 Media, Ukrinform, Euromaidan Press, plus Ukrainian Air Force
Command and individual brigade channels.

**Automated night triage.** Can't filter by "night" at the search layer, so over-collect and
auto-bucket: sample frames with ffmpeg, compute mean luma and mean saturation. Low luma →
night. Near-zero chroma variance → IR/thermal. Cheap and reliable.

---

## 6. Labeling protocol

Never label frame-by-frame. Three layers:

1. **Keyframe + interpolate.** CVAT track mode — a box every 10–30 frames, linear interpolation
   fills the rest. 10 min @ 25fps = 15,000 frames, but only ~300–600 boxes placed by hand.
2. **Click-and-propagate.** SAM 2 video object segmentation from a single click, propagated
   across the clip; convert masks to boxes. Substantially better than classic trackers
   (CSRT/KCF) on small low-contrast targets. Biggest single time-saver.
3. **Model-in-the-loop — refinement only.** Pre-annotating with our own model after the first
   training run is fine for correction, but it will not propose boxes where it misses, silently
   biasing labels toward the model's blind spots. Never for initial labeling; always sweep for
   missed targets by hand.

Tooling: CVAT (self-hosted or cvat.ai free tier) — track mode, interpolation, built-in trackers,
viewing-only brightness/contrast, YOLO export. Label Studio is the alternative.

### 6.1 Night clips

- **Scrub, don't stare.** A night Shahed is often a ~5px dot; on a frozen frame it is
  indistinguishable from sensor noise. Play back and forth — coherent motion is the
  discriminator. **Most important night rule.**
- **Enhance the view, not the data.** Use the tool's brightness/contrast/gamma so the annotator
  can see; store labels against the **original** frames. Labeling and evaluating on enhanced
  frames builds a test set that doesn't match deployment.
- **Frame-difference as a finding aid.** On static-camera clips, temporal median background
  subtraction makes a moving dot pop. Use it to *locate*, then label on the original.
- **Use the audio.** The moped engine is distinctive and audible in most civilian/AD footage —
  it brackets when the drone is in frame. Tracer fire and searchlights are the same kind of cue.
- **Flag uncertainty.** `difficult`/`ignore` attribute for genuinely uncallable frames. Score
  twice — once counting them, once excluding. Otherwise ambiguous frames read as clean misses.

### 6.2 Box convention — decide before anyone labels

**Does the box cover the airframe, or the airframe plus engine glow?** At night the exhaust glow
is frequently larger and brighter than the airframe. If the Roboflow training set boxes the
airframe and our test labels box the glow, IoU craters and we will misread a labeling-convention
mismatch as a detection failure. **Inspect the Roboflow annotations first and match them.**

### 6.3 Inter-annotator agreement

Four people labeling means a written spec or the metrics are noise. Specify: entry criteria
(first frame where any part visible? where >=4px?), minimum size below which we don't label,
occlusion and out-of-frame handling, glow convention (§7.2).

**Double-label a 10% subset across two annotators and measure IoU agreement.** Below ~0.7 on
night clips, the test set cannot separate model error from label noise and every downstream
number is unfalsifiable.

---

## 7. Traps to watch

### 7.1 Train/test leakage — highest priority

Our Shahed training set is from Roboflow. Shahed stills on Roboflow and Kaggle are very
commonly **frame-grabs from exactly the news and OSINT videos we are about to scrape.** If a
test clip's frames are already in training, the numbers are meaningless and it is invisible
from the outside.

**Check before scraping anything:**
1. Read the Roboflow dataset page for stated provenance and source links.
2. Inspect filenames — video-derived sets leak sequence naming (`frame_0042_jpg.rf.<hash>.jpg`).
3. **Diagnostic:** pHash-cluster the 8k set *against itself*. Video frame-grabs produce large
   tight near-duplicate clusters. A genuinely photographic set does not. This tells us the
   answer even if Roboflow states no provenance.

**Mitigation:** pHash-index all 8k training images; pHash every candidate test frame against
it; on any near-match, **drop the whole clip**, not just the matching frame — adjacent frames
leak nearly as badly. Expect this to kill a meaningful fraction of candidates, so collect
2–3× more than needed.

### 7.2 Status report R-03 contradicts the current plan

Slide 5, risk R-03 mitigation reads: *"fine-tune on authentic AOD4 and web-scraped Shahed
combat video stills"* — i.e. web-scraped footage into **training**. Current plan is
web-scraped video into **testing**, with no fine-tuning stage. Both are defensible; pick one
per video and never both from the same clip. Resolve before Checkpoint 1 (Sep 27) so the risk
register matches reality.

### 7.3 News packaging artifacts

Crop the lower-third / chyron band and drop studio, map, and graphics shots during triage, so
we evaluate on the insert footage rather than on the network's compositing. Where the upstream
original is reachable, prefer it — but don't burn days chasing geoblocked or deleted Telegram
links when a usable news copy exists.

### 7.4 Don't commit video to git

Ship a manifest (URL, timestamps, source, license, contents) plus a fetch script — the way
Kinetics and AVA distribute. Keeps the repo small and sidesteps redistribution entirely.

### 7.5 Ask the sponsor

Status report risk R-02 — *"real video testing data of Shaheds is insufficient or unvaried"* —
is owned by Muxin/Sandra, **due Sep 24**. TI may already have evaluation footage. One email is
worth more than a day of scraping.

---

## 8. Metrics

Report **per bucket**, never aggregate:

- Recall on positive clips
- **False alarms per minute** on negative clips
- Time-to-first-detection after the target enters frame
- ID switches per track, if ByteTrack is wired up (status report R-06)

"mAP 0.87" will not survive a sponsor question. "Detects 6/6 day clips, 1/5 IR clips, 4 false
alarms/min on bird footage" says exactly what to fix.

---

## 9. Open items

| # | Item | Owner | Status |
|---|---|---|---|
| 1 | Check Roboflow 8k Shahed set provenance (§7.1) | — | Open, blocks scraping |
| 2 | Request Halmstad dataset | — | Open |
| 3 | Email WOSDETC for DvB access (human in loop) | — | Open |
| 4 | Ask Chris/TI for existing evaluation footage (R-02, due Sep 24) | Muxin / Sandra | Open |
| 5 | Resolve R-03 train-vs-test contradiction (§7.2) | — | Open, before Sep 27 |
| 6 | Inspect Roboflow box convention: airframe vs airframe+glow (§6.2) | — | Open, blocks labeling |
| 7 | Verify UAV-Anti-UAV release status at the stated repo | — | Open |
| 8 | Mine existing Shahed projects for source video URLs (seed + blocklist) | — | Open |
| 9 | Write labeling spec + run 10% double-label agreement check (§6.3) | — | Open |
| 10 | Test rclone Box OAuth (SMU SSO may add friction) | — | Open, blocks upload |
| 11 | Build scraper + dedupe pipeline | — | Planned, see scraping-plan.md |
