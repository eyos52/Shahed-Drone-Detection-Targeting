# Shahed Test Video Scraping — Plan

Companion to [test-video-dataset.md](test-video-dataset.md), which holds the test set design,
labeling protocol, and traps. This document is the collection pipeline. Nothing built yet.

Last updated: 2026-09-22 (smoke test: 12 clips downloaded, findings folded in)

---

## 1. The storage problem, and the resolution

Naive approach: download ~300 candidate news videos in full, keep them, push to Box.
That's ~18 GB local, and every byte of it has to come back down to be processed.

The pipeline below never materializes that. Two ideas do the work:

**(a) Never download a whole video.** `yt-dlp --download-sections` fetches only the timestamp
range we want. A 20-second insert out of a 3-minute news package is ~8 MB instead of ~60 MB.

**(b) The manifest is the source of truth, not the media.** Every raw download is *regenerable*
from a URL + timestamp + crop spec. So raw downloads are disposable scratch — they never go to
Box, and deleting them loses nothing.

What goes to Box is only what costs human effort to produce, or is expensive to re-fetch. That
is a small set, so the Box round-trip the team worried about happens once, for well under a
gigabyte, at eval time.

### Storage tiers

| Tier | Contents | Lives | Size | Regenerable? |
|---|---|---|---|---|
| 0 | Manifest, crop specs, hashes, labeling spec | **git** | KB | — (it *is* the source) |
| 1 | Raw downloads, triage proxies, intermediate cuts | **local scratch, deleted** | ~3 GB peak | Yes, from Tier 0 |
| 2 | Final curated clips + annotation exports | **Box** + labels in git | ~500 MB | Expensive (human curation) |
| 3 | Public dataset subsets (Halmstad, Anti-UAV) | **Box** | few GB | Expensive (large re-download) |

**Rule: nothing from Tier 1 is ever uploaded.** If it can be rebuilt by running the pipeline,
it is scratch.

### Size estimates

| Stage | Count | Each | Total |
|---|---|---|---|
| Discovery (metadata only) | ~300 URLs | — | ~0 |
| Triage proxies (360p sections) | ~300 | ~8 MB | ~2.4 GB *(transient)* |
| Harvested sections (1080p, survives triage) | ~120 | ~8 MB | ~1 GB *(transient)* |
| Final clips after leakage screen + crop | ~40 | ~12 MB | **~500 MB → Box** |
| Extracted frames, *if materialized* | 15,000 | ~300 KB | 4.5 GB — **avoid, see §6** |

Peak local footprint ≈ 3–4 GB, transient. Steady state under 1 GB.

**The real storage risk is the reference datasets, not the scraping.** UAV-Anti-UAV is 1.05M
frames — likely 100 GB+. Do not pull it whole. Subset-then-discard (§5, Stage 7).

---

## 2. Box mechanics

Use **rclone** with its Box backend, not Box Drive.

- Box Drive is a sync/streaming client: it keeps a **local cache that grows**, which defeats the
  purpose. If anyone uses it, mark the project folder online-only.
- rclone uploads straight from disk with no second local copy. `rclone move` uploads and deletes
  the local file in one step — exactly the Tier-1 discipline we want.

**Test auth early.** Box's rclone backend uses an OAuth browser flow (`rclone config`, possibly
`rclone authorize` from a machine with a browser). SMU SSO may add friction. This is a
5-minute task that could turn into a half-day surprise — do it in Stage 0, not at upload time.

Proposed Box layout, mirroring the shared project folder:

```
Shahed-Project/
  test-video/
    clips/            # Tier 2 — final curated clips
    annotations/      # CVAT exports (also mirrored to git; they're small)
    reference/
      halmstad/       # subset only
      anti-uav-rgbt/  # subset only
```

---

## 3. Directory layout (local)

```
repo/
  docs/                      # this plan, design notes
  manifests/
    candidates.csv           # Stage 1 output — every URL considered
    clips.csv                # the manifest; final curated set
    rejected.csv             # what was dropped and why (report artifact)
  scripts/                   # pipeline stages
  labels/                    # YOLO txt exports — small, git is fine

~/scratch/shahed/            # Tier 1. Outside the repo. Deletable at any time.
  proxies/  sections/  cuts/
```

---

## 4. Manifest schema (`manifests/clips.csv`)

| Field | Notes |
|---|---|
| `clip_id` | stable, e.g. `sh_night_ir_003` |
| `class` | `shahed` / `bird` / `airplane` / `helicopter` / `quad` / `empty` |
| `bucket` | maps to the composition table in the design doc |
| `source_url` | canonical page or video URL |
| `outlet` | CNN, Militarnyi, UA AF, Halmstad, … |
| `publish_date` | for event correlation and provenance |
| `section_start` / `section_end` | timestamps into the source |
| `crop` | ffmpeg crop spec (chyron removal), or empty |
| `native_res` / `native_fps` | recorded, **never normalized** — see §6 |
| `lighting` | `day` / `night` / `ir` — from auto-triage, human-confirmed |
| `sha256` | of the final clip |
| `phash_status` | `clean` / `dropped:<train_img_id>` |
| `license_note` | CC0, news/fair-use-research, etc. |
| `notes` | free text |

`rejected.csv` mirrors this plus a `reason` column. Keep it — "we screened 300 candidates and
dropped 47 for training overlap" is a sponsor-report artifact, not just bookkeeping.

---

## 5. Pipeline stages

### Stage 0 — Preflight (blocking, do first)

1. **Roboflow provenance.** Read the dataset page; inspect filenames for sequence naming.
2. **pHash self-cluster the 8k Shahed set.** Large tight near-duplicate clusters ⇒ video-derived
   ⇒ leakage screening is mandatory, not precautionary.
3. **Inspect box convention** — airframe, or airframe + engine glow? Test labels must match.
4. **Build the pHash index** of all 8k training images. Persist it; every later stage uses it.
5. **Test rclone Box auth.**

### Stage 1 — Discovery (no downloads)

Enumerate URLs + metadata only, via `yt-dlp --flat-playlist --print`.

- Multilingual seed queries: `шахед`, `збиття шахеда`, `нічна атака шахедів`,
  `мобільна вогнева група`, `Герань-2`, plus English `shahed night`, `shahed shot down thermal`.
- `yt-dlp "ytsearch100:<query>"` for bulk search enumeration.
- Channel sweeps: UA Air Force, brigade channels, Militarnyi, Defense Express, United24, CNN.
- **Mine existing Shahed projects** (EDTH-Warsaw, takzen, alexandre196, CEUR-WS paper) for source
  video URLs — a seed list *and* a leakage blocklist in one.
- Event-driven: find dated overnight-raid reports, then hunt that night's footage.

→ `manifests/candidates.csv`

### Stage 2 — Cheap triage (proxies only)

Pull a 360p proxy, or sample frames directly off the stream URL (`yt-dlp -g` piped to ffmpeg)
with no file written at all.

**Auto-classify lighting** with one ffmpeg pass:
```
ffmpeg -i X -vf "select='not(mod(n,50))',signalstats,metadata=print:key=lavfi.signalstats.YAVG" -f null -
```
Low `YAVG` ⇒ night. Low `SATAVG` ⇒ IR/thermal (near-zero chroma). Cheap, reliable, sorts the
whole candidate pool into the buckets we actually need.

**Human pass:** does it contain real aerial footage, or is it studio/maps/graphics? Mark
in/out timestamps for the insert. This is the irreducible manual step and it is the right
place to spend attention.

### Stage 2.5 — Candidate-level dedup (added after 2026-09-22 smoke test)

**Near-duplicate clips across channels are common in this domain.** The same unit-released
footage gets republished by multiple outlets with different watermarks, resolutions and time
offsets. In a 12-clip smoke test, 2 of 12 picks were the same footage (pHash Hamming **0**).

So pHash screening runs **twice**, against different references:

1. **Here (Stage 2.5)** — candidates against each other. Keep the highest-resolution,
   least-overlaid copy; drop the rest. Cheap, and it happens before we spend bandwidth.
2. **Stage 5** — survivors against the 8k training set.

Note the time-offset wrinkle: republished copies are often shifted by a few seconds, so
comparing frame-for-frame at matched timestamps produces false negatives. Hash **several**
timepoints per clip and take the **minimum** Hamming distance across all pairs.

### Stage 3 — Section harvest

```
yt-dlp --download-sections "*HH:MM:SS-HH:MM:SS" -f <best> <URL>
```

**Do not pass `--force-keyframes-at-cuts`.** It re-encodes, and at a default quality that is
visibly lossy — catastrophic for a small-target detection benchmark. Instead cut at the nearest
keyframe (stream copy, no re-encode) and **pad the range ±3 s**. Precise trimming happens in
Stage 4, where we are re-encoding anyway.

### Stage 4 — Shot split, crop, single encode

- Scene-cut split so a news package becomes individual shots: PySceneDetect, or
  `select='gt(scene,0.3)'`. Keep only the shots with aerial footage.
- Chyron/lower-third crop, per-outlet spec stored in the manifest.
- **Encode exactly once, at CRF 16–18.** Trim, crop, and encode in a single ffmpeg pass. Never
  re-encode a clip twice.

### Stage 5 — Leakage screen

pHash every 5th frame against the Stage 0 index. On any near-match (Hamming ≤ ~8):
**drop the entire clip**, not the matching frame — neighbors leak nearly as badly. Log to
`rejected.csv`.

Expect real attrition. This is why Stage 1 targets ~300 candidates for a ~40-clip set.

### Stage 6 — Publish

`rclone move` survivors to Box. Delete scratch. Commit the manifest. Upload a copy to CVAT for
annotation.

### Stage 7 — Reference datasets (parallel track)

Different discipline: these are *downloaded*, not scraped, and some are enormous.

| Dataset | Licence | Approach |
|---|---|---|
| Halmstad (Svanström) | **CC0-1.0** — free to use and edit, citation requested | Highest priority. Grab it, subset to the bird/airplane/helicopter clips at relevant DRI ranges, upload subset, delete the rest |
| Anti-UAV RGB-T (300-series) | Research use | 318 RGB-T pairs, day+night, IR+visible. Pull per-split if offered, not monolithic |
| UAV-Anti-UAV | Verify release status | 1.05M frames. **Never pull whole.** Fixed-wing subset only |
| Drone-vs-Bird (WOSDETC) | DUA required | Email `wosdetc@googlegroups.com` — human in the loop, send early |

**Subset-then-discard:** download → extract the ~20 clips needed → upload subset to Box →
delete the full archive. Check download size *before* starting each pull.

---

## 6. Gotchas

**Encode once.** Every re-encode destroys small-target detail — precisely the signal the model
is being tested on. Stage 3 stream-copies, Stage 4 encodes, and that is the only encode.

**Don't normalize frame rate.** Clips arrive at 24/25/30/60 fps. Resampling to a common rate
introduces duplicated or blended frames. Record `native_fps` and normalize *in the metrics*
instead — "false alarms per minute" and time-to-first-detection are computed from it.

**Don't upscale.** Record `native_res` and let the inference pipeline letterbox to the model's
input size. Upscaling a 480p news clip invents detail that isn't there and flatters the model.

**Don't materialize frames.** CVAT ingests video directly and indexes frames itself. Expanding
40 clips to 15,000 JPEGs costs 4.5 GB for no benefit.

**Labels go in git, not just Box.** Annotation exports are text and tiny. Version them.

**Do not install yt-dlp from PyPI.** Its index lagged by ~10 months when tested (latest
published 2025.10.14, vs 2026.08.19 actual). The stale version fails every YouTube download
with "The page needs to be reloaded."

- **macOS:** `brew install yt-dlp` — gives the current version. Simplest path.
- **Otherwise:** install from git master, on Python 3.10+. yt-dlp has deprecated 3.9, and the
  macOS system Python's LibreSSL warnings mask real errors:
  `python3.13 -m venv venv && ./venv/bin/pip install "yt-dlp[default] @ git+https://github.com/yt-dlp/yt-dlp"`

Both routes were verified at 2026.08.19.

**Don't trust titles for bucketing.** In the smoke test, 3 of 5 clips whose titles implied
daytime were actually IR/thermal. The YAVG/SATAVG pass (Stage 2) is not optional.

**Don't triage on a mid-point frame.** Roughly half the smoke-test clips had B-roll,
talking heads or split-screen composites at their midpoint. Scene-split first (Stage 4),
then triage the shots.

---

## 7. Sequencing

Stage 0 blocks everything — it decides whether scraped clips are usable at all, and what the
box convention must be. Stages 1–2 are cheap and parallelizable across the team. Stage 7's
email-gated requests (WOSDETC) should go out on day one regardless, since they have a human in
the loop.

| Order | Work | Blocking? |
|---|---|---|
| 1 | Stage 0 preflight (all five checks) | **Blocks 3–6** |
| 1 | Send WOSDETC request; start Halmstad download | No — do in parallel |
| 2 | Stage 1 discovery | No |
| 3 | Stage 2 triage | No |
| 4 | Stages 3–6 harvest → publish | After Stage 0 |
| 5 | Labeling per design doc §6 | After box convention settled |
