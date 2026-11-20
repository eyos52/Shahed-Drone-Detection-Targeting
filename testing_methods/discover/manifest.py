#!/usr/bin/env python3
"""Manifest schema for the discovery spine -- the source of truth, not the media.

Every candidate is regenerable from url + timestamps, so the CSVs here are the
only thing that must survive. Three of them:

  channels.csv    the snowball registry. Discovery walks channels, never dates,
                  because enumerating a channel's whole history is lag-immune:
                  footage of a March raid posted in September still shows up.
  candidates.csv  one row per discovered video, carrying every screening score
                  so a verdict can always be traced back to the number that
                  produced it.
  rejected.csv    candidates.csv rows that lost, plus a reason. Kept on purpose
                  -- "screened 300, dropped N for training overlap" is a
                  sponsor-report artifact, not bookkeeping.

Writes are upserts keyed on a stable id, so every stage can re-run without
duplicating rows and a half-finished sweep can be resumed.

Dependencies: stdlib only.
"""
import csv
import re
from pathlib import Path

CANDIDATE_FIELDS = [
    # identity
    "candidate_id", "platform", "channel", "url", "posted_date",
    "title", "duration_s", "native_res", "native_fps",
    # verdict
    "stage", "verdict", "reject_reason",
    # S1 sparse frame probe
    "sky_frac", "graphics_score", "scene_variety", "yavg", "satavg", "lighting",
    # S2 motion probe
    "motion_persistence", "motion_straightness", "n_tracks",
    # S3 audio
    "audio_score",
    # C dedup + leakage
    "dup_group", "dup_keep", "leak_status",
    # event ledger cross-reference
    "event_date", "event_ref",
]

CHANNEL_FIELDS = [
    "handle", "platform", "channel_class", "preview_available",
    "enumerated_through", "last_seen_id", "n_found", "added_by", "notes",
]

# Minute-resolution, not nightly counts. The Air Force Command channel (kpszsu)
# posts every incursion as it happens -- "21:36 jet UAV heading for Dnipro" --
# which correlates with footage far better than a morning total would.
EVENT_FIELDS = ["event_id", "ts_utc", "source", "threat_type", "emoji",
                "location", "text", "n_clips"]

# platform -> how its native id is spelled, for stable candidate_id
_SLUG = re.compile(r"[^A-Za-z0-9_.-]+")


def candidate_id(platform, channel, native_id):
    """Stable across re-runs: the same video always lands on the same row."""
    return f"{platform}:{_SLUG.sub('-', channel or '?')}:{_SLUG.sub('-', str(native_id))}"


def blank_candidate(**kw):
    row = {k: "" for k in CANDIDATE_FIELDS}
    row["stage"] = "discovered"
    row["verdict"] = "pending"
    row.update({k: v for k, v in kw.items() if k in CANDIDATE_FIELDS})
    return row


def read_rows(path, key):
    """-> OrderedDict keyed on `key`. Missing file is an empty manifest."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r[key]: r for r in csv.DictReader(f) if r.get(key)}


def write_rows(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    tmp.replace(path)          # atomic: a killed sweep never truncates the manifest


def upsert(path, fields, key, new_rows):
    """Merge new_rows into the manifest. Existing non-empty values win, so a
    re-run of discovery never clobbers screening scores written later."""
    cur = read_rows(path, key)
    added = 0
    for r in new_rows:
        k = r.get(key)
        if not k:
            continue
        if k in cur:
            for f, v in r.items():
                if v not in ("", None) and not cur[k].get(f):
                    cur[k][f] = v
        else:
            cur[k] = r
            added += 1
    write_rows(path, fields, cur.values())
    return added, len(cur)
