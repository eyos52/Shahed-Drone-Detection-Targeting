#!/usr/bin/env python3
"""Score and order the candidate queue from text alone -- free, before any media.

This is a PRIORITISER, not a gate. Titles lie: the 2026-09-22 smoke test found
3 of 5 clips whose titles implied daylight were actually IR. So only the
obviously-dead rows are rejected outright; everything else is ranked and the
real verdict comes from the S1-S3 probes that actually look at pixels.

The scoring is built from what a 424-candidate Telegram sweep actually
contained, not from guesswork, and the noise was not what the original plan
assumed. Explainer videos are a YouTube problem; on Telegram the dominant
false positive is OTHER DRONE CONTENT -- FPV strike footage, naval drones,
Ukrainian attack drones hitting Russian vehicles. A keyword list of
"documentary / 3D / museum" does nothing about that, so three signals are
scored separately:

  target     the Shahed family by any of its names. Note Герань-4 and
             "реактивний дрон-камікадзе" -- the jet variants now dominate the
             traffic and neither matches a naive /shahed/ search.
  intercept  air-defence context: downing, interception, fighters, mobile fire
             groups. This is what separates "enemy drone being shot at" from
             "our drone attacking something", which is the actual problem.
  negative   FPV, naval drones, and post-impact wreckage. Wreckage is a real
             reject: it is a Shahed, but it is on the ground and there is no
             flying target to detect.

Dependencies: stdlib only.

Usage:
    python prefilter.py --out ../../manifests [--min-duration 4] [--apply]
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import CANDIDATE_FIELDS, read_rows, write_rows

# Stems only -- Ukrainian and Russian inflect heavily, so anchoring on endings
# loses most matches.
TARGET = [
    r"шахед", r"шахид", r"shahed",
    r"геран", r"герань", r"geran",          # Geran-2 / -3 / -4
    r"мопед",                                # common slang for the piston Shahed
    r"реактивн\w*\s+(?:бпла|дрон)",          # jet variants, the current majority
    r"дрон[-\s]?камік?адзе",
]
INTERCEPT = [
    r"збит", r"збива", r"збиття",            # downing
    r"перехоп", r"перехват",                 # intercept
    r"\bппо\b", r"\bпво\b", r"протиповітр",
    r"винищувач", r"\bf-?16\b", r"міг-?29", r"mig-?29",
    r"мобільн\w*\s+вогнев",                  # mobile fire group -- thermal M2s at night
    r"зеніт", r"ураж\w*\s+ціл",
]
NEGATIVE = [
    (r"\bfpv\b", "fpv"),
    (r"морськ\w*\s+дрон", "naval_drone"),
    (r"надводн\w*\s+дрон", "naval_drone"),
    (r"уламк", "wreckage"),                  # debris: real Shahed, but not airborne
    (r"наслідк", "aftermath"),
    (r"як\s+працює", "explainer"),
    (r"документальн", "explainer"),
    (r"анімац", "explainer"),
    (r"\b3d\b", "explainer"),
    (r"музе", "explainer"),
    # Fundraising appeals score high on every positive term -- "HELP US SHOOT
    # DOWN JET SHAHEDS, 100 MILLION FUNDRAISER" is target + intercept + urgency
    # and almost no usable footage. They were the top false positive in the
    # first 424-candidate run.
    (r"збір\s+на", "fundraiser"),
    (r"донат", "fundraiser"),
    (r"допоможіть", "fundraiser"),
    (r"реквізит", "fundraiser"),
    (r"monobank|монобанк", "fundraiser"),
    (r"приєднуйтесь\s+до\s+збор", "fundraiser"),
]

_T = [re.compile(p, re.I) for p in TARGET]
_I = [re.compile(p, re.I) for p in INTERCEPT]
_N = [(re.compile(p, re.I), tag) for p, tag in NEGATIVE]


def score(text):
    t = sum(1 for r in _T if r.search(text))
    i = sum(1 for r in _I if r.search(text))
    neg = sorted({tag for r, tag in _N if r.search(text)})
    return t, i, neg


def verdict(row, min_dur, max_dur):
    txt = row.get("title") or ""
    dur = row.get("duration_s")
    dur = int(dur) if str(dur).isdigit() else None

    if dur is not None and dur < min_dur:
        return "reject", f"too_short:{dur}s", 0
    t, i, neg = score(txt)
    # These beat a target mention outright. Wreckage and aftermath: the airframe
    # is on the ground, there is nothing airborne to detect. Fundraiser: scores
    # maximally on every positive term and is a person talking to camera.
    if "fundraiser" in neg:
        return "reject", "fundraiser", 0
    if neg and ("wreckage" in neg or "aftermath" in neg) and i == 0:
        return "reject", ";".join(neg), 0
    if t == 0 and i == 0:
        return "reject", "no_target_terms", 0
    if neg and t == 0:
        return "reject", ";".join(neg), 0

    rank = 3 * t + 2 * i - 2 * len(neg)
    # Long-form is almost always a podcast, news roundup or explainer: it scores
    # well on vocabulary and contains little usable footage. The YouTube leg
    # surfaced an 88-minute podcast at rank 3 before this. A clip worth
    # annotating is 10-20s, so length is a strong negative, not a tiebreak.
    if dur is not None and dur > max_dur:
        rank -= 6
        if t and i:
            return "maybe", f"long_form:{dur}s", rank
    if t and i:
        return "keep", "", rank         # named target + air-defence context
    return "maybe", ("target_only" if t else "context_only"), rank


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="manifests/ directory")
    ap.add_argument("--min-duration", type=int, default=4)
    ap.add_argument("--max-duration", type=int, default=300)
    ap.add_argument("--apply", action="store_true",
                    help="write verdicts back to candidates.csv (default: dry run)")
    ap.add_argument("--rescore", action="store_true",
                    help="re-evaluate rows already marked prefiltered. Needed "
                         "whenever the rules change: without it a row keeps the "
                         "verdict it got under the OLD rules, which is how a "
                         "4760s video stayed 'keep' after long-form was added.")
    ap.add_argument("--show", type=int, default=12)
    a = ap.parse_args()

    path = Path(a.out) / "candidates.csv"
    rows = read_rows(path, "candidate_id")
    if not rows:
        print(f"no candidates at {path}")
        return

    scored, counts = [], {}
    for r in rows.values():
        v, why, rank = verdict(r, a.min_duration, a.max_duration)
        if a.rescore or r.get("stage") in ("discovered", "", None):
            r["verdict"], r["reject_reason"] = v, why
            r["stage"] = "prefiltered"
        counts[v] = counts.get(v, 0) + 1
        if why:
            counts[f"  reason:{why}"] = counts.get(f"  reason:{why}", 0) + 1
        scored.append((rank, v, r))

    print(f"{len(rows)} candidates")
    for k in sorted(counts, key=lambda k: (k.startswith("  "), -counts[k])):
        print(f"  {k:<28}{counts[k]}")

    scored.sort(key=lambda x: -x[0])
    print(f"\n--- top {a.show} by rank ---")
    for rank, v, r in scored[:a.show]:
        print(f"  [{rank:>3}] {v:<6} {r['duration_s']:>4}s  {r['title'][:84]}")

    if a.apply:
        write_rows(path, CANDIDATE_FIELDS, [r for _, _, r in scored])
        print(f"\nwrote verdicts to {path} (ordered by rank)")
    else:
        print("\ndry run -- pass --apply to write verdicts")


if __name__ == "__main__":
    main()
