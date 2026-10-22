#!/usr/bin/env python3
"""Stage A -- fetch Roboflow Universe datasets, with provenance.

Reads the key from the environment (never a file, never a CLI arg that lands in
shell history):
    export ROBOFLOW_API_KEY=...

Usage:
    python fetch_roboflow.py --out ./pool --slugs slugs.txt
    python fetch_roboflow.py --out ./pool horus-al939/shahed-136-m2zkt:3

A slug is `workspace/project` with an optional `:version`; without one the
project's latest version is used. Each dataset lands in its own directory with a
provenance.json recording slug, version, licence, class histogram, image count
and the zip's sha256 -- that provenance is what makes later dedup and
clip-grouping auditable.

Disk is checked before every download and the run aborts rather than filling the
volume, because this is expected to run both on a laptop with a few GB free and
on Lustre scratch under quota.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

API = "https://api.roboflow.com"


def api_get(url, tries=3, timeout=60):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and i < tries - 1:
                time.sleep(2 ** i)
                continue
            raise
        except Exception:
            if i < tries - 1:
                time.sleep(2 ** i)
                continue
            raise


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def fetch(slug, version, out_root, key, fmt, min_free_gb):
    ws, proj = slug.split("/", 1)
    dest = out_root / f"{ws}__{proj}"
    prov = dest / "provenance.json"
    if prov.exists():
        print(f"  [skip] {slug} already fetched")
        return json.loads(prov.read_text())

    meta = api_get(f"{API}/{slug}?api_key={key}")
    p = meta.get("project") or {}
    if version is None:
        vs = [v.get("id", "").split("/")[-1] for v in (meta.get("versions") or [])]
        vs = [v for v in vs if v.isdigit()]
        if not vs:
            print(f"  [!!] {slug}: no generated version -- skipping "
                  f"(a version must be generated in the Roboflow UI first)")
            return None
        version = max(vs, key=int)

    ex = (api_get(f"{API}/{slug}/{version}/{fmt}?api_key={key}").get("export") or {})
    link, size_mb = ex.get("link"), ex.get("size")
    if not link:
        print(f"  [!!] {slug} v{version}: no export link")
        return None

    have = free_gb(out_root)
    need = (size_mb or 0) / 1024 * 2.2       # zip + extracted, with headroom
    if have - need < min_free_gb:
        print(f"  [STOP] {slug}: needs ~{need:.2f} GB, only {have:.2f} GB free "
              f"(floor {min_free_gb} GB). Aborting before filling the volume.")
        raise SystemExit(2)

    dest.mkdir(parents=True, exist_ok=True)
    zp = dest / "export.zip"
    print(f"  [get ] {slug} v{version}  {size_mb:.1f} MB", flush=True)
    t0 = time.time()
    with urllib.request.urlopen(link, timeout=600) as r, open(zp, "wb") as f:
        sha = hashlib.sha256()
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            sha.update(chunk)
    digest = sha.hexdigest()

    with zipfile.ZipFile(zp) as z:
        z.extractall(dest)
    zp.unlink()                               # keep only the extracted tree

    n_img = sum(1 for _ in dest.rglob("*")
                if _.suffix.lower() in (".jpg", ".jpeg", ".png"))
    rec = {
        "slug": slug, "version": str(version), "format": fmt,
        "license": p.get("license"), "classes": p.get("classes"),
        "images_reported": p.get("images"), "images_on_disk": n_img,
        "splits_reported": p.get("splits"),
        "export_size_mb": size_mb, "zip_sha256": digest,
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 1),
    }
    prov.write_text(json.dumps(rec, indent=1))
    print(f"         {n_img:,} images on disk in {rec['seconds']}s")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="*", help="workspace/project[:version]")
    ap.add_argument("--slugs-file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", default="yolov8")
    ap.add_argument("--min-free-gb", type=float, default=1.0,
                    help="abort rather than let free space drop below this")
    args = ap.parse_args()

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("ROBOFLOW_API_KEY not set in the environment")

    items = list(args.slugs)
    if args.slugs_file:
        items += [l.strip() for l in Path(args.slugs_file).read_text().splitlines()
                  if l.strip() and not l.startswith("#")]
    if not items:
        sys.exit("no slugs given")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"out={out}  free={free_gb(out):.2f} GB  datasets={len(items)}\n")

    got = []
    for it in items:
        slug, _, ver = it.partition(":")
        try:
            rec = fetch(slug, ver or None, out, key, args.format, args.min_free_gb)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  [err ] {slug}: {type(e).__name__}: {e}")
            continue
        if rec:
            got.append(rec)

    (out / "fetch_summary.json").write_text(json.dumps(got, indent=1))
    tot = sum(r["images_on_disk"] for r in got)
    print(f"\nfetched {len(got)} datasets, {tot:,} images, "
          f"{free_gb(out):.2f} GB still free")
    print(f"summary -> {out/'fetch_summary.json'}")


if __name__ == "__main__":
    main()
