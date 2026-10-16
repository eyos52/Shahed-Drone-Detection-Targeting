#!/usr/bin/env python3
"""Local backend for the Shahed dashboard."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
DASHBOARD_DIR = ROOT / "dashboard"
UPLOAD_DIR = ROOT / "outputs" / "uploads"
DETECTIONS_DIR = ROOT / "outputs" / "detections"
VIDEO_OUTPUT_DIR = ROOT / "outputs" / "videos"
MODEL_PATH = ROOT / "models" / "shahed_ft_real_v4" / "weights" / "best.pt"

sys.path.insert(0, str(ROOT / "scripts"))
from run_inference import run_inference  # noqa: E402

# Run inference on a compute node via srun instead of on the login node.
# Override with e.g. SLURM_PARTITION=batch, SLURM_TIME=00:30:00, USE_SLURM=0.
USE_SLURM = os.environ.get("USE_SLURM", "1") != "0" and shutil.which("srun") is not None
SLURM_PARTITION = os.environ.get("SLURM_PARTITION", "short")
SLURM_TIME = os.environ.get("SLURM_TIME", "00:30:00")

app = Flask(__name__, static_folder=str(DASHBOARD_DIR), static_url_path="")


def run_inference_on_node(model_path, video_path, conf, imgsz, out_json):
    """Submit run_inference.py as a blocking srun job and return its payload."""
    cmd = [
        "srun", f"--partition={SLURM_PARTITION}", "--gres=gpu:1",
        "--cpus-per-task=4", "--mem=16G", f"--time={SLURM_TIME}",
        "--job-name=shahed-infer", "--chdir=" + str(ROOT),
        sys.executable, str(ROOT / "scripts" / "run_inference.py"),
        "--model", str(model_path), "--video", str(video_path),
        "--out-json", str(out_json), "--conf", str(conf), "--imgsz", str(imgsz),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip()[-2000:] or "srun failed")
    return json.loads(Path(out_json).read_text())


def slugify(name):
    stem = Path(name).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return stem or "uploaded-video"


@app.get("/")
def index():
    return send_from_directory(DASHBOARD_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "model_exists": MODEL_PATH.exists(),
        "model": str(MODEL_PATH.relative_to(ROOT)) if MODEL_PATH.exists() else None,
    })


@app.post("/api/infer")
def infer():
    if "video" not in request.files:
        return jsonify({"error": "Upload a video file first."}), 400
    if not MODEL_PATH.exists():
        return jsonify({"error": "Model file missing: models/shahed_best.pt"}), 400

    video_file = request.files["video"]
    if not video_file.filename:
        return jsonify({"error": "The uploaded video has no filename."}), 400

    try:
        confidence = float(request.form.get("conf", "0.7"))
    except ValueError:
        return jsonify({"error": "Confidence must be a number."}), 400
    confidence = max(0.01, min(confidence, 0.99))

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filename = secure_filename(video_file.filename) or "uploaded-video.mp4"
    video_path = UPLOAD_DIR / filename
    video_file.save(video_path)

    output_stem = slugify(filename)
    output_json = DETECTIONS_DIR / f"{output_stem}.json"

    try:
        if USE_SLURM:
            payload = run_inference_on_node(MODEL_PATH, video_path, confidence, 640, output_json)
            copied_annotated = None
        else:
            payload, copied_annotated = run_inference(
                model_path=MODEL_PATH,
                video_path=video_path,
                conf=confidence,
                imgsz=640,
                save_annotated=False,
                annotated_dir=VIDEO_OUTPUT_DIR,
            )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    payload["video_url"] = f"/uploads/{filename}"
    payload["json_path"] = str(output_json.relative_to(ROOT))
    if copied_annotated:
        payload["annotated_video_url"] = f"/outputs/videos/{copied_annotated.name}"

    output_json.write_text(json.dumps(payload, indent=2))
    return jsonify(payload)


@app.get("/uploads/<path:filename>")
def uploaded_video(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.get("/outputs/videos/<path:filename>")
def output_video(filename):
    return send_from_directory(VIDEO_OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
