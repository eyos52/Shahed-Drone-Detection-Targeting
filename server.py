#!/usr/bin/env python3
"""Local backend for the Shahed dashboard."""

import json
import re
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
DASHBOARD_DIR = ROOT / "dashboard"
UPLOAD_DIR = ROOT / "outputs" / "uploads"
DETECTIONS_DIR = ROOT / "outputs" / "detections"
VIDEO_OUTPUT_DIR = ROOT / "outputs" / "videos"
MODEL_PATH = ROOT / "models" / "shahed_best.pt"

sys.path.insert(0, str(ROOT / "scripts"))
from run_inference import run_inference  # noqa: E402

app = Flask(__name__, static_folder=str(DASHBOARD_DIR), static_url_path="")


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
