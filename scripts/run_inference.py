#!/usr/bin/env python3
"""Run YOLOv8 inference and export detections for the dashboard.

Example:
    python3 scripts/run_inference.py \
      --model models/shahed_best.pt \
      --video data/demo/videos/sample.mov \
      --out-json outputs/detections/sample.json \
      --save-annotated
"""

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to shahed_best.pt or another Ultralytics model")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--out-json", required=True, help="Dashboard detections JSON output path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--save-annotated", action="store_true", help="Also save Ultralytics annotated video")
    parser.add_argument("--annotated-dir", default="outputs/videos", help="Where to copy annotated video")
    return parser.parse_args()


def box_to_normalized_xywh(xyxy, width, height):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return {
        "x": ((x1 + x2) / 2) / width,
        "y": ((y1 + y2) / 2) / height,
        "w": (x2 - x1) / width,
        "h": (y2 - y1) / height,
    }


def run_inference(model_path, video_path, conf=0.25, imgsz=640, save_annotated=False, annotated_dir="outputs/videos"):
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency. Install with `pip install -r requirements-dashboard.txt`."
        ) from exc

    model_path = Path(model_path).expanduser().resolve()
    video_path = Path(video_path).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width_px = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height_px = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    fps = fps if fps > 0 else None

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(video_path),
        conf=conf,
        imgsz=imgsz,
        save=save_annotated,
        stream=True,
        verbose=False,
    )

    detections = []
    class_counts = {}
    annotated_save_dir = None
    frames_processed = 0

    for frame_index, result in enumerate(results):
        frames_processed = frame_index + 1
        names = result.names
        height, width = result.orig_shape
        time_seconds = frame_index / fps if fps else None

        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = names.get(cls_id, str(cls_id))
            confidence = float(box.conf[0])
            class_counts[label] = class_counts.get(label, 0) + 1
            detections.append({
                "frame": frame_index,
                "time": time_seconds,
                "label": label,
                "confidence": confidence,
                "box": box_to_normalized_xywh(box.xyxy[0], width, height),
            })

        if save_annotated:
            annotated_save_dir = str(result.save_dir)

    payload = {
        "video": video_path.name,
        "model": model_path.name,
        "confidence_threshold": conf,
        "image_size": imgsz,
        "fps": fps,
        "frame_count": frame_count,
        "video_size": {
            "width": width_px,
            "height": height_px,
        },
        "classes": getattr(model, "names", {}),
        "summary": {
            "frames_processed": frames_processed,
            "detections": len(detections),
            "class_counts": class_counts,
        },
        "detections": detections,
    }

    copied_annotated = None
    if save_annotated and annotated_save_dir:
        annotated_dir = Path(annotated_dir).expanduser().resolve()
        annotated_dir.mkdir(parents=True, exist_ok=True)
        candidates = sorted(Path(annotated_save_dir).glob("*"))
        video_outputs = [p for p in candidates if p.suffix.lower() in {".avi", ".mp4", ".mov"}]
        if video_outputs:
            copied_annotated = annotated_dir / video_outputs[0].name
            shutil.copy(video_outputs[0], copied_annotated)
            payload["annotated_video"] = str(copied_annotated)

    return payload, copied_annotated


def main():
    args = parse_args()

    model_path = Path(args.model).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    try:
        payload, copied_annotated = run_inference(
            model_path=model_path,
            video_path=video_path,
            conf=args.conf,
            imgsz=args.imgsz,
            save_annotated=args.save_annotated,
            annotated_dir=args.annotated_dir,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote detections: {out_json}")

    if args.save_annotated:
        if copied_annotated:
            print(f"Copied annotated video: {copied_annotated}")
        else:
            print("Annotated output was requested, but no video file was found.")


if __name__ == "__main__":
    main()
