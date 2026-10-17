import argparse
import subprocess
import tempfile
from pathlib import Path
import cv2
import imageio_ffmpeg
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
import torch # 👈 CHANGED: Added torch import to sanitize model checkpoints

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Shahed-136 YOLO model on real images")
    parser.add_argument(
        "--model", 
        type=str, 
        default="/users/apgoel/senior_design/runs/final_v5i_1280_e40/weights/last.pt",
        help="Path to trained .pt weights"
    )
    parser.add_argument(
        "--source", 
        type=str, 
        default="/users/apgoel/senior_design/data/real_images",
        help="Path to folder of test images"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.30, 
        help="Confidence threshold"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="/users/apgoel/senior_design/runs/real_world_eval_v5i",
        help="Directory to save annotated prediction images"
    )
    return parser.parse_args()

def convert_avi_to_mp4(output_dir):
    # Ultralytics saves annotated videos as MJPG .avi, which VS Code/browsers can't play
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for avi in Path(output_dir).glob("*.avi"):
        mp4 = avi.with_suffix(".mp4")
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(avi),
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
             "-movflags", "+faststart", str(mp4)],
            check=True,
        )
        print(f"Playable video saved to: {mp4}")

def usable_sources(source, tmp):
    """Folder of links to the files in `source` that OpenCV can actually open. Ultralytics aborts
    the whole run on one unreadable video (audio-only HLS tracks, half-finished *.temp.mp4
    downloads), so skip those. It must be a folder: a list of paths is read as in-memory images."""
    path = Path(source)
    if not path.is_dir():
        return source
    keep, skipped = [], []
    for f in sorted(path.iterdir()):
        ext = f.suffix[1:].lower()
        if ext in VID_FORMATS:
            cap = cv2.VideoCapture(str(f))
            ok = cap.isOpened() and cap.read()[0]
            cap.release()
            (keep if ok else skipped).append(f)
        elif ext in IMG_FORMATS:
            keep.append(f)
    for f in skipped:
        print(f"Skipping unreadable video: {f.name}")
    for f in keep:
        (Path(tmp) / f.name).symlink_to(f.resolve())
    return tmp

def main():
    args = parse_args()

    #CHANGED: Added automatic cleanup of hardcoded training pathways from metadata
    try:
        ckpt = torch.load(args.model, map_location="cpu")
        if isinstance(ckpt, dict) and "train_args" in ckpt:
            del ckpt["train_args"]
            torch.save(ckpt, args.model)
    except Exception as e:
        print(f"Note: Could not sanitize model path metadata: {e}")
    
    print(f"Loading weights from: {args.model}")
    model = YOLO(args.model)
    
    with tempfile.TemporaryDirectory() as tmp:
        results = model.predict(
            source=usable_sources(args.source, tmp),
            conf=args.conf,
            iou=0.45,
            save=True,
            project=Path(args.output).parent,
            name=Path(args.output).name,
            exist_ok=True
        )
    
    print("\n--- Detection Summary ---")
    detected_count = 0
    for r in results:
        img_name = Path(r.path).name
        boxes = len(r.boxes)
        if boxes > 0:
            detected_count += 1
            confs = [f"{float(b.conf):.2f}" for b in r.boxes]
            print(f"✅ {img_name}: {boxes} target(s) detected (confidence: {confs})")
        else:
            print(f"❌ {img_name}: No target detected")
            
    print(f"\nFinished: {detected_count}/{len(results)} images had detections.")
    print(f"Annotated outputs saved to: {args.output}")
    convert_avi_to_mp4(args.output)

if __name__ == "__main__":
    main()
