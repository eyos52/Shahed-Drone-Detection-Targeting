import argparse
import subprocess
from pathlib import Path
import imageio_ffmpeg
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Shahed-136 YOLO model on real images")
    parser.add_argument(
        "--model", 
        type=str, 
        default="/users/apgoel/runs/shahed_ft_real_v3/weights/best.pt",
        help="Path to trained .pt weights"
    )
    parser.add_argument(
        "--source", 
        type=str, 
        default="/users/apgoel/data/real_images",
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
        default="/users/apgoel/runs/real_world_eval",
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

def main():
    args = parse_args()
    
    print(f"Loading weights from: {args.model}")
    model = YOLO(args.model)
    
    results = model.predict(
        source=args.source,
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