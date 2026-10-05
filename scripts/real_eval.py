import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Shahed-136 YOLO model on real images")
    parser.add_argument(
        "--model", 
        type=str, 
        default="/users/apgoel/runs/shahed_merged_v8s-5/weights/best.pt",
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
        default=0.01, 
        help="Confidence threshold"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="/users/apgoel/runs/real_world_eval",
        help="Directory to save annotated prediction images"
    )
    return parser.parse_args()

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

if __name__ == "__main__":
    main()