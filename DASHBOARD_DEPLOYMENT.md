# Dashboard Deployment

Use this when the trained model already exists on another laptop. The dashboard is committed
as normal project code under `dashboard/`, so anyone who downloads the repo gets it.

## 1. Put the model in the repo

Create this folder and copy the trained checkpoint into it:

```bash
mkdir -p models
```

Expected file:

```text
models/shahed_best.pt
```

The Colab notebook saved this same model as:

```text
/content/drive/MyDrive/Shahed_DataSets/trained_models/shahed_best.pt
```

## 2. Install the dashboard inference dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dashboard.txt
```

## 3. Run inference on a video

Put a video somewhere local, for example:

```text
data/demo/videos/test-video.mov
```

Then run:

```bash
python3 scripts/run_inference.py \
  --model models/shahed_best.pt \
  --video data/demo/videos/test-video.mov \
  --out-json outputs/detections/test-video.json \
  --save-annotated
```

This creates:

```text
outputs/detections/test-video.json
outputs/videos/<annotated-video>
```

## 4. Open the dashboard

```bash
python3 -m http.server 8000
```

Open:

```text
http://127.0.0.1:8000/dashboard/
```

In the dashboard:

1. Click **Open video** and select the same input video.
2. Click **Open detections** and select the JSON from `outputs/detections/`.
3. Review detections, confidence, timestamps, and notes.

The page starts empty by design. Values appear only after a real video or real detections JSON
is loaded.

## Notes

The dashboard is designed around the YOLOv8 notebook model:

```text
0 = other
1 = shahed
```

It can still display any detector output as long as the JSON contains:

```json
{
  "video": "test-video.mov",
  "model": "shahed_best.pt",
  "detections": [
    {
      "frame": 126,
      "time": 4.2,
      "label": "shahed",
      "confidence": 0.91,
      "box": { "x": 0.51, "y": 0.36, "w": 0.12, "h": 0.09 }
    }
  ]
}
```
