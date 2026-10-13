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

## 3. Start the dashboard backend

Run:

```bash
source .venv/bin/activate
python server.py
```

Open:

```text
http://127.0.0.1:5000/
```

In the dashboard:

1. Click **Open video** and select a local video.
2. Choose the confidence threshold. `0.7` is a cleaner demo default because it filters weak guesses.
3. Click **Run model**.
4. Review detections, confidence, timestamps, and notes.

The backend saves the generated detections JSON under:

```text
outputs/detections/
```

## Optional terminal inference

The dashboard can also open an existing detections JSON. To make one manually:

```bash
python3 scripts/run_inference.py \
  --model models/shahed_best.pt \
  --video data/demo/videos/test-video.mov \
  --out-json outputs/detections/test-video.json \
  --conf 0.7 \
  --save-annotated
```

The page starts empty by design. Values appear only after a real video or real detections JSON
is loaded.

## VS Code workflow

Open the project folder in VS Code, then choose **Terminal > Run Task**.

- `Dashboard: serve locally` starts the local dashboard server.
- `Dashboard: run inference` asks for a video path, output name, and confidence threshold.
- `Dashboard: validate files` checks the dashboard JavaScript and inference bridge.

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
