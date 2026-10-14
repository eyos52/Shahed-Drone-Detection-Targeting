# Shahed-Drone-Detection-Targeting

## Dashboard

The `dashboard/` folder is the project dashboard. It opens a local video, runs the trained
YOLO model through a local Python backend, draws detection boxes, collects review notes, and
exports the review.

The dashboard intentionally starts empty. It does not show placeholder metrics or fake clips.
Values appear only after a real video is loaded or a real detections JSON is opened.

Run it locally:

```bash
source .venv/bin/activate
python server.py
```

Then open `http://127.0.0.1:5000/`.

In VS Code, open the project folder and use **Terminal > Run Task**:

- `Dashboard: serve locally` starts the dashboard at `http://127.0.0.1:5000/`.
- `Dashboard: run inference` can still run `models/shahed_best.pt` from the terminal and write JSON to
  `outputs/detections/`.
- `Dashboard: validate files` checks the dashboard JavaScript and inference script.

To run the trained YOLOv8 model and produce dashboard-ready JSON, see
`DASHBOARD_DEPLOYMENT.md`.
