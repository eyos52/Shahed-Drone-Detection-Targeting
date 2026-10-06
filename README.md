# Shahed-Drone-Detection-Targeting

## Dashboard

The `dashboard/` folder is the project dashboard. It opens a local video, loads real detector
output JSON, draws detection boxes, collects review notes, and exports the review.

The dashboard intentionally starts empty. It does not show placeholder metrics or fake clips.
Generate detections first, then load the video and JSON into the page.

Run it locally:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000/dashboard/`.

To run the trained YOLOv8 model and produce dashboard-ready JSON, see
`DASHBOARD_DEPLOYMENT.md`.
