const video = document.querySelector("#reviewVideo");
const canvas = document.querySelector("#overlayCanvas");
const ctx = canvas.getContext("2d");

const elements = {
  detailGrid: document.querySelector("#detailGrid"),
  detectionsInput: document.querySelector("#detectionsInput"),
  emptyVideoState: document.querySelector("#emptyVideoState"),
  exportCsv: document.querySelector("#exportCsv"),
  exportJson: document.querySelector("#exportJson"),
  findingTitle: document.querySelector("#findingTitle"),
  confidenceValue: document.querySelector("#confidenceValue"),
  timeValue: document.querySelector("#timeValue"),
  fpsValue: document.querySelector("#fpsValue"),
  sizeValue: document.querySelector("#sizeValue"),
  detectionRows: document.querySelector("#detectionRows"),
  inspectorPanel: document.querySelector("#inspectorPanel"),
  nextFinding: document.querySelector("#nextFinding"),
  openVideoButton: document.querySelector("#openVideoButton"),
  prevFinding: document.querySelector("#prevFinding"),
  reviewNote: document.querySelector("#reviewNote"),
  runModel: document.querySelector("#runModel"),
  summaryClip: document.querySelector("#summaryClip"),
  summaryDetections: document.querySelector("#summaryDetections"),
  summaryLine: document.querySelector("#summaryLine"),
  summaryModel: document.querySelector("#summaryModel"),
  summaryStatus: document.querySelector("#summaryStatus"),
  timeline: document.querySelector("#timeline"),
  toast: document.querySelector("#toast"),
  videoDropzone: document.querySelector("#videoDropzone"),
};

let detections = [];
let activeDetectionIndex = -1;
let currentDecision = "confirm";
let loadedVideoName = "";
let loadedPayload = null;
let selectedVideoFile = null;
let selectedVideoUrl = "";
let animationFrameId = null;
let toastTimer = null;

function formatTime(seconds) {
  if (seconds == null || Number.isNaN(Number(seconds))) return "";
  const value = Number(seconds);
  const mins = Math.floor(value / 60).toString().padStart(2, "0");
  const secs = (value % 60).toFixed(1).padStart(4, "0");
  return `${mins}:${secs}`;
}

function detectionTimeLabel(detection) {
  const time = formatTime(detection.time);
  if (time) return time;
  if (detection.frame != null) return `frame ${detection.frame}`;
  return "n/a";
}

function boxLabel(box) {
  if (!box) return "n/a";
  return `${box.x.toFixed(3)}, ${box.y.toFixed(3)}, ${box.w.toFixed(3)}, ${box.h.toFixed(3)}`;
}

function estimateBoxSize(box) {
  if (!box || !video.videoWidth || !video.videoHeight) return "n/a";
  const width = Math.round(box.w * video.videoWidth);
  const height = Math.round(box.h * video.videoHeight);
  return `${width} x ${height} px`;
}

function fpsLabel() {
  const fps = Number(loadedPayload && loadedPayload.fps);
  if (!Number.isFinite(fps) || fps <= 0) return "n/a";
  return fps.toFixed(2);
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function renderedVideoRect() {
  const rect = canvas.getBoundingClientRect();
  const videoAspect = video.videoWidth && video.videoHeight ? video.videoWidth / video.videoHeight : rect.width / rect.height;
  const shellAspect = rect.width / rect.height;

  if (shellAspect > videoAspect) {
    const height = rect.height;
    const width = height * videoAspect;
    return { x: (rect.width - width) / 2, y: 0, width, height };
  }

  const width = rect.width;
  const height = width / videoAspect;
  return { x: 0, y: (rect.height - height) / 2, width, height };
}

function detectionTimeTolerance() {
  const fps = Number(loadedPayload && loadedPayload.fps);
  if (Number.isFinite(fps) && fps > 0) return Math.max(1 / fps, 0.04);
  return 0.08;
}

function detectionsAtCurrentTime() {
  if (!detections.length) return [];
  const timedDetections = detections.filter((detection) => Number.isFinite(Number(detection.time)));
  if (!timedDetections.length) {
    return activeDetectionIndex >= 0 ? [detections[activeDetectionIndex]] : [];
  }

  const currentTime = Number(video.currentTime || 0);
  const tolerance = detectionTimeTolerance();
  const matches = timedDetections.filter((detection) => Math.abs(Number(detection.time) - currentTime) <= tolerance);
  if (matches.length) return matches;

  const nearest = timedDetections.reduce((best, detection) => {
    const distance = Math.abs(Number(detection.time) - currentTime);
    if (!best || distance < best.distance) return { detection, distance };
    return best;
  }, null);

  return nearest && nearest.distance <= tolerance * 2.5 ? [nearest.detection] : [];
}

function drawDetection(detection, videoRect) {
  const { box } = detection;
  const w = box.w * videoRect.width;
  const h = box.h * videoRect.height;
  const x = videoRect.x + (box.x * videoRect.width) - (w / 2);
  const y = videoRect.y + (box.y * videoRect.height) - (h / 2);
  const confidence = Math.round(detection.confidence * 100);
  const label = `${detection.label} ${confidence}%`;

  ctx.lineWidth = 2;
  ctx.strokeStyle = detection.label === "shahed" ? "#55b879" : "#d2a34f";
  ctx.fillStyle = "rgba(18, 20, 24, 0.88)";
  ctx.strokeRect(x, y, w, h);
  ctx.fillRect(x, Math.max(0, y - 30), Math.max(104, label.length * 7.2), 24);
  ctx.fillStyle = "#eceff3";
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText(label, x + 8, Math.max(16, y - 13));
}

function drawOverlay() {
  resizeCanvas();
  const rect = canvas.getBoundingClientRect();
  const videoRect = renderedVideoRect();
  ctx.clearRect(0, 0, rect.width, rect.height);

  if (!loadedVideoName || !video.videoWidth || !video.videoHeight) return;
  detectionsAtCurrentTime()
    .filter((detection) => detection && detection.box)
    .forEach((detection) => drawDetection(detection, videoRect));
}

function updateOverlayWhilePlaying() {
  drawOverlay();
  if (!video.paused && !video.ended) {
    animationFrameId = requestAnimationFrame(updateOverlayWhilePlaying);
  }
}

function setStatus(text, tone = "idle") {
  elements.summaryStatus.innerHTML = `<span class="status-dot" aria-hidden="true"></span>${text}`;
  elements.summaryStatus.className = `status ${tone}`;
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    elements.toast.hidden = true;
  }, 2200);
}

function syncControls() {
  const hasDetections = detections.length > 0;
  elements.prevFinding.disabled = !hasDetections;
  elements.nextFinding.disabled = !hasDetections;
  elements.exportCsv.disabled = !hasDetections;
  elements.runModel.hidden = !selectedVideoFile;
  elements.exportJson.hidden = !loadedPayload;
  elements.openVideoButton.classList.toggle("primary", !selectedVideoFile);
  elements.openVideoButton.classList.toggle("secondary", Boolean(selectedVideoFile));
  elements.runModel.classList.toggle("primary", Boolean(selectedVideoFile) && !loadedPayload);
  elements.runModel.classList.toggle("secondary", Boolean(loadedPayload));
  elements.exportJson.classList.toggle("primary", Boolean(loadedPayload));
  elements.exportJson.classList.toggle("secondary", !loadedPayload);
}

function syncInspector() {
  const detection = detections[activeDetectionIndex];
  if (!detection) {
    elements.inspectorPanel.hidden = true;
    elements.detailGrid.hidden = true;
    elements.findingTitle.textContent = "Select a detection.";
    elements.confidenceValue.textContent = "";
    elements.timeValue.textContent = "";
    elements.fpsValue.textContent = "";
    elements.sizeValue.textContent = "";
    drawOverlay();
    syncControls();
    return;
  }

  elements.inspectorPanel.hidden = false;
  elements.detailGrid.hidden = false;
  elements.findingTitle.textContent = detection.label;
  elements.confidenceValue.textContent = detection.confidence.toFixed(3);
  elements.timeValue.textContent = detectionTimeLabel(detection);
  elements.fpsValue.textContent = fpsLabel();
  elements.sizeValue.textContent = detection.size || estimateBoxSize(detection.box);
  syncDecisionButtons(detection.reviewStatus || currentDecision);
  drawOverlay();
  syncControls();
}

function syncDecisionButtons(decision) {
  document.querySelectorAll(".decision").forEach((button) => {
    button.classList.toggle("active", button.dataset.decision === decision);
  });
}

function syncSummary() {
  elements.summaryClip.textContent = loadedVideoName || (loadedPayload && loadedPayload.video) || "No video loaded";
  elements.summaryModel.textContent = (loadedPayload && loadedPayload.model) || "No model metadata";
  elements.summaryDetections.textContent = loadedPayload ? String(detections.length) : "0";
  elements.summaryLine.hidden = !loadedVideoName;

  if (loadedVideoName && loadedPayload) {
    setStatus("Ready", "ok");
  } else if (loadedVideoName) {
    setStatus("Video", "idle");
  } else if (loadedPayload) {
    setStatus("Detections", "idle");
  } else {
    setStatus("Waiting", "idle");
  }
}

function seekDetection(index) {
  if (!detections.length) return;
  activeDetectionIndex = (index + detections.length) % detections.length;
  const detection = detections[activeDetectionIndex];
  if (detection.time != null && !Number.isNaN(Number(detection.time))) {
    video.currentTime = Number(detection.time);
  }
  syncInspector();
  highlightActiveRow();
}

function highlightActiveRow() {
  document.querySelectorAll("[data-detection-index]").forEach((row) => {
    row.classList.toggle("selected", Number(row.dataset.detectionIndex) === activeDetectionIndex);
  });
}

function renderTimeline() {
  elements.timeline.innerHTML = "";
  if (!detections.length) return;

  const duration = video.duration && Number.isFinite(video.duration)
    ? video.duration
    : Number(loadedPayload && loadedPayload.frame_count) / Number(loadedPayload && loadedPayload.fps);

  detections.forEach((detection) => {
    if (!Number.isFinite(Number(detection.time)) || !Number.isFinite(duration) || duration <= 0) return;
    const marker = document.createElement("span");
    marker.className = `timeline-marker ${detection.reviewStatus || ""}`;
    marker.style.left = `${Math.min(100, Math.max(0, (Number(detection.time) / duration) * 100))}%`;
    elements.timeline.appendChild(marker);
  });
}

function normalizeDetections(payload) {
  return (payload.detections || [])
    .map((item) => ({
      frame: item.frame,
      time: item.time,
      label: String(item.label || item.class || "object"),
      confidence: Number(item.confidence ?? item.conf ?? 0),
      lighting: item.lighting,
      reviewStatus: item.reviewStatus || "",
      size: item.size,
      box: item.box,
    }))
    .filter((item) => item.box && Number.isFinite(item.confidence));
}

function renderDetectionRows() {
  elements.detectionRows.innerHTML = "";
  elements.detectionRows.className = "";

  if (!detections.length) {
    elements.detectionRows.className = "table-body-empty";
    elements.detectionRows.textContent = "No detections loaded.";
    return;
  }

  detections.forEach((detection, index) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "table-row";
    row.dataset.detectionIndex = String(index);
    row.innerHTML = `
      <span>${detectionTimeLabel(detection)}</span>
      <span>${detection.label}</span>
      <span>${detection.confidence.toFixed(3)}</span>
      <span>${boxLabel(detection.box)}</span>
    `;
    row.addEventListener("click", () => seekDetection(index));
    elements.detectionRows.appendChild(row);
  });
}

function loadDetectionPayload(payload) {
  loadedPayload = payload;
  detections = normalizeDetections(payload);
  if (payload.video) loadedVideoName = payload.video;
  activeDetectionIndex = detections.length ? 0 : -1;
  renderDetectionRows();
  renderTimeline();
  syncSummary();
  if (detections.length) {
    seekDetection(0);
  } else {
    syncInspector();
    highlightActiveRow();
  }
}

function setVideoSource(file, url) {
  selectedVideoFile = file || selectedVideoFile;
  loadedVideoName = (file && file.name) || loadedVideoName;
  if (selectedVideoUrl && selectedVideoUrl.startsWith("blob:")) {
    URL.revokeObjectURL(selectedVideoUrl);
  }
  selectedVideoUrl = url || URL.createObjectURL(file);
  video.src = selectedVideoUrl;
  video.load();
  elements.emptyVideoState.hidden = true;
  elements.videoDropzone.classList.remove("drag-over");
  syncSummary();
  syncControls();
}

async function runModelFromDashboard() {
  if (!selectedVideoFile) return;

  const formData = new FormData();
  formData.append("video", selectedVideoFile);
  formData.append("conf", "0.7");

  elements.runModel.disabled = true;
  elements.runModel.textContent = "Running";
  setStatus("Running", "running");

  try {
    const response = await fetch("/api/infer", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Model run failed.");

    if (payload.video_url) {
      setVideoSource(selectedVideoFile, payload.video_url);
    }
    loadDetectionPayload(payload);
    showToast("Model finished");
  } catch (error) {
    setStatus("Failed", "alert");
    elements.detectionRows.className = "table-body-empty";
    elements.detectionRows.textContent = error.message.includes("Failed to fetch")
      ? "Start the backend with `python server.py`, then open http://127.0.0.1:5000."
      : error.message;
  } finally {
    elements.runModel.textContent = "Run model";
    elements.runModel.disabled = false;
    syncControls();
  }
}

function downloadFile(filename, mimeType, content) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

document.querySelector("#videoInput").addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (!file) return;
  setVideoSource(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  elements.videoDropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.videoDropzone.classList.add("drag-over");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  elements.videoDropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.videoDropzone.classList.remove("drag-over");
  });
});

elements.videoDropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file && file.type.startsWith("video/")) {
    setVideoSource(file);
  }
});

elements.detectionsInput.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) return;
  try {
    loadDetectionPayload(JSON.parse(await file.text()));
  } catch (error) {
    setStatus("Invalid JSON", "alert");
    elements.detectionRows.className = "table-body-empty";
    elements.detectionRows.textContent = "Could not read this detections file.";
  }
});

elements.runModel.addEventListener("click", runModelFromDashboard);
elements.prevFinding.addEventListener("click", () => seekDetection(activeDetectionIndex - 1));
elements.nextFinding.addEventListener("click", () => seekDetection(activeDetectionIndex + 1));

document.querySelectorAll(".decision").forEach((button) => {
  button.addEventListener("click", () => {
    currentDecision = button.dataset.decision;
    if (detections[activeDetectionIndex]) {
      detections[activeDetectionIndex].reviewStatus = currentDecision;
      renderTimeline();
    }
    syncDecisionButtons(currentDecision);
  });
});

elements.exportJson.addEventListener("click", () => {
  const payload = {
    video: loadedVideoName || (loadedPayload && loadedPayload.video) || null,
    model: loadedPayload ? loadedPayload.model || null : null,
    decision: currentDecision,
    reviewer_note: elements.reviewNote.value,
    detections,
  };
  downloadFile("shahed-review.json", "application/json", JSON.stringify(payload, null, 2));
  showToast("Saved");
});

elements.exportCsv.addEventListener("click", () => {
  const rows = [
    ["time_or_frame", "label", "confidence", "box_x", "box_y", "box_w", "box_h"],
    ...detections.map((detection) => [
      detectionTimeLabel(detection),
      detection.label,
      detection.confidence.toFixed(6),
      detection.box.x,
      detection.box.y,
      detection.box.w,
      detection.box.h,
    ]),
  ];
  downloadFile("shahed-detections.csv", "text/csv", rows.map((row) => row.join(",")).join("\n"));
});

video.addEventListener("loadedmetadata", () => {
  elements.emptyVideoState.hidden = true;
  syncInspector();
  renderTimeline();
});
video.addEventListener("play", () => {
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  updateOverlayWhilePlaying();
});
video.addEventListener("pause", drawOverlay);
video.addEventListener("seeked", drawOverlay);
video.addEventListener("timeupdate", drawOverlay);
window.addEventListener("resize", drawOverlay);

document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea")) return;
  const key = event.key.toLowerCase();
  if (key === "arrowleft") seekDetection(activeDetectionIndex - 1);
  if (key === "arrowright") seekDetection(activeDetectionIndex + 1);
  if (key === "c") document.querySelector('[data-decision="confirm"]').click();
  if (key === "i") document.querySelector('[data-decision="ignore"]').click();
  if (key === "m") document.querySelector('[data-decision="miss"]').click();
});

renderDetectionRows();
syncSummary();
syncInspector();
