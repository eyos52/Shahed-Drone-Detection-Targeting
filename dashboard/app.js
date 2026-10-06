const video = document.querySelector("#reviewVideo");
const canvas = document.querySelector("#overlayCanvas");
const ctx = canvas.getContext("2d");

const elements = {
  detectionsInput: document.querySelector("#detectionsInput"),
  emptyVideoState: document.querySelector("#emptyVideoState"),
  exportCsv: document.querySelector("#exportCsv"),
  exportJson: document.querySelector("#exportJson"),
  findingTitle: document.querySelector("#findingTitle"),
  confidenceValue: document.querySelector("#confidenceValue"),
  timeValue: document.querySelector("#timeValue"),
  lightingValue: document.querySelector("#lightingValue"),
  sizeValue: document.querySelector("#sizeValue"),
  detectionRows: document.querySelector("#detectionRows"),
  nextFinding: document.querySelector("#nextFinding"),
  prevFinding: document.querySelector("#prevFinding"),
  reviewNote: document.querySelector("#reviewNote"),
  sidebarStatus: document.querySelector("#sidebarStatus"),
  summaryClip: document.querySelector("#summaryClip"),
  summaryDetections: document.querySelector("#summaryDetections"),
  summaryModel: document.querySelector("#summaryModel"),
  summaryStatus: document.querySelector("#summaryStatus"),
};

let detections = [];
let activeDetectionIndex = -1;
let currentDecision = "confirm";
let loadedVideoName = "";
let loadedPayload = null;

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
  return "not provided";
}

function boxLabel(box) {
  if (!box) return "not provided";
  return `${box.x.toFixed(3)}, ${box.y.toFixed(3)}, ${box.w.toFixed(3)}, ${box.h.toFixed(3)}`;
}

function estimateBoxSize(box) {
  if (!box || !video.videoWidth || !video.videoHeight) return "not available";
  const width = Math.round(box.w * video.videoWidth);
  const height = Math.round(box.h * video.videoHeight);
  return `${width} x ${height} px`;
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function drawOverlay() {
  resizeCanvas();
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);

  const detection = detections[activeDetectionIndex];
  if (!detection || !detection.box) return;

  const { box } = detection;
  const x = box.x * rect.width;
  const y = box.y * rect.height;
  const w = box.w * rect.width;
  const h = box.h * rect.height;
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

function setStatus(text, tone = "idle") {
  elements.summaryStatus.textContent = text;
  elements.summaryStatus.className = `status ${tone}`;
  elements.sidebarStatus.textContent = text;
}

function syncControls() {
  const hasDetections = detections.length > 0;
  elements.prevFinding.disabled = !hasDetections;
  elements.nextFinding.disabled = !hasDetections;
  elements.exportCsv.disabled = !hasDetections;
  elements.exportJson.disabled = !loadedPayload && !loadedVideoName;
}

function syncInspector() {
  const detection = detections[activeDetectionIndex];
  if (!detection) {
    elements.findingTitle.textContent = "No detection selected";
    elements.confidenceValue.textContent = "not available";
    elements.timeValue.textContent = "not available";
    elements.lightingValue.textContent = "not available";
    elements.sizeValue.textContent = "not available";
    drawOverlay();
    syncControls();
    return;
  }

  elements.findingTitle.textContent = detection.label;
  elements.confidenceValue.textContent = detection.confidence.toFixed(3);
  elements.timeValue.textContent = detectionTimeLabel(detection);
  elements.lightingValue.textContent = detection.lighting || "not provided";
  elements.sizeValue.textContent = detection.size || estimateBoxSize(detection.box);
  drawOverlay();
  syncControls();
}

function syncSummary() {
  elements.summaryClip.textContent = loadedVideoName || (loadedPayload && loadedPayload.video) || "No video loaded";
  elements.summaryModel.textContent = (loadedPayload && loadedPayload.model) || "No model metadata";
  elements.summaryDetections.textContent = loadedPayload ? String(detections.length) : "No detections loaded";

  if (loadedVideoName && loadedPayload) {
    setStatus("Ready for review", "ok");
  } else if (loadedVideoName) {
    setStatus("Video loaded", "idle");
  } else if (loadedPayload) {
    setStatus("Detections loaded", "idle");
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

function normalizeDetections(payload) {
  return (payload.detections || [])
    .map((item) => ({
      frame: item.frame,
      time: item.time,
      label: String(item.label || item.class || "object"),
      confidence: Number(item.confidence ?? item.conf ?? 0),
      lighting: item.lighting,
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
  activeDetectionIndex = detections.length ? 0 : -1;
  renderDetectionRows();
  syncSummary();
  syncInspector();
  highlightActiveRow();
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
  loadedVideoName = file.name;
  video.src = URL.createObjectURL(file);
  video.load();
  elements.emptyVideoState.hidden = true;
  syncSummary();
  syncControls();
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

elements.prevFinding.addEventListener("click", () => seekDetection(activeDetectionIndex - 1));
elements.nextFinding.addEventListener("click", () => seekDetection(activeDetectionIndex + 1));

document.querySelectorAll(".decision").forEach((button) => {
  button.addEventListener("click", () => {
    currentDecision = button.dataset.decision;
    document.querySelectorAll(".decision").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
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
});
video.addEventListener("pause", drawOverlay);
video.addEventListener("seeked", drawOverlay);
window.addEventListener("resize", drawOverlay);

renderDetectionRows();
syncSummary();
syncInspector();
