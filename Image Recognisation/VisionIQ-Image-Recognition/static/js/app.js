/* ================================================================
   VisionIQ — app.js
   Handles: file upload, drag-drop, URL prediction, model switch,
            loading animation steps, results rendering
   ================================================================ */

"use strict";

// ── State ─────────────────────────────────────────────────────────────────────
let selectedFile  = null;
let activeModel   = "mobilenetv2";
let isAnalyzing   = false;

// ── Bar colours (rank → gradient) ────────────────────────────────────────────
const BAR_COLORS = [
  "linear-gradient(90deg,#00e5ff,#00b0ff)",
  "linear-gradient(90deg,#00e676,#00b0ff)",
  "linear-gradient(90deg,#69f0ae,#00e5ff)",
  "linear-gradient(90deg,#ffd740,#69f0ae)",
  "linear-gradient(90deg,#ff6e40,#ffd740)",
];

// ── File select via click ─────────────────────────────────────────────────────
function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) setFile(file);
}

function setFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    document.getElementById("previewImg").src = ev.target.result;
    document.getElementById("scannerImg").src = ev.target.result;
    document.getElementById("dropInner").classList.add("hidden");
    document.getElementById("previewWrap").classList.remove("hidden");
    document.getElementById("analyzeBtn").disabled = false;
  };
  reader.readAsDataURL(file);
}

// ── Drag & Drop ───────────────────────────────────────────────────────────────
function handleDragOver(e) {
  e.preventDefault();
  document.getElementById("dropZone").classList.add("drag-over");
}
function handleDragLeave(e) {
  document.getElementById("dropZone").classList.remove("drag-over");
}
function handleDrop(e) {
  e.preventDefault();
  document.getElementById("dropZone").classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    setFile(file);
  } else {
    toast("Please drop an image file");
  }
}

// ── Model selection ───────────────────────────────────────────────────────────
function selectModel(btn) {
  document.querySelectorAll(".mtab").forEach(t => t.classList.remove("active"));
  btn.classList.add("active");
  activeModel = btn.dataset.model;
}

// ── Predict (file upload) ─────────────────────────────────────────────────────
async function predict() {
  if (!selectedFile || isAnalyzing) return;
  isAnalyzing = true;

  showLoading();

  const form = new FormData();
  form.append("file", selectedFile);
  form.append("model", activeModel);

  try {
    stepLoading(0, "Loading model…");
    await sleep(300);
    stepLoading(1, "Preprocessing image…");

    const res  = await fetch("/predict", { method:"POST", body:form });
    stepLoading(2, "Running inference…");
    const data = await res.json();

    if (!res.ok) {
      hideLoading();
      toast("Error: " + (data.error || "Unknown error"));
      isAnalyzing = false;
      return;
    }

    stepLoading(3, "Decoding predictions…");
    await sleep(400);

    hideLoading();
    renderResult(data);
  } catch(e) {
    hideLoading();
    toast("Network error — is the server running?");
  }
  isAnalyzing = false;
}

// ── Predict from URL ──────────────────────────────────────────────────────────
async function predictFromURL() {
  const url = document.getElementById("urlInput").value.trim();
  if (!url) { toast("Please enter an image URL"); return; }
  if (isAnalyzing) return;
  isAnalyzing = true;

  // Show URL image in scanner
  document.getElementById("scannerImg").src = url;
  showLoading();

  try {
    stepLoading(0, "Downloading image…");
    const res  = await fetch("/predict/url", {
      method: "POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ url, model: activeModel }),
    });
    stepLoading(2, "Running inference…");
    const data = await res.json();

    if (!res.ok) {
      hideLoading();
      toast("Error: " + (data.error || "Unknown error"));
      isAnalyzing = false;
      return;
    }

    stepLoading(3, "Decoding predictions…");
    await sleep(400);
    hideLoading();
    renderResult(data);
  } catch(e) {
    hideLoading();
    toast("Network error");
  }
  isAnalyzing = false;
}

// ── Loading state ─────────────────────────────────────────────────────────────
const STEPS = ["lstep-0","lstep-1","lstep-2","lstep-3"];

function showLoading() {
  document.getElementById("uploadCard").classList.add("hidden");
  document.getElementById("resultCard").classList.add("hidden");
  document.getElementById("loadingCard").classList.remove("hidden");
  setStatus("analyzing", "Analyzing…");
  // Reset steps
  document.querySelectorAll(".lstep").forEach(s => {
    s.classList.remove("active","done");
  });
  document.querySelectorAll(".lstep")[0].classList.add("active");
}

function hideLoading() {
  document.getElementById("loadingCard").classList.add("hidden");
}

function stepLoading(step, label) {
  const steps = document.querySelectorAll(".lstep");
  steps.forEach((s, i) => {
    s.classList.remove("active","done");
    if (i < step)  s.classList.add("done");
    if (i === step) s.classList.add("active");
  });
  document.getElementById("loadingLabel").textContent = label;
}

// ── Render result ─────────────────────────────────────────────────────────────
function renderResult(data) {
  document.getElementById("resultCard").classList.remove("hidden");
  setStatus("ready", "Done");

  // Update nav badge
  document.getElementById("modelBadge").textContent = data.model_used || activeModel;

  // ── Hero ──
  document.getElementById("topLabel").textContent = data.top_label || "—";
  document.getElementById("topConf").textContent  = `${data.top_conf?.toFixed(1)}%`;
  document.getElementById("metaModel").textContent = `🧠 ${data.model_used}`;
  document.getElementById("metaTime").textContent  = `⚡ ${data.inference_ms} ms`;

  const meta = data.metadata || {};
  document.getElementById("metaSize").textContent = meta.width
    ? `🖼 ${meta.width}×${meta.height}`
    : "🖼 —";

  // Animate confidence bar
  setTimeout(() => {
    document.getElementById("topConfBar").style.width = `${data.top_conf}%`;
  }, 100);

  // ── Predictions list ──
  const list  = document.getElementById("predictionsList");
  list.innerHTML = "";
  (data.predictions || []).forEach((p, i) => {
    const row = document.createElement("div");
    row.className = "pred-row";
    row.style.animationDelay = `${i * 80}ms`;
    row.innerHTML = `
      <span class="pred-rank">${p.rank}</span>
      <span class="pred-label">${p.label}</span>
      <div class="pred-bar-track">
        <div class="pred-bar-fill" style="background:${BAR_COLORS[i]};width:0"
             data-width="${p.bar_width}"></div>
      </div>
      <span class="pred-conf">${p.confidence.toFixed(1)}%</span>
    `;
    list.appendChild(row);
  });
  // Animate bars after paint
  setTimeout(() => {
    document.querySelectorAll(".pred-bar-fill").forEach(bar => {
      bar.style.width = bar.dataset.width + "%";
    });
  }, 150);

  // ── Analysis grid ──
  const grid = document.getElementById("analysisGrid");
  const cells = [];

  if (meta.format)    cells.push(["Format",     meta.format]);
  if (meta.size_kb)   cells.push(["File size",  `${meta.size_kb} KB`]);
  if (meta.mode)      cells.push(["Color mode", meta.mode]);

  const cv = data.cv_analysis || {};
  if (cv.brightness !== undefined) cells.push(["Brightness", cv.brightness]);
  if (cv.contrast   !== undefined) cells.push(["Contrast",   cv.contrast]);
  if (cv.edge_density !== undefined) cells.push(["Edge density", `${cv.edge_density}%`]);
  if (cv.dominant_color) {
    cells.push(["Dominant color", cv.dominant_color, true]);
  }

  grid.innerHTML = cells.map(([label, val, isColor]) => `
    <div class="ag-cell">
      <div class="ag-label">${label}</div>
      <div class="ag-val">
        ${isColor ? `<span class="ag-color-swatch" style="background:${val}"></span>` : ""}
        ${val}
      </div>
    </div>
  `).join("");

  // Scroll to result
  document.getElementById("resultCard").scrollIntoView({ behavior:"smooth", block:"start" });
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetUI() {
  selectedFile = null;
  document.getElementById("fileInput").value = "";
  document.getElementById("previewImg").src  = "";
  document.getElementById("previewWrap").classList.add("hidden");
  document.getElementById("dropInner").classList.remove("hidden");
  document.getElementById("analyzeBtn").disabled = true;
  document.getElementById("urlInput").value = "";
  document.getElementById("resultCard").classList.add("hidden");
  document.getElementById("uploadCard").classList.remove("hidden");
  window.scrollTo({ top:0, behavior:"smooth" });
  setStatus("ready", "Ready");
}

// ── Model panel ───────────────────────────────────────────────────────────────
function showModelPanel() {
  document.getElementById("modelOverlay").classList.remove("hidden");
  document.getElementById("modelPanel").classList.remove("hidden");
  // Mark active model
  document.querySelectorAll(".mp-card").forEach(c => {
    c.classList.toggle("active", c.dataset.model === activeModel);
  });
}
function hideModelPanel() {
  document.getElementById("modelOverlay").classList.add("hidden");
  document.getElementById("modelPanel").classList.add("hidden");
}

async function switchModel(name) {
  hideModelPanel();
  activeModel = name;

  // Sync model tabs
  document.querySelectorAll(".mtab").forEach(t => {
    t.classList.toggle("active", t.dataset.model === name);
  });

  const modelNames = { mobilenetv2:"MobileNetV2", resnet50:"ResNet-50", efficientnetb0:"EfficientNet-B0" };
  toast(`Switched to ${modelNames[name] || name}`);
  document.getElementById("modelBadge").textContent = modelNames[name] || name;

  // Pre-load in background
  fetch("/api/switch-model", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({ model: name }),
  }).catch(() => {});
}

// ── Status dot ────────────────────────────────────────────────────────────────
function setStatus(state, text) {
  const dot  = document.getElementById("statusDot");
  const txt  = document.getElementById("statusText");
  txt.textContent = text;
  dot.style.color = state === "analyzing" ? "#ffd740" : "#00e676";
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function toast(msg, ms = 3000) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), ms);
}

// Keyboard: Enter on URL input
document.getElementById("urlInput").addEventListener("keydown", e => {
  if (e.key === "Enter") predictFromURL();
});