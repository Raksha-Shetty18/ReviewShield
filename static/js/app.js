/* ==========================================================================
   ReviewShield – Frontend Application Controller
   Modern Cybersecurity Analytics SaaS
   ========================================================================== */

// Global State
let currentSingleResult = null;
let patternAnalysisData = null;
let modelInsightsData = null;
let currentInsightGraph = "roc";
let currentModalPredictionId = null;

// Charts
let chartDashboardPie = null;
let chartDashboardHistory = null;
let chartPatternRating = null;
let chartPatternTiming = null;
let chartPatternBurst = null;
let chartPatternProducts = null;
let chartInsightCurves = null;
let chartXaiWaterfall = null;
let chartXaiGlobal = null;
let chartXaiSampleWaterfall = null;

// Bulk State
let bulkFilteredResults = [];
let bulkCurrentPage = 1;
const BULK_PAGE_SIZE = 10;
let bulkActiveFile = null;
let bulkRawData = null;

// History State
let historyCurrentPage = 0;
const HISTORY_PAGE_SIZE = 10;
let historySearchTimeout = null;

// Explainable AI State
let xaiGlobalData = null;
let xaiSamplesData = null;
let currentXaiSubtab = "global";

/* ================= INITIALIZATION ================= */
document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initStarRating();
  loadDashboardStats();
  loadHistoryPage(0);
  initDragAndDrop();

  // Load initial tab from hash or default to home
  const hash = window.location.hash.replace("#", "") || "home";
  switchTab(hash);

  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.get("demo_predict") === "true") {
    setTimeout(async () => {
      switchTab("single-review");
      await loadRandomSampleIntoForm();
      setTimeout(() => {
        predictSingleReview();
      }, 400);
    }, 400);
  }
});

/* ================= NAVIGATION ================= */
function initNavigation() {
  const navItems = document.querySelectorAll(".sidebar-nav-container .nav-link");
  navItems.forEach(item => {
    item.addEventListener("click", () => {
      const tabId = item.getAttribute("data-tab");
      switchTab(tabId);
    });
  });

  window.addEventListener("hashchange", () => {
    const hash = window.location.hash.replace("#", "") || "home";
    switchTab(hash);
  });
}

function switchTab(tabId) {
  // Update sidebar active link
  document.querySelectorAll(".sidebar-nav-container .nav-link").forEach(item => {
    if (item.getAttribute("data-tab") === tabId) {
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  });

  // Update tab pane active
  document.querySelectorAll(".tab-pane").forEach(pane => {
    pane.classList.remove("active");
  });

  const targetPane = document.getElementById(`pane-${tabId}`);
  if (targetPane) {
    targetPane.classList.add("active");
  }

  // Header Title
  const titles = {
    "home": "ReviewShield – Autonomous Fraud Verification",
    "dashboard": "Security Analytics Overview",
    "single-review": "Single Review Prediction & Analysis",
    "bulk-review": "Bulk Review Dataset Inference",
    "explainable-ai": "Explainable AI (TreeSHAP & Lexical Attribution)",
    "behaviour-patterns": "Behaviour & Temporal Pattern Intelligence",
    "model-insights": "Model Performance & Evaluation Workspace",
    "history": "Prediction History Audit Trail"
  };
  const pageTitleEl = document.getElementById("pageTitle");
  if (pageTitleEl) pageTitleEl.textContent = titles[tabId] || "Security Analytics Overview";
  window.location.hash = tabId;

  // Lazy load modules
  if (tabId === "dashboard") {
    loadDashboardStats();
  } else if (tabId === "behaviour-patterns") {
    loadPatternAnalysis();
  } else if (tabId === "model-insights") {
    loadModelInsights();
  } else if (tabId === "history") {
    loadHistoryPage(0);
  } else if (tabId === "explainable-ai") {
    if (!xaiGlobalData) loadXaiGlobal();
  }
}

/* ================= CLICKABLE STAR RATING WIDGET (SECTION 18) ================= */
function initStarRating() {
  const stars = document.querySelectorAll("#starRatingSelector .star");
  const hiddenInput = document.getElementById("singleRating");

  stars.forEach(star => {
    star.addEventListener("click", () => {
      const val = parseInt(star.getAttribute("data-value") || star.getAttribute("data-val") || "5");
      if (hiddenInput) hiddenInput.value = val;
      updateStars(val);
    });

    star.addEventListener("mouseenter", () => {
      const val = parseInt(star.getAttribute("data-value") || star.getAttribute("data-val") || "5");
      stars.forEach(s => {
        const sv = parseInt(s.getAttribute("data-value") || s.getAttribute("data-val") || "5");
        if (sv <= val) s.classList.add("hovered");
        else s.classList.remove("hovered");
      });
    });

    star.addEventListener("mouseleave", () => {
      stars.forEach(s => s.classList.remove("hovered"));
    });
  });
}

function updateStars(value) {
  const stars = document.querySelectorAll("#starRatingSelector .star");
  stars.forEach(s => {
    const sv = parseInt(s.getAttribute("data-value") || s.getAttribute("data-val") || "5");
    if (sv <= value) {
      s.classList.add("active");
      s.classList.add("selected");
    } else {
      s.classList.remove("active");
      s.classList.remove("selected");
    }
  });
}

/* ================= DASHBOARD MODULE (SECTION 6) ================= */
async function loadDashboardStats() {
  try {
    const res = await fetch("/api/dashboard/stats");
    if (!res.ok) throw new Error("Dashboard stats error");
    const stats = await res.json();

    const elTotal = document.getElementById("kpiTotalPredictions");
    if (elTotal) elTotal.textContent = stats.total_analyzed.toLocaleString();

    const elFake = document.getElementById("kpiFakeDetected");
    if (elFake) elFake.textContent = stats.fake_count.toLocaleString();

    const elFakePct = document.getElementById("kpiFakePercent");
    if (elFakePct) elFakePct.textContent = `${stats.fake_percentage}% of total`;

    const elGen = document.getElementById("kpiGenuineDetected");
    if (elGen) elGen.textContent = stats.genuine_count.toLocaleString();

    const elGenPct = document.getElementById("kpiGenuinePercent");
    if (elGenPct) elGenPct.textContent = `${stats.genuine_percentage}% of total`;

    const elFakeRate = document.getElementById("kpiFakeRate");
    if (elFakeRate) elFakeRate.textContent = `${stats.fake_percentage}%`;

    const elHomeTotal = document.getElementById("homeTotalRecords");
    if (elHomeTotal) elHomeTotal.textContent = stats.total_analyzed.toLocaleString();

    const elNavBadge = document.getElementById("navBadgeTotal");
    if (elNavBadge) elNavBadge.textContent = stats.total_analyzed.toLocaleString();

    // Render Doughnut Chart
    renderDashboardDoughnut(stats.fake_count, stats.genuine_count);

    // Render Risk Sequence
    if (stats.distribution_points && stats.distribution_points.length > 0) {
      renderDashboardHistoryChart(stats.distribution_points);
    }

    // Render Recent Table
    renderDashboardRecentTable(stats.recent_activity || []);
  } catch (err) {
    console.error("Dashboard error:", err);
  }
}

function renderDashboardDoughnut(fakeCount, genuineCount) {
  const ctx = document.getElementById("chartDashboardPie");
  if (!ctx) return;

  if (chartDashboardPie) {
    chartDashboardPie.data.datasets[0].data = [fakeCount, genuineCount];
    chartDashboardPie.update();
    return;
  }

  chartDashboardPie = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Potentially Fake", "Potentially Genuine"],
      datasets: [{
        data: [fakeCount, genuineCount],
        backgroundColor: ["#DC5252", "#15966A"],
        borderWidth: 2,
        borderColor: "#FFFFFF",
        hoverOffset: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            boxWidth: 12,
            font: { size: 11.5, family: "Inter" },
            color: "#334155"
          }
        }
      },
      cutout: "68%"
    }
  });
}

function renderDashboardHistoryChart(points) {
  const ctx = document.getElementById("chartDashboardHistory");
  if (!ctx) return;

  const labels = points.map((_, i) => `#${i + 1}`);
  const dataVals = points.map(p => Math.round(p.fake_prob * 100));
  const colors = points.map(p => p.fake_prob >= 0.54 ? "#DC5252" : "#15966A");

  if (chartDashboardHistory) {
    chartDashboardHistory.data.labels = labels;
    chartDashboardHistory.data.datasets[0].data = dataVals;
    chartDashboardHistory.data.datasets[0].backgroundColor = colors;
    chartDashboardHistory.update();
    return;
  }

  chartDashboardHistory = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Ensemble Fake Probability (%)",
        data: dataVals,
        backgroundColor: colors,
        borderRadius: 3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { font: { size: 10 }, color: "#64748B" }
        },
        y: {
          min: 0,
          max: 100,
          ticks: {
            stepSize: 20,
            callback: v => v + "%",
            font: { size: 10 },
            color: "#64748B"
          },
          grid: { color: "#E2E8F0" }
        }
      }
    }
  });
}

function renderDashboardRecentTable(activity) {
  const tbody = document.getElementById("dashRecentActivityBody");
  if (!tbody) return;

  if (activity.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding: 20px; color: var(--text-dim);">No reviews analysed yet. Run a single analysis or upload a dataset.</td></tr>`;
    return;
  }

  tbody.innerHTML = "";
  activity.forEach(item => {
    const isFake = item.ensemble_prediction === "Potentially Fake";
    const ratingVal = item.rating || 5;
    const starsStr = "★".repeat(Math.min(5, Math.max(1, Math.round(ratingVal))));
    const predId = item.prediction_id || `PRED_${item.id}`;
    const timeStr = item.analysis_timestamp ? String(item.analysis_timestamp).slice(0, 19).replace('T', ' ') : 'Recent';

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><code>${predId}</code></td>
      <td style="white-space:nowrap; color:var(--text-muted);">${timeStr}</td>
      <td>
        <div style="font-weight:600; color:var(--text-heading); font-size:0.85rem;">${escapeHtml(item.review_title || 'Untitled Review')}</div>
        <div style="font-size:0.75rem; color:var(--text-muted); max-width:280px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(item.review_text || '')}</div>
      </td>
      <td style="color:#F59E0B; white-space:nowrap;">${starsStr}</td>
      <td><strong style="color:${isFake ? 'var(--fake-red)' : 'var(--genuine-green)'};">${(item.ensemble_fake_prob * 100).toFixed(1)}%</strong></td>
      <td><span class="sample-badge ${isFake ? 'fake' : 'genuine'}">${item.ensemble_prediction}</span></td>
      <td>
        <button class="btn btn-pdf btn-sm" onclick="downloadSinglePdf('${predId}')">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
          PDF
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

/* ================= SINGLE REVIEW ANALYSIS (SECTIONS 7 & 19) ================= */
async function predictSingleReview(e) {
  if (e && e.preventDefault) e.preventDefault();
  return handleSingleReviewSubmit(e || window.event || { preventDefault: () => {} });
}

async function handleSingleReviewSubmit(e) {
  if (e && e.preventDefault) e.preventDefault();
  const btn = document.getElementById("btnAnalyzeSingle");
  const originalText = btn ? btn.innerHTML : "Run Fraud Inspection";
  const chkAll = document.getElementById("chkIncludeAllModels");
  const includeAll = chkAll ? chkAll.checked : false;

  if (btn) {
    btn.innerHTML = `<span style="font-size:11px;">Scoring Ensemble...</span>`;
    btn.disabled = true;
  }

  const payload = {
    review_title: document.getElementById("singleTitle").value,
    review_text: document.getElementById("singleText").value,
    rating: parseFloat(document.getElementById("singleRating").value) || 5.0,
    review_date: document.getElementById("singleDate").value || new Date().toISOString().split("T")[0],
    reviewer_id: document.getElementById("singleReviewerId").value || null,
    product_id: document.getElementById("singleProductId").value || null,
    include_all_models: includeAll
  };

  try {
    const res = await fetch("/api/predict/single", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error("Prediction API call failed.");
    const data = await res.json();
    currentSingleResult = data;

    renderSinglePredictionResult(data);
    loadDashboardStats(); // update live counter
  } catch (err) {
    console.error("Single review error:", err);
    alert("Prediction failed: " + err.message);
  } finally {
    if (btn) {
      btn.innerHTML = originalText;
      btn.disabled = false;
    }
  }
}

function renderSinglePredictionResult(data) {
  const container = document.getElementById("singleResultContainer");
  if (!container) return;

  const isFake = data.prediction === "Potentially Fake";
  const fakePct = data.ensemble_fake_pct;
  const genuinePct = data.ensemble_genuine_pct;

  container.innerHTML = `
    <div class="result-card ${isFake ? 'fake' : 'genuine'}">
      <div class="result-header">
        <div class="result-verdict">
          <span class="verdict-badge ${isFake ? 'fake' : 'genuine'}">
            ${isFake ? '⚠️ Potentially Fake' : '✓ Potentially Genuine'}
          </span>
        </div>
        <span class="ensemble-badge">
          Primary Ensemble (55% XGB + 45% RoBERTa)
        </span>
      </div>

      <!-- Two Compact Probability Cards (Section 19) -->
      <div class="prob-cards-grid">
        <div class="prob-card fake">
          <div class="prob-card-header">
            <span class="prob-card-title">Fake Probability</span>
            <span class="prob-card-tag">${isFake ? 'Flagged Risk' : 'Low Risk'}</span>
          </div>
          <div class="prob-card-value">${fakePct}%</div>
          <div class="prob-card-sub">Decision cutoff: 54.0%</div>
        </div>
        <div class="prob-card genuine">
          <div class="prob-card-header">
            <span class="prob-card-title">Genuine Probability</span>
            <span class="prob-card-tag">${!isFake ? 'Verified Organic' : 'Unlikely Organic'}</span>
          </div>
          <div class="prob-card-value">${genuinePct}%</div>
          <div class="prob-card-sub">Organic pattern match</div>
        </div>
      </div>

      <!-- Horizontal Segmented Probability Bar (Section 19) -->
      <div class="prob-bar-container">
        <div class="prob-complement-bar">
          <div class="prob-segment-fake" style="width: ${fakePct}%" title="Fake: ${fakePct}%"></div>
          <div class="prob-segment-genuine" style="width: ${genuinePct}%" title="Genuine: ${genuinePct}%"></div>
        </div>
        <div class="prob-legend">
          <span class="fake-leg">Fake: ${fakePct}%</span>
          <span class="threshold-indicator">Decision Cutoff: 54%</span>
          <span class="genuine-leg">Genuine: ${genuinePct}%</span>
        </div>
      </div>

      <div class="disclaimer-box">
        <strong>Probabilistic Intelligence Note:</strong>
        Classification derived from empirical linguistic stylometry, temporal burst density, and RoBERTa semantics. Threshold: 0.54.
      </div>

      <!-- All Four Individual Models Breakdown -->
      <div style="margin-top: 14px;">
        <div style="font-size: 0.8rem; font-weight: 650; color: var(--text-heading); margin-bottom: 6px; display:flex; justify-content:space-between; align-items:center;">
          <span>All 4 Model Outputs</span>
          <span style="font-size:0.7rem; color:var(--text-dim); font-weight:normal;">Component Cutoff: 0.50</span>
        </div>
        <div class="models-grid">
          ${renderModelBox(data.models_breakdown.xgboost)}
          ${renderModelBox(data.models_breakdown.roberta)}
          ${renderModelBox(data.models_breakdown.bert)}
          ${renderModelBox(data.models_breakdown.deberta)}
        </div>
      </div>

      <!-- Actions -->
      <div style="margin-top: 14px; display: flex; flex-direction: column; gap: 8px;">
        <button class="btn btn-primary" style="width: 100%; justify-content: center; font-weight: 600; padding: 7px 12px; font-size: 0.82rem;" onclick="downloadSinglePdf('${data.prediction_id}')">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
          Download Complete Report (PDF)
        </button>
        <div style="display: flex; gap: 8px;">
          <button class="btn btn-secondary" style="flex: 1; justify-content: center; font-size: 0.78rem; padding: 6px 10px;" onclick="switchTab('history')">
            View in History
          </button>
          <button class="btn btn-secondary" style="flex: 1.2; justify-content: center; font-size: 0.78rem; padding: 6px 10px;" onclick="navigateToXai()">
            Inspect TreeSHAP Signals →
          </button>
        </div>
      </div>
    </div>
  `;
}

function renderModelBox(model) {
  if (!model) return '';
  const isFake = model.prediction === "Potentially Fake";
  const isStandby = model.status === "standby" || model.fake_pct === null || model.fake_pct === undefined;
  const probDisplay = isStandby ? '<span style="font-size:0.8rem; color:var(--text-dim); font-weight:normal;">Standby</span>' : `${model.fake_pct}%`;
  const tagClass = isStandby ? 'neutral' : (isFake ? 'fake' : 'genuine');
  const tagLabel = isStandby ? 'Standby' : model.prediction;
  return `
    <div class="model-box">
      <div>
        <div class="model-name">${escapeHtml(model.name)}</div>
        <div class="model-prob">${probDisplay}</div>
      </div>
      <span class="model-tag ${tagClass}">${tagLabel}</span>
    </div>
  `;
}

function navigateToXai() {
  switchTab("explainable-ai");
  switchXaiSubtab("dynamic");
}

async function loadRandomSampleIntoForm() {
  try {
    const res = await fetch("/api/sample-reviews");
    const data = await res.json();
    if (data.samples && data.samples.length > 0) {
      const rand = data.samples[Math.floor(Math.random() * data.samples.length)];
      document.getElementById("singleTitle").value = rand.title || "";
      document.getElementById("singleText").value = rand.text || "";
      document.getElementById("singleRating").value = rand.rating || 5;
      updateStars(rand.rating || 5);
      document.getElementById("singleDate").value = rand.date || new Date().toISOString().split("T")[0];
      document.getElementById("singleReviewerId").value = rand.reviewer_id || "";
      document.getElementById("singleProductId").value = rand.product_id || "";
    }
  } catch (e) {
    console.error(e);
  }
}

/* ================= BULK REVIEW ANALYSIS (SECTIONS 8 & 20) ================= */
function initDragAndDrop() {
  const dropzone = document.getElementById("bulkDropzone");
  const fileInput = document.getElementById("bulkFileInput");
  const btnBrowse = document.getElementById("btnBrowseBulk");

  if (!dropzone || !fileInput) return;

  btnBrowse.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => handleFileSelected(e.target.files[0]));

  ["dragenter", "dragover"].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files.length > 0) {
      handleFileSelected(dt.files[0]);
    }
  });
}

function handleFileSelected(file) {
  if (!file) return;
  bulkActiveFile = file;

  const reader = new FileReader();
  reader.onload = function(e) {
    const text = e.target.result;
    const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
    if (lines.length <= 1) {
      alert("CSV appears empty or has only a header row.");
      return;
    }
    const cols = lines[0].split(",").map(c => c.trim().replace(/^["']|["']$/g, ""));
    openColumnModal(cols);
  };
  reader.readAsText(file.slice(0, 5000));
}

function openColumnModal(columns) {
  const modal = document.getElementById("columnMappingModal");
  const selects = ["mapTitle", "mapText", "mapRating", "mapDate", "mapReviewerId", "mapProductId"];

  selects.forEach(id => {
    const sel = document.getElementById(id);
    if (!sel) return;
    sel.innerHTML = `<option value="">-- Ignore --</option>`;
    columns.forEach(col => {
      const opt = document.createElement("option");
      opt.value = col;
      opt.textContent = col;
      sel.appendChild(opt);
    });
  });

  preselectMapping(columns);
  if (modal) modal.classList.add("active");
}

function preselectMapping(columns) {
  const lower = columns.map(c => c.toLowerCase());
  const mapRules = {
    mapTitle: ["review_title", "title", "headline", "summary"],
    mapText: ["review_text", "text", "body", "content", "review"],
    mapRating: ["rating", "stars", "score", "star_rating"],
    mapDate: ["review_date", "date", "timestamp"],
    mapReviewerId: ["reviewer_id", "reviewerid", "user_id", "author"],
    mapProductId: ["product_id", "asin", "item_id", "sku"]
  };

  Object.keys(mapRules).forEach(selId => {
    const sel = document.getElementById(selId);
    if (!sel) return;
    for (let candidate of mapRules[selId]) {
      const idx = lower.indexOf(candidate);
      if (idx !== -1) {
        sel.value = columns[idx];
        break;
      }
    }
  });
}

function closeColumnModal() {
  const modal = document.getElementById("columnMappingModal");
  if (modal) modal.classList.remove("active");
}

async function confirmColumnMappingAndAnalyze() {
  const mapping = {
    review_title: document.getElementById("mapTitle").value,
    review_text: document.getElementById("mapText").value,
    rating: document.getElementById("mapRating").value,
    review_date: document.getElementById("mapDate").value,
    reviewer_id: document.getElementById("mapReviewerId").value,
    product_id: document.getElementById("mapProductId").value
  };

  if (!mapping.review_text) {
    alert("Review Text is a mandatory mapped column.");
    return;
  }

  closeColumnModal();
  runBulkAnalysisWithMapping(mapping);
}

async function runBulkAnalysisWithMapping(mapping) {
  const loading = document.getElementById("bulkLoadingIndicator");
  const summary = document.getElementById("bulkSummarySection");

  if (loading) loading.style.display = "block";
  if (summary) summary.style.display = "none";

  const formData = new FormData();
  formData.append("file", bulkActiveFile);
  formData.append("mapping", JSON.stringify(mapping));

  try {
    const res = await fetch("/api/predict/bulk", {
      method: "POST",
      body: formData
    });
    if (!res.ok) throw new Error("Bulk upload processing error");
    const data = await res.json();
    bulkRawData = data;
    bulkFilteredResults = data.results || [];
    bulkCurrentPage = 1;

    // Update KPI summary cards
    document.getElementById("bulkTotalUploaded").textContent = data.total_uploaded.toLocaleString();
    document.getElementById("bulkFakeCount").textContent = data.fake_count.toLocaleString();
    document.getElementById("bulkFakePct").textContent = `${data.fake_percentage}% of batch`;
    document.getElementById("bulkGenuineCount").textContent = data.genuine_count.toLocaleString();
    document.getElementById("bulkGenuinePct").textContent = `${data.genuine_percentage}% of batch`;

    renderBulkTable();
    if (summary) summary.style.display = "block";
    loadDashboardStats(); // refresh database counts
  } catch (err) {
    console.error("Bulk processing failed:", err);
    alert("Bulk processing failed: " + err.message);
  } finally {
    if (loading) loading.style.display = "none";
  }
}

function filterBulkTable() {
  const term = (document.getElementById("bulkFilterInput").value || "").toLowerCase();
  const verdict = document.getElementById("bulkVerdictFilter").value;

  if (!bulkRawData) return;

  bulkFilteredResults = bulkRawData.results.filter(item => {
    const matchTerm = !term ||
      (item.review_title && item.review_title.toLowerCase().includes(term)) ||
      (item.preview && item.preview.toLowerCase().includes(term));
    const matchVerdict = verdict === "ALL" || item.prediction === verdict;
    return matchTerm && matchVerdict;
  });

  bulkCurrentPage = 1;
  renderBulkTable();
}

function renderBulkTable() {
  const tbody = document.getElementById("bulkTableBody");
  if (!tbody) return;

  const total = bulkFilteredResults.length;
  const startIdx = (bulkCurrentPage - 1) * BULK_PAGE_SIZE;
  const endIdx = Math.min(startIdx + BULK_PAGE_SIZE, total);
  const pageItems = bulkFilteredResults.slice(startIdx, endIdx);

  tbody.innerHTML = "";
  if (pageItems.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding: 20px;">No matching reviews found.</td></tr>`;
  } else {
    pageItems.forEach(item => {
      const isFake = item.prediction === "Potentially Fake";
      const fakePct = item.fake_pct;
      const genPct = item.genuine_pct;

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${item.row_id}</td>
        <td style="color:#F59E0B; white-space:nowrap;">★ ${item.rating}</td>
        <td>
          <div style="font-weight:600; color:var(--text-heading); font-size:0.85rem;">${escapeHtml(item.review_title || 'Untitled')}</div>
          <div style="font-size:0.75rem; color:var(--text-muted); max-width:280px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
            ${escapeHtml(item.preview || '')}
          </div>
        </td>
        <td><strong style="color:var(--fake-red);">${fakePct}%</strong></td>
        <td><strong style="color:var(--genuine-green);">${genPct}%</strong></td>
        <td>
          <div class="table-prob-bar-wrapper">
            <div class="table-prob-bar" title="Fake: ${fakePct}%, Genuine: ${genPct}%">
              <div class="table-prob-seg-fake" style="width: ${fakePct}%;"></div>
              <div class="table-prob-seg-gen" style="width: ${genPct}%;"></div>
            </div>
            <span class="table-prob-label" style="color: ${isFake ? 'var(--fake-red)' : 'var(--genuine-green)'};">${fakePct}%</span>
          </div>
        </td>
        <td>
          <span class="sample-badge ${isFake ? 'fake' : 'genuine'}">${item.prediction}</span>
        </td>
        <td>
          <button class="btn btn-secondary btn-sm" onclick="inspectBulkReviewInSingle(${item.row_id})">Inspect</button>
        </td>
      `;
      tbody.appendChild(tr);
    });
  }

  const pageInfo = document.getElementById("bulkPaginationInfo");
  if (pageInfo) {
    pageInfo.textContent = total > 0 ? `Showing ${startIdx + 1}-${endIdx} of ${total} records` : "Showing 0-0 of 0";
  }
}

function changeBulkPage(delta) {
  const totalPages = Math.ceil(bulkFilteredResults.length / BULK_PAGE_SIZE);
  const newPage = bulkCurrentPage + delta;
  if (newPage >= 1 && newPage <= totalPages) {
    bulkCurrentPage = newPage;
    renderBulkTable();
  }
}

function inspectBulkReviewInSingle(rowId) {
  if (!bulkRawData || !bulkRawData.results) return;
  const match = bulkRawData.results.find(r => r.row_id === rowId);
  if (!match) return;

  document.getElementById("singleTitle").value = match.review_title || "";
  document.getElementById("singleText").value = match.preview || "";
  document.getElementById("singleRating").value = match.rating || 5;
  updateStars(match.rating || 5);
  document.getElementById("singleDate").value = match.review_date || new Date().toISOString().split("T")[0];
  document.getElementById("singleReviewerId").value = match.reviewer_id || "";
  document.getElementById("singleProductId").value = match.product_id || "";

  switchTab("single-review");
  predictSingleReview();
}

async function exportBulkToCSV() {
  if (!bulkRawData || !bulkRawData.results) return;
  try {
    const res = await fetch("/api/export/bulk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ results: bulkRawData.results })
    });
    if (!res.ok) throw new Error("Export failed");
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `reviewshield_bulk_${bulkRawData.batch_id || 'results'}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } catch (err) {
    alert("Export failed: " + err.message);
  }
}

/* ================= EXPLAINABLE AI MODULE (SECTIONS 9 & 17) ================= */
function switchXaiSubtab(mode) {
  currentXaiSubtab = mode;
  const tabs = ["global", "samples", "dynamic"];
  tabs.forEach(t => {
    const btn = document.getElementById(`btnXaiSub${t.charAt(0).toUpperCase() + t.slice(1)}`);
    const view = document.getElementById(`xaiSubView${t.charAt(0).toUpperCase() + t.slice(1)}`);
    if (btn) {
      if (t === mode) btn.classList.add("active");
      else btn.classList.remove("active");
    }
    if (view) {
      view.style.display = t === mode ? "block" : "none";
    }
  });

  if (mode === "global") {
    loadXaiGlobal();
  } else if (mode === "samples") {
    loadXaiSamplesList();
  } else if (mode === "dynamic") {
    if (currentSingleResult) populateExplainableAiTab(currentSingleResult);
  }
}

async function loadXaiGlobal() {
  if (xaiGlobalData) {
    renderXaiGlobalView(xaiGlobalData);
    return;
  }
  try {
    const res = await fetch("/api/shap/global?limit=15");
    if (!res.ok) throw new Error("Could not load global SHAP data.");
    const data = await res.json();
    xaiGlobalData = data;
    renderXaiGlobalView(data);
  } catch (err) {
    console.error("Global SHAP error:", err);
  }
}

function renderXaiGlobalView(features) {
  // 1. Render Table
  const tbody = document.getElementById("xaiGlobalTableBody");
  if (tbody) {
    tbody.innerHTML = "";
    features.forEach(f => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>#${f.rank}</strong></td>
        <td><code>${f.feature}</code></td>
        <td><span class="badge-source single">${f.category}</span></td>
        <td><strong style="color:var(--primary-blue); font-size:0.85rem;">${f.mean_absolute_shap}</strong></td>
        <td style="color:var(--text-body); font-size:0.75rem; line-height:1.4;">${f.description}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // 2. Render Horizontal Bar Chart
  const ctx = document.getElementById("chartXaiGlobal");
  if (!ctx) return;

  const top10 = features.slice(0, 10).reverse();
  const labels = top10.map(f => f.feature);
  const values = top10.map(f => f.mean_absolute_shap);

  if (chartXaiGlobal) chartXaiGlobal.destroy();

  chartXaiGlobal = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Mean Absolute SHAP Value (Impact Magnitude)",
        data: values,
        backgroundColor: "#0D9488",
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          grid: { color: "#E2E8F0" },
          ticks: { font: { size: 10 }, color: "#64748B" }
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 10, family: "monospace" }, color: "#172338" }
        }
      }
    }
  });
}

async function loadXaiSamplesList() {
  const select = document.getElementById("selectShapSample");
  if (!select) return;

  if (xaiSamplesData) {
    return;
  }

  try {
    const res = await fetch("/api/shap/samples?limit=10");
    if (!res.ok) throw new Error("Could not load sampled SHAP data.");
    const samples = await res.json();
    xaiSamplesData = samples;

    select.innerHTML = "";
    samples.forEach(s => {
      const opt = document.createElement("option");
      opt.value = s.sample_id;
      opt.textContent = `Sample #${s.sample_id} – ${s.predicted_class} (${(s.fake_probability * 100).toFixed(1)}%)`;
      select.appendChild(opt);
    });

    loadSelectedShapSample();
  } catch (err) {
    console.error("SHAP samples error:", err);
  }
}

async function loadSelectedShapSample() {
  const select = document.getElementById("selectShapSample");
  if (!select) return;
  const sampleId = parseInt(select.value);

  try {
    const res = await fetch(`/api/shap/sample/${sampleId}`);
    if (!res.ok) throw new Error("Sample lookup failed");
    const data = await res.json();

    const isFake = data.predicted_class === "Fake";
    const fakePct = (data.fake_probability * 100).toFixed(1);

    // 1. Meta Card
    const metaCard = document.getElementById("xaiSampleMetaCard");
    if (metaCard) {
      metaCard.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div>
            <span class="sample-badge ${isFake ? 'fake' : 'genuine'}">${isFake ? '⚠️ Flagged Deceptive' : '✓ Verified Organic'}</span>
            <span style="font-size:0.85rem; font-weight:600; margin-left:8px; color:var(--text-heading);">Sample Review #${data.sample_id}</span>
          </div>
          <div style="font-size:0.85rem; color:var(--text-heading);">
            Model Fake Probability: <strong style="color:${isFake ? 'var(--fake-red)' : 'var(--genuine-green)'}; font-size:1rem;">${fakePct}%</strong> (Cutoff: 54%)
          </div>
        </div>
      `;
    }

    // 2. Waterfall Chart
    const ctx = document.getElementById("chartXaiSampleWaterfall");
    if (ctx) {
      if (chartXaiSampleWaterfall) chartXaiSampleWaterfall.destroy();

      const sortedFeatures = [...data.features].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)).slice(0, 8);
      const labels = sortedFeatures.map(f => f.feature);
      const values = sortedFeatures.map(f => f.shap_value);
      const bgColors = values.map(v => v >= 0 ? "#DC5252" : "#15966A");

      chartXaiSampleWaterfall = new Chart(ctx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [{
            label: "TreeSHAP Contribution Score",
            data: values,
            backgroundColor: bgColors,
            borderRadius: 4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false }
          },
          scales: {
            x: {
              grid: { display: false },
              ticks: { font: { size: 10, family: "monospace" }, color: "#172338" }
            },
            y: {
              grid: { color: "#E2E8F0" },
              ticks: { font: { size: 10 }, color: "#64748B" }
            }
          }
        }
      });
    }

    // 3. Positive vs Negative Panels
    const posContainer = document.getElementById("xaiSamplePosDrivers");
    if (posContainer) {
      posContainer.innerHTML = "";
      if (data.positive_contributions.length === 0) {
        posContainer.innerHTML = `<span style="font-size:0.75rem; color:var(--text-dim);">No significant positive risk factors.</span>`;
      } else {
        data.positive_contributions.slice(0, 5).forEach(f => {
          posContainer.innerHTML += `
            <div style="display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid #EDF2F7; font-size:0.75rem;">
              <div>
                <strong>${f.feature}</strong>: <code>${f.feature_value}</code>
                <div style="font-size:0.7rem; color:var(--text-muted);">${f.description}</div>
              </div>
              <span style="font-weight:700; color:var(--fake-red); margin-left:8px;">+${f.shap_value}</span>
            </div>
          `;
        });
      }
    }

    const negContainer = document.getElementById("xaiSampleNegDrivers");
    if (negContainer) {
      negContainer.innerHTML = "";
      if (data.negative_contributions.length === 0) {
        negContainer.innerHTML = `<span style="font-size:0.75rem; color:var(--text-dim);">No significant genuine mitigating factors.</span>`;
      } else {
        data.negative_contributions.slice(0, 5).forEach(f => {
          negContainer.innerHTML += `
            <div style="display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid #EDF2F7; font-size:0.75rem;">
              <div>
                <strong>${f.feature}</strong>: <code>${f.feature_value}</code>
                <div style="font-size:0.7rem; color:var(--text-muted);">${f.description}</div>
              </div>
              <span style="font-weight:700; color:var(--genuine-green); margin-left:8px;">${f.shap_value}</span>
            </div>
          `;
        });
      }
    }
  } catch (err) {
    console.error("Sample details error:", err);
  }
}

function populateExplainableAiTab(data) {
  const xai = data.explainable_ai;
  if (!xai) return;

  // 1. Prediction summary badge
  const sumDiv = document.getElementById("xaiPredictionSummary");
  if (sumDiv) {
    const isFake = data.prediction === "Potentially Fake";
    sumDiv.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; padding:12px 14px; background:var(--bg-subtle); border-radius:var(--radius-sm); border:1px solid var(--border-light);">
        <div>
          <span class="sample-badge ${isFake ? 'fake' : 'genuine'}">${data.prediction}</span>
          <span style="font-size:0.85rem; font-weight:600; margin-left:8px; color:var(--text-heading);">${escapeHtml(data.review_title || 'Untitled Review')}</span>
        </div>
        <div>
          Fake Probability: <strong style="color:${isFake ? 'var(--fake-red)' : 'var(--genuine-green)'}; font-size:0.95rem;">${data.ensemble_fake_pct}%</strong>
        </div>
      </div>
    `;
  }

  // 2. Lexical Pills
  const pillsContainer = document.getElementById("xaiTopSignalsContainer");
  if (pillsContainer) {
    pillsContainer.innerHTML = "";
    if (!xai.top_tfidf_signals || xai.top_tfidf_signals.length === 0) {
      pillsContainer.innerHTML = `<span style="font-size:0.8rem; color:var(--text-dim);">No significant lexical outliers identified.</span>`;
    } else {
      xai.top_tfidf_signals.forEach(sig => {
        const isPos = sig.influence === "Increases fake score";
        const div = document.createElement("div");
        div.style.cssText = `padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 500; display: inline-flex; gap: 6px; align-items: center; background: ${isPos ? '#FEE2E2' : '#D1FAE5'}; color: ${isPos ? '#DC5252' : '#15966A'}; border: 1px solid ${isPos ? '#FECACA' : '#A7F3D0'};`;
        div.innerHTML = `<span><strong>${escapeHtml(sig.word)}</strong></span> <span>${sig.shap > 0 ? '+' : ''}${sig.shap}</span>`;
        pillsContainer.appendChild(div);
      });
    }
  }

  // 3. Dynamic Factors Table
  const tableBody = document.getElementById("xaiGlobalFeaturesTableBody");
  if (tableBody && xai.factors) {
    tableBody.innerHTML = "";
    xai.factors.forEach(f => {
      const isPos = f.influence === "Increases fake score";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${escapeHtml(f.name)}</strong></td>
        <td><code>${escapeHtml(String(f.actual_value))}</code></td>
        <td style="color:${isPos ? 'var(--fake-red)' : 'var(--genuine-green)'}; font-weight:700;">
          ${f.shap_value > 0 ? '+' : ''}${f.shap_value}
        </td>
        <td style="font-size:0.75rem; color:var(--text-body);">${escapeHtml(f.interpretation)}</td>
      `;
      tableBody.appendChild(tr);
    });
  }

  // 4. Waterfall Chart
  const ctx = document.getElementById("chartXaiWaterfall");
  if (ctx && xai.factors) {
    if (chartXaiWaterfall) chartXaiWaterfall.destroy();

    const labels = xai.factors.map(f => f.name.length > 20 ? f.name.substring(0, 18) + '...' : f.name);
    const vals = xai.factors.map(f => f.shap_value);
    const colors = vals.map(v => v >= 0 ? "#DC5252" : "#15966A");

    chartXaiWaterfall = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "XGBoost TreeSHAP Additive Contribution",
          data: vals,
          backgroundColor: colors,
          borderRadius: 3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false }, ticks: { font: { size: 10 }, color: "#172338" } },
          y: { grid: { color: "#E2E8F0" }, ticks: { font: { size: 10 }, color: "#64748B" } }
        }
      }
    });
  }

  // 5. Natural language summary
  const expText = document.getElementById("xaiExplanationText");
  if (expText && xai.natural_language_summary) {
    expText.textContent = xai.natural_language_summary;
  }
}

/* ================= BEHAVIOUR & PATTERNS MODULE (SECTION 10) ================= */
async function loadPatternAnalysis() {
  if (patternAnalysisData) return;

  try {
    const res = await fetch("/api/patterns");
    if (!res.ok) throw new Error("Patterns API error");
    const data = await res.json();
    patternAnalysisData = data;
    renderPatternCharts(data);
  } catch (err) {
    console.error("Pattern analysis error:", err);
  }
}

function renderPatternCharts(data) {
  // Chart 1: Rating Distribution
  const ctxRating = document.getElementById("patternRatingChart") || document.getElementById("chartPatternRating");
  const ratingData = data.rating_distribution;
  if (ctxRating && ratingData && !chartPatternRating) {
    chartPatternRating = new Chart(ctxRating, {
      type: "bar",
      data: {
        labels: ratingData.labels,
        datasets: [
          { label: "Fake Reviews", data: ratingData.fake, backgroundColor: "#DC5252", borderRadius: 3 },
          { label: "Genuine Reviews", data: ratingData.genuine, backgroundColor: "#15966A", borderRadius: 3 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#172338", boxWidth: 12, font: { size: 11 } } } },
        scales: {
          x: { grid: { display: false }, ticks: { font: { size: 10 }, color: "#64748B" } },
          y: { grid: { color: "#E2E8F0" }, ticks: { font: { size: 10 }, color: "#64748B" } }
        }
      }
    });
  }

  // Chart 2: Timing / Temporal Gaps
  const ctxTiming = document.getElementById("patternTimingChart") || document.getElementById("chartPatternTiming");
  const timingData = data.temporal_gaps || data.timing_gaps;
  if (ctxTiming && timingData && !chartPatternTiming) {
    chartPatternTiming = new Chart(ctxTiming, {
      type: "bar",
      data: {
        labels: timingData.labels,
        datasets: [
          { label: "Fake Reviews", data: timingData.fake, backgroundColor: "#DC5252", borderRadius: 3 },
          { label: "Genuine Reviews", data: timingData.genuine, backgroundColor: "#15966A", borderRadius: 3 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#172338", boxWidth: 12, font: { size: 11 } } } },
        scales: {
          x: { grid: { display: false }, ticks: { font: { size: 10 }, color: "#64748B" } },
          y: { grid: { color: "#E2E8F0" }, ticks: { font: { size: 10 }, color: "#64748B" } }
        }
      }
    });
  }

  // Chart 3: Burst Activity
  const ctxBurst = document.getElementById("patternBurstChart") || document.getElementById("chartPatternBurst");
  const burstData = data.burst_patterns_7day || data.burst_behavior;
  if (ctxBurst && burstData && !chartPatternBurst) {
    chartPatternBurst = new Chart(ctxBurst, {
      type: "bar",
      data: {
        labels: burstData.labels,
        datasets: [
          { label: "Fake Reviews", data: burstData.fake, backgroundColor: "#DC5252", borderRadius: 3 },
          { label: "Genuine Reviews", data: burstData.genuine, backgroundColor: "#15966A", borderRadius: 3 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#172338", boxWidth: 12, font: { size: 11 } } } },
        scales: {
          x: { grid: { display: false }, ticks: { font: { size: 10 }, color: "#64748B" } },
          y: { grid: { color: "#E2E8F0" }, ticks: { font: { size: 10 }, color: "#64748B" } }
        }
      }
    });
  }

  // Chart 4: Multi-Product Activity
  const ctxProducts = document.getElementById("patternProductsChart") || document.getElementById("chartPatternProducts");
  const productData = data.product_diversity || data.reviewer_products;
  if (ctxProducts && productData && !chartPatternProducts) {
    chartPatternProducts = new Chart(ctxProducts, {
      type: "bar",
      data: {
        labels: productData.labels,
        datasets: [
          { label: "Fake Reviews", data: productData.fake, backgroundColor: "#DC5252", borderRadius: 3 },
          { label: "Genuine Reviews", data: productData.genuine, backgroundColor: "#15966A", borderRadius: 3 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#172338", boxWidth: 12, font: { size: 11 } } } },
        scales: {
          x: { grid: { display: false }, ticks: { font: { size: 10 }, color: "#64748B" } },
          y: { grid: { color: "#E2E8F0" }, ticks: { font: { size: 10 }, color: "#64748B" } }
        }
      }
    });
  }
}

/* ================= MODEL INSIGHTS MODULE (SECTIONS 11 & 18) ================= */
async function loadModelInsights() {
  if (modelInsightsData) {
    switchInsightGraph(currentInsightGraph);
    renderSelectedConfusionMatrix();
    return;
  }

  try {
    const res = await fetch("/api/model-insights");
    if (!res.ok) throw new Error("Model insights error");
    const data = await res.json();
    modelInsightsData = data;

    renderSelectedConfusionMatrix();
    switchInsightGraph("roc");
  } catch (err) {
    console.error("Model insights error:", err);
  }
}

function selectModelInsight(modelKey) {
  // Update active state on cards
  const cards = document.querySelectorAll(".model-select-card");
  cards.forEach(c => {
    if (c.getAttribute("data-model") === modelKey) c.classList.add("active");
    else c.classList.remove("active");
  });

  const sel = document.getElementById("selectConfusionModel");
  if (sel) {
    sel.value = modelKey;
    renderSelectedConfusionMatrix();
  }
}

function renderSelectedConfusionMatrix() {
  const container = document.getElementById("confusionMatrixContainer");
  const selEl = document.getElementById("selectConfusionModel");
  if (!container || !selEl || !modelInsightsData) return;

  const selVal = selEl.value;
  let cm, name, split;
  const valModels = modelInsightsData.validation_benchmark.models;
  const testData = modelInsightsData.held_out_test;

  if (selVal === "ensemble_test") {
    cm = testData.ensemble_metrics.confusion_matrix;
    name = "Primary Ensemble (55% XGB + 45% RoBERTa)";
    split = "Official Held-Out Test Set (6,665 reviews)";
  } else if (selVal === "xgboost_full_test") {
    cm = testData.xgboost_full_test_metrics.confusion_matrix;
    name = "Tuned XGBoost (Standalone)";
    split = "Official Held-Out Test Set (6,665 reviews)";
  } else if (valModels[selVal]) {
    cm = valModels[selVal].metrics.confusion_matrix;
    name = valModels[selVal].name;
    split = "Validation Benchmark (1,000 reviews)";
  } else {
    return;
  }

  const titleEl = document.getElementById("confMatrixTitle");
  if (titleEl) titleEl.textContent = `Confusion Matrix: ${name} (${split})`;

  const tn = cm[0][0];
  const fp = cm[0][1];
  const fn = cm[1][0];
  const tp = cm[1][1];
  const total = tn + fp + fn + tp;
  const acc = ((tn + tp) / total * 100).toFixed(2);
  const prec = (tp / (tp + fp) * 100).toFixed(2);
  const rec = (tp / (tp + fn) * 100).toFixed(2);
  const f1 = (2 * (prec * rec) / (parseFloat(prec) + parseFloat(rec))).toFixed(2);

  container.innerHTML = `
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; align-items:center;">
      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px;">
        <div style="background:#ECFDF5; border:1px solid #A7F3D0; padding:12px; border-radius:6px; text-align:center;">
          <div style="font-size:10px; color:#047857; font-weight:700;">TRUE NEGATIVE (TN)</div>
          <div style="font-size:20px; font-weight:700; color:#047857;">${tn.toLocaleString()}</div>
          <div style="font-size:10px; color:#047857;">Correct Genuine</div>
        </div>
        <div style="background:#FEF2F2; border:1px solid #FECACA; padding:12px; border-radius:6px; text-align:center;">
          <div style="font-size:10px; color:#B91C1C; font-weight:700;">FALSE POSITIVE (FP)</div>
          <div style="font-size:20px; font-weight:700; color:#B91C1C;">${fp.toLocaleString()}</div>
          <div style="font-size:10px; color:#B91C1C;">False Alarm</div>
        </div>
        <div style="background:#FFFBEB; border:1px solid #FDE68A; padding:12px; border-radius:6px; text-align:center;">
          <div style="font-size:10px; color:#B45309; font-weight:700;">FALSE NEGATIVE (FN)</div>
          <div style="font-size:20px; font-weight:700; color:#B45309;">${fn.toLocaleString()}</div>
          <div style="font-size:10px; color:#B45309;">Missed Fake</div>
        </div>
        <div style="background:#EFF6FF; border:1px solid #BFDBFE; padding:12px; border-radius:6px; text-align:center;">
          <div style="font-size:10px; color:#1D4ED8; font-weight:700;">TRUE POSITIVE (TP)</div>
          <div style="font-size:20px; font-weight:700; color:#1D4ED8;">${tp.toLocaleString()}</div>
          <div style="font-size:10px; color:#1D4ED8;">Correct Fake</div>
        </div>
      </div>
      <div style="display:flex; flex-direction:column; gap:8px; font-size:11.5px;">
        <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #E2E8F0;">
          <span style="color:var(--text-muted);">Empirical Accuracy:</span>
          <strong>${acc}%</strong>
        </div>
        <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #E2E8F0;">
          <span style="color:var(--text-muted);">Precision (Fake Detection):</span>
          <strong>${prec}%</strong>
        </div>
        <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #E2E8F0;">
          <span style="color:var(--text-muted);">Recall (Fake Detection):</span>
          <strong style="color:var(--genuine-green);">${rec}%</strong>
        </div>
        <div style="display:flex; justify-content:space-between; padding:4px 0;">
          <span style="color:var(--text-muted);">Balanced F1-Score:</span>
          <strong style="color:var(--primary-blue);">${f1}%</strong>
        </div>
      </div>
    </div>
  `;
}

function switchInsightGraph(type) {
  currentInsightGraph = type;
  const btnRoc = document.getElementById("btnCurveRoc");
  const btnPr = document.getElementById("btnCurvePr");

  if (btnRoc) btnRoc.className = type === "roc" ? "btn btn-primary btn-sm" : "btn btn-secondary btn-sm";
  if (btnPr) btnPr.className = type === "pr" ? "btn btn-primary btn-sm" : "btn btn-secondary btn-sm";

  const ctx = document.getElementById("chartInsightCurves");
  if (!ctx || !modelInsightsData) return;

  if (chartInsightCurves) chartInsightCurves.destroy();

  const valModels = modelInsightsData.validation_benchmark.models;
  const testData = modelInsightsData.held_out_test;
  const capEl = document.getElementById("insightChartCaption");

  if (type === "roc") {
    if (capEl) {
      capEl.innerHTML = `
        <strong>Receiver Operating Characteristic (ROC):</strong> True Positive Rate vs False Positive Rate.<br>
        Full 6,665-record test set curve for <strong>Tuned XGBoost (ROC-AUC: 91.22%)</strong> is plotted alongside validation benchmarks.<br>
        Official verified Primary Ensemble held-out test ROC-AUC is <strong>93.715%</strong>.
      `;
    }

    chartInsightCurves = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          { label: "Primary Ensemble (Validation - AUC: 0.945)", data: valModels.ensemble.roc_curve, borderColor: "#0D9488", borderWidth: 3, fill: false, tension: 0.2 },
          { label: "Tuned XGBoost (Full Test - AUC: 0.912)", data: testData.xgboost_full_test_metrics.roc_curve, borderColor: "#2563EB", borderWidth: 2.5, fill: false, tension: 0.2 },
          { label: "RoBERTa (Validation - AUC: 0.921)", data: valModels.roberta.roc_curve, borderColor: "#10B981", borderWidth: 2, fill: false, tension: 0.2 },
          { label: "BERT (Validation - AUC: 0.850)", data: valModels.bert.roc_curve, borderColor: "#F59E0B", borderWidth: 1.5, fill: false, tension: 0.2 },
          { label: "DeBERTa (Validation - AUC: 0.859)", data: valModels.deberta.roc_curve, borderColor: "#EC4899", borderWidth: 1.5, fill: false, tension: 0.2 },
          { label: "Random Baseline", data: [{x:0,y:0},{x:1,y:1}], borderColor: "rgba(0,0,0,0.2)", borderDash: [4,4], borderWidth: 1, fill: false }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#172338", boxWidth: 12, font: { size: 10.5 } } } },
        scales: {
          x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False Positive Rate (FPR)", font: { size: 10 } }, grid: { color: "#E2E8F0" } },
          y: { type: "linear", min: 0, max: 1, title: { display: true, text: "True Positive Rate (TPR / Recall)", font: { size: 10 } }, grid: { color: "#E2E8F0" } }
        }
      }
    });
  } else if (type === "pr") {
    if (capEl) {
      capEl.innerHTML = `
        <strong>Precision-Recall (PR) Curves:</strong> Precision vs Recall trade-off across classification thresholds.<br>
        Full 6,665-record test set PR curve for <strong>Tuned XGBoost (PR-AUC: 95.89%)</strong> plotted alongside validation benchmarks.<br>
        Official verified Primary Ensemble test PR-AUC is <strong>96.935%</strong>.
      `;
    }

    chartInsightCurves = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          { label: "Primary Ensemble (Validation - PR-AUC: 0.958)", data: valModels.ensemble.pr_curve, borderColor: "#0D9488", borderWidth: 3, fill: false, tension: 0.2 },
          { label: "Tuned XGBoost (Full Test - PR-AUC: 0.959)", data: testData.xgboost_full_test_metrics.pr_curve, borderColor: "#2563EB", borderWidth: 2.5, fill: false, tension: 0.2 },
          { label: "RoBERTa (Validation - PR-AUC: 0.951)", data: valModels.roberta.pr_curve, borderColor: "#10B981", borderWidth: 2, fill: false, tension: 0.2 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#172338", boxWidth: 12, font: { size: 10.5 } } } },
        scales: {
          x: { type: "linear", min: 0, max: 1, title: { display: true, text: "Recall", font: { size: 10 } }, grid: { color: "#E2E8F0" } },
          y: { type: "linear", min: 0, max: 1, title: { display: true, text: "Precision", font: { size: 10 } }, grid: { color: "#E2E8F0" } }
        }
      }
    });
  }
}

/* ================= PREDICTION HISTORY MODULE (SECTION 12) ================= */
function debounceHistorySearch() {
  clearTimeout(historySearchTimeout);
  historySearchTimeout = setTimeout(() => {
    loadHistoryPage(0);
  }, 300);
}

async function loadHistoryPage(page) {
  historyCurrentPage = page;
  const search = document.getElementById("historySearchInput") ? document.getElementById("historySearchInput").value.trim() : "";
  const cat = document.getElementById("historyCategoryFilter") ? document.getElementById("historyCategoryFilter").value : "ALL";
  const src = document.getElementById("historySourceFilter") ? document.getElementById("historySourceFilter").value : "ALL";

  const params = new URLSearchParams({
    limit: HISTORY_PAGE_SIZE,
    offset: page * HISTORY_PAGE_SIZE
  });
  if (search) params.append("search", search);
  if (cat && cat !== "ALL") params.append("category", cat);
  if (src && src !== "ALL") params.append("source", src);

  try {
    const res = await fetch(`/api/history?${params.toString()}`);
    if (!res.ok) throw new Error("History fetch error");
    const data = await res.json();
    renderHistoryTable(data);
  } catch (err) {
    console.error("History error:", err);
  }
}

function renderHistoryTable(data) {
  const tbody = document.getElementById("historyTableBody");
  if (!tbody) return;

  const records = data.records || [];
  const total = data.total || 0;

  tbody.innerHTML = "";
  if (records.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; padding: 24px; color: var(--text-dim);">No matching records found.</td></tr>`;
  } else {
    records.forEach(item => {
      const isFake = item.verdict === "Potentially Fake";
      const fakePct = item.fake_prob !== null && item.fake_prob !== undefined ? (item.fake_prob * 100).toFixed(1) + "%" : "N/A";
      const genPct = item.genuine_prob !== null && item.genuine_prob !== undefined ? (item.genuine_prob * 100).toFixed(1) + "%" : "N/A";
      const ratingVal = item.rating || 5;
      const starsStr = "★".repeat(Math.min(5, Math.max(1, Math.round(ratingVal))));
      const predId = item.prediction_id || `PRED_${item.id}`;
      const timeStr = item.analysis_timestamp ? String(item.analysis_timestamp).slice(0, 19).replace('T', ' ') : 'Recent';
      const isBulk = item.source === "bulk";

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><code>${predId}</code></td>
        <td style="white-space:nowrap; color:var(--text-muted); font-size:0.75rem;">${timeStr}</td>
        <td>
          <div style="font-weight:600; color:var(--text-heading); font-size:0.85rem;">${escapeHtml(item.title || 'Untitled')}</div>
        </td>
        <td>
          <div style="font-size:0.75rem; color:var(--text-muted); max-width:240px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(item.text || '')}</div>
        </td>
        <td style="color:#F59E0B; white-space:nowrap;">${starsStr}</td>
        <td>
          <span class="sample-badge ${isFake ? 'fake' : 'genuine'}">${item.verdict}</span>
        </td>
        <td><strong style="color:var(--fake-red);">${fakePct}</strong></td>
        <td><strong style="color:var(--genuine-green);">${genPct}</strong></td>
        <td>
          <span class="badge-source ${isBulk ? 'bulk' : 'single'}">${isBulk ? 'Bulk' : 'Single'}</span>
        </td>
        <td>
          <div style="display:flex; gap:4px;">
            <button class="btn btn-secondary btn-sm" onclick="inspectHistoryRecord('${predId}')">Inspect</button>
            <button class="btn btn-pdf btn-sm" onclick="downloadSinglePdf('${predId}')">PDF</button>
          </div>
        </td>
      `;
      tbody.appendChild(tr);
    });
  }

  const start = historyCurrentPage * HISTORY_PAGE_SIZE + 1;
  const end = Math.min((historyCurrentPage + 1) * HISTORY_PAGE_SIZE, total);
  const infoEl = document.getElementById("historyPaginationInfo");
  if (infoEl) infoEl.textContent = total > 0 ? `Showing ${start}-${end} of ${total} records` : "Showing 0-0 of 0";

  const btnPrev = document.getElementById("btnPrevHistory");
  if (btnPrev) btnPrev.disabled = historyCurrentPage === 0;

  const btnNext = document.getElementById("btnNextHistory");
  if (btnNext) btnNext.disabled = end >= total;
}

function changeHistoryPage(delta) {
  const newPage = historyCurrentPage + delta;
  if (newPage >= 0) {
    loadHistoryPage(newPage);
  }
}

async function inspectHistoryRecord(predId) {
  try {
    const res = await fetch(`/api/prediction/${encodeURIComponent(predId)}`);
    if (!res.ok) throw new Error("Could not load details");
    const data = await res.json();
    currentModalPredictionId = data.prediction_id;

    document.getElementById("histModalTitle").textContent = data.review_title || "Review Prediction Dossier";
    document.getElementById("histModalSubtitle").textContent = `ID: ${data.prediction_id} • Timestamp: ${data.analysis_timestamp || 'N/A'}`;

    const predLabel = data.ensemble_prediction || data.prediction || (data.verdict || "Potentially Fake");
    const isFake = predLabel === "Potentially Fake";
    const fakePct = data.ensemble_fake_prob !== undefined && data.ensemble_fake_prob !== null ? (data.ensemble_fake_prob * 100).toFixed(1) : "N/A";
    const genPct = data.ensemble_genuine_prob !== undefined && data.ensemble_genuine_prob !== null ? (data.ensemble_genuine_prob * 100).toFixed(1) : "N/A";

    document.getElementById("histModalBody").innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <span class="sample-badge ${isFake ? 'fake' : 'genuine'}" style="font-size:0.85rem; padding:4px 10px;">${predLabel}</span>
        <div>
          Ensemble Fake Prob: <strong style="color:${isFake ? 'var(--fake-red)' : 'var(--genuine-green)'}; font-size:1.05rem;">${fakePct}%</strong> (Threshold: 0.54)
        </div>
      </div>
      <div style="background:var(--bg-subtle); padding:10px 12px; border-radius:6px; margin-bottom:12px; font-size:0.82rem; line-height:1.45; border:1px solid var(--border-light);">
        ${escapeHtml(data.review_text || '')}
      </div>
      <div style="font-size:0.75rem; color:var(--text-muted);">
        Reviewer Account: <code>${escapeHtml(data.reviewer_id || 'Not Provided')}</code> | Product ASIN: <code>${escapeHtml(data.product_id || 'Not Provided')}</code> | Rating: ★ ${data.rating}
      </div>
    `;

    const modal = document.getElementById("historyDetailModal");
    if (modal) modal.classList.add("active");
  } catch (err) {
    alert("Could not load record: " + err.message);
  }
}

function closeHistoryModal() {
  const modal = document.getElementById("historyDetailModal");
  if (modal) modal.classList.remove("active");
}

function downloadModalPdf() {
  if (currentModalPredictionId) {
    downloadSinglePdf(currentModalPredictionId);
  }
}

function downloadSinglePdf(predId) {
  window.open(`/api/prediction/${encodeURIComponent(predId)}/pdf`, "_blank");
}

/* ================= UTILITY HELPERS ================= */
function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
