/* CUAD highlighter -- main page.
   Flow: choose PDF + model -> /api/parse -> /api/extract -> /api/review/sync
   -> render two panes. The right pane is a review surface: every answer can be
   Approved / Rejected / Edited, answers can be added manually per label, and
   clicking an answer jumps to it on the PDF (left pane). Decisions persist on
   the server per doc_id; "Export to Excel" downloads the kept answers.
   Chips toggle which labels are shown/highlighted; the categories page controls
   which chips exist at all (saved in localStorage). */

const VISIBLE_KEY = "cuad_visible_categories";   // shared with categories.js
const SESSION_KEY = "cuad_last_session";         // restore doc after reload
const SPLIT_KEY = "cuad_split_ratio";            // remembered pane split

const el = (id) => document.getElementById(id);
const escapeHtml = (s) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

const state = {
  categories: [],          // [{label, color, ...}] in CSV order
  colorByLabel: {},        // label -> color
  docId: null,
  filename: null,
  lastResult: null,        // last /api/extract result (for meta + restore)
  spansByLabel: {},        // label -> [verbatim span strings] (model output)
  items: [],               // review items from the server
  itemsByLabel: {},        // label -> [item, ...]
  countByLabel: {},        // label -> #non-rejected items (drives chip badges)
  selected: new Set(),     // labels currently shown/highlighted
  hasResults: false,       // becomes true after the first extraction
};

/* ---- which labels are allowed to appear as chips ----------------------- */
function visibleLabels() {
  const raw = localStorage.getItem(VISIBLE_KEY);
  if (!raw) return state.categories.map((c) => c.label); // default: all
  try {
    const arr = JSON.parse(raw);
    const set = new Set(arr);
    return state.categories.map((c) => c.label).filter((l) => set.has(l));
  } catch {
    return state.categories.map((c) => c.label);
  }
}

/* ---- init -------------------------------------------------------------- */
async function init() {
  const [models, cats] = await Promise.all([
    fetch("/api/models").then((r) => r.json()),
    fetch("/api/categories").then((r) => r.json()),
  ]);

  state.categories = cats;
  cats.forEach((c) => (state.colorByLabel[c.label] = c.color));

  const sel = el("model");
  models.forEach((m) => {
    const o = document.createElement("option");
    o.value = m.id;
    o.textContent = m.label;
    sel.appendChild(o);
  });

  el("file").addEventListener("change", onFilePicked);
  el("analyze").addEventListener("click", () => runAnalyze(true));
  el("rerun").addEventListener("click", () => runAnalyze(false));
  el("selectAll").addEventListener("click", () => setAllSelected(true));
  el("selectNone").addEventListener("click", () => setAllSelected(false));
  el("exportBtn").addEventListener("click", onExport);
  initAddCategory();

  initSplitter();
  initZoom();
  initPdfSearch();

  // Show all category tags up front (fixed legend + selector), like displaCy.
  state.selected = new Set(visibleLabels());
  el("chipsHead").style.display = "flex";
  el("chipsHint").textContent =
    "all 41 clause types · click a tag to toggle · counts appear after Analyze";
  renderChips();

  // If we analysed a document earlier this server session, bring it back.
  restoreSession();
}

function onFilePicked() {
  const f = el("file").files[0];
  if (!f) return;
  el("filePick").textContent = "Change PDF…";
  el("fileName").textContent = f.name + "  (" + Math.round(f.size / 1024) + " KB)";
  el("analyze").disabled = false;
}

/* ---- loading overlay --------------------------------------------------- */
function showOverlay(show) {
  el("overlay").classList.toggle("show", show);
}
function setStep(stage, status) {
  const node = el("step-" + stage);
  if (!node) return;
  node.classList.remove("active", "done");
  if (status) node.classList.add(status);
}
function stageText(t) { el("loaderStage").textContent = t; }

function banner(msg, kind) {
  const b = el("banner");
  b.classList.remove("show", "warn");
  if (!msg) { b.textContent = ""; return; }
  b.textContent = msg;
  if (kind === "warn") b.classList.add("warn");
  b.classList.add("show");
}

/* ---- progress bar ------------------------------------------------------ */
function setProgress(pct, label) {
  el("progressFill").style.width = pct + "%";
  if (label !== undefined)
    el("progressLabel").textContent = label ? `${label} · ${pct}%` : `${pct}%`;
}
function setIndeterminate(on) {
  el("progress").classList.toggle("indeterminate", on);
}

/* Poll the extraction job until it finishes, updating the progress bar. */
function pollJob(jobId) {
  return new Promise((resolve, reject) => {
    setIndeterminate(false);
    const tick = async () => {
      try {
        const r = await fetch("/api/extract_status/" + jobId);
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || "status check failed");
        if (j.stage === "embedding") {
          // spinner until the first batch lands, then a real bar of chunks embedded
          setIndeterminate(!j.completed);
          const epct = j.total ? Math.round((j.completed / j.total) * 100) : 0;
          stageText("Embedding chunks for retrieval (text-embedding-3-small)…");
          setProgress(epct, `embedding chunks ${j.completed} / ${j.total}`);
          setTimeout(tick, 500);
          return;
        }
        setIndeterminate(false);
        const pct = j.total ? Math.round((j.completed / j.total) * 100) : 0;
        const failTxt = j.failed ? ` · ${j.failed} failed` : "";
        const unit = (j.mode === "rag" || j.mode === "hybrid") ? "category" : "chunk";
        setProgress(pct, `${unit} ${j.completed} / ${j.total}${failTxt}`);
        if (j.finished) {
          if (j.error) return reject(new Error(j.error));
          return resolve(j.result);
        }
        setTimeout(tick, 700);
      } catch (e) {
        reject(e);
      }
    };
    tick();
  });
}

/* ---- analyze ----------------------------------------------------------- */
async function runAnalyze(needParse) {
  banner("");
  const model = el("model").value;
  const mode = el("mode").value;   // "scan" (all chunks), "rag" (cosine top-k), or "hybrid"
  showOverlay(true);
  ["parse", "extract", "render"].forEach((s) => setStep(s, ""));
  setIndeterminate(true);
  setProgress(0, "");

  try {
    if (needParse) {
      const f = el("file").files[0];
      if (!f) { banner("Pick a PDF first."); showOverlay(false); return; }
      setStep("parse", "active");
      setIndeterminate(true);
      el("progressLabel").textContent = "parsing PDF…";
      stageText("Parsing PDF with docling (OCR off)… first run may take a while");
      const fd = new FormData();
      fd.append("file", f);
      const pr = await fetch("/api/parse", { method: "POST", body: fd });
      const pj = await pr.json();
      if (!pr.ok) throw new Error(pj.error || "Parse failed");
      state.docId = pj.doc_id;
      state.filename = pj.filename || f.name;

      // Render the actual PDF in the left pane.
      stageText("Rendering PDF…");
      el("panes").style.display = "grid";
      el("emptyState").style.display = "none";
      if (PdfView.ready()) {
        el("progressLabel").textContent = "rendering PDF…";
        await PdfView.load("/api/pdf/" + state.docId, (p, n) =>
          stageText(`Rendering PDF… page ${p} / ${n}`));
        el("pdfNote").textContent = state.filename || "";
        updateZoomLabel();
      } else {
        el("pdfPane").innerHTML =
          '<div class="snip-empty">PDF.js could not load. ' +
          'Highlights still appear in the right pane.</div>';
      }
      setStep("parse", "done");
      el("rerun").classList.remove("hidden");
    } else {
      setStep("parse", "done");
    }

    // Start the background extraction job, then poll it for live progress.
    setStep("extract", "active");
    const modeTxt = mode === "rag" ? " · RAG (top-3 retrieval)"
      : mode === "hybrid" ? " · Hybrid (BM25 top-5 + cosine top-5)" : "";
    stageText("Running " + el("model").selectedOptions[0].textContent + " over the contract" + modeTxt + "…");
    const er = await fetch("/api/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: state.docId, model, mode }),
    });
    const ej = await er.json();
    if (!er.ok) throw new Error(ej.error || "Could not start extraction");

    const result = await pollJob(ej.job_id);
    setStep("extract", "done");

    setStep("render", "active");
    stageText("Rendering highlights…");
    state.lastResult = result;
    state.spansByLabel = result.spans_by_label || {};
    // Merge the model's spans into the server-side review state (keeps any
    // prior decisions on a re-run), then render the review surface.
    await syncReview();
    onResultsReady(result);
    setStep("render", "done");
  } catch (e) {
    banner("Error: " + e.message);
  } finally {
    showOverlay(false);
  }
}

/* Push the model's spans to the server, which returns the merged review items.
   Page numbers are computed here from the rendered PDF text layer. */
async function syncReview() {
  const answers = [];
  for (const [label, spans] of Object.entries(state.spansByLabel)) {
    for (const text of spans) {
      const page = PdfView.ready() ? PdfView.findPage(text) : null;
      answers.push({ label, text, page });
    }
  }
  const r = await fetch(`/api/review/${state.docId}/sync`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ answers }),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "review sync failed");
  setItems(j.items || []);
}

function setItems(items) {
  state.items = items;
  reindexItems();
}

/* Rebuild itemsByLabel + the non-rejected per-label counts from state.items. */
function reindexItems() {
  state.itemsByLabel = {};
  state.countByLabel = {};
  for (const it of state.items) {
    (state.itemsByLabel[it.label] ||= []).push(it);
    if (it.status !== "rejected")
      state.countByLabel[it.label] = (state.countByLabel[it.label] || 0) + 1;
  }
}

/* Selected+visible labels with at least one non-rejected answer, for the PDF. */
function selectedEntries() {
  const out = [];
  for (const label of visibleLabels()) {
    if (!state.selected.has(label)) continue;
    const items = (state.itemsByLabel[label] || []).filter((it) => it.status !== "rejected");
    if (!items.length) continue;
    out.push({ label, color: state.colorByLabel[label] || "#fde047",
               spans: items.map((it) => it.text) });
  }
  return out;
}

/* ---- post-result UI ---------------------------------------------------- */
function onResultsReady(result) {
  el("emptyState").style.display = "none";
  el("panes").style.display = "grid";
  state.hasResults = true;
  reindexItems();
  renderChips();
  applyAll();

  const modelHits = Object.keys(state.spansByLabel).length;
  const totalModelSpans =
    Object.values(state.spansByLabel).reduce((a, s) => a + s.length, 0);
  const cost = (result.estimated_cost_usd ?? 0).toFixed(4);
  const usage = result.usage || { input: 0, output: 0 };
  const isHybrid = result.mode === "hybrid";
  const isRag = result.mode === "rag" || isHybrid;
  // scan: how many chunks succeeded. rag/hybrid: how many category calls succeeded.
  const unitsTxt = result.n_failed
    ? `${result.n_ok}/${isRag ? result.n_labels : result.n_chunks} <span style="color:#a23a3a">(${result.n_failed} failed)</span>`
    : `${isRag ? result.n_labels : result.n_chunks}`;
  const modeLabel = isHybrid
    ? `Hybrid · BM25-${result.hybrid_n} &rarr; cosine-${result.top_k}`
    : `RAG · top-${result.top_k}`;
  const modeSpan = isRag
    ? `<span>Mode: <b>${modeLabel}</b></span>` +
      `<span>Categories queried: <b>${unitsTxt}</b></span>`
    : `<span>Mode: <b>Full scan</b></span>` +
      `<span>Chunks ok: <b>${unitsTxt}</b></span>`;
  el("meta").innerHTML =
    `<span>Model: <b>${escapeHtml(result.model || "")}</b></span>` +
    modeSpan +
    `<span>Clause types found: <b>${modelHits}</b> / 41</span>` +
    `<span>Model spans: <b>${totalModelSpans}</b></span>` +
    `<span>Tokens: <b>${(usage.input + usage.output).toLocaleString()}</b></span>` +
    `<span>Est. cost: <b>$${cost}</b></span>`;
  el("chipsHint").textContent =
    `${modelHits} of 41 clause types found · click a tag to toggle · review answers on the right`;

  persistSession();
  if (result.warning) banner(result.warning, "warn");
}

function renderChips() {
  const wrap = el("chips");
  wrap.innerHTML = "";
  visibleLabels().forEach((label) => {
    const count = state.countByLabel[label] || 0;
    const color = state.colorByLabel[label] || "#ddd";
    const chip = document.createElement("button");
    chip.className = "chip" + (state.hasResults && count === 0 ? " empty" : "");
    chip.style.setProperty("--c", color);
    chip.setAttribute("aria-pressed", state.selected.has(label) ? "true" : "false");
    const countBadge = state.hasResults ? `<span class="count">${count}</span>` : "";
    chip.innerHTML = `<span class="swatch"></span>${escapeHtml(label)}${countBadge}`;
    chip.addEventListener("click", () => {
      if (state.selected.has(label)) state.selected.delete(label);
      else state.selected.add(label);
      chip.setAttribute("aria-pressed", state.selected.has(label) ? "true" : "false");
      applyAll();
    });
    wrap.appendChild(chip);
  });
}

function setAllSelected(on) {
  const labels = visibleLabels();
  state.selected = on ? new Set(labels) : new Set();
  renderChips();
  applyAll();
}

/* Re-render the right pane and re-highlight the PDF for the current selection.
   Called on result-ready and on every tag toggle (no data change). */
function applyAll() {
  if (!state.hasResults) return;
  renderReview();
  if (PdfView.ready()) PdfView.highlight(selectedEntries());
}

/* Full refresh after the review data itself changed (approve/reject/edit/add/
   delete): re-index, update chip counts, re-render, re-highlight, persist. */
function refreshReviewUI() {
  reindexItems();
  renderChips();
  applyAll();
  persistSession();
}

/* ---- right pane: reviewable answers grouped by label ------------------- */
function renderReview() {
  const pane = el("extractPane");
  pane.innerHTML = "";
  const labels = visibleLabels().filter((l) => state.selected.has(l));

  if (!labels.length) {
    pane.innerHTML =
      '<div class="snip-empty">No labels selected. ' +
      "Pick clause tags above to review and edit their answers.</div>";
    el("snipNote").textContent = "";
    return;
  }

  let totalShown = 0;
  for (const label of labels) {
    const items = state.itemsByLabel[label] || [];
    const color = state.colorByLabel[label] || "#ddd";
    const visibleCount = items.filter((it) => it.status !== "rejected").length;
    totalShown += visibleCount;

    const group = document.createElement("div");
    group.className = "snip-group";

    const head = document.createElement("div");
    head.className = "snip-group-head";
    head.innerHTML =
      `<span class="swatch" style="background:${color}"></span>` +
      `${escapeHtml(label)}<span class="count">${visibleCount}</span>`;
    // Per-category AI verification: re-checks every non-rejected answer with the
    // model currently selected in the top dropdown.
    if (visibleCount > 0) {
      const vbtn = document.createElement("button");
      vbtn.className = "rev verify-btn";
      vbtn.innerHTML = "&#129302; AI verify";
      vbtn.title = "Ask the selected model to check whether each answer is correctly labelled";
      vbtn.addEventListener("click", () => verifyLabel(label, vbtn));
      head.appendChild(vbtn);
    }
    group.appendChild(head);

    for (const it of items) group.appendChild(renderItem(it, color));
    group.appendChild(renderAddControl(label));
    pane.appendChild(group);
  }

  el("snipNote").textContent = totalShown
    ? `${totalShown} highlight${totalShown === 1 ? "" : "s"} · ${labels.length} label${labels.length === 1 ? "" : "s"}`
    : `${labels.length} label${labels.length === 1 ? "" : "s"}`;
}

function renderItem(it, color) {
  const div = document.createElement("div");
  div.className = "snip item" + (it.status === "rejected" ? " rejected" : "");
  div.style.setProperty("--c", color);
  div.dataset.id = it.id;

  const text = document.createElement("div");
  text.className = "snip-text";
  text.textContent = it.text;
  if (it.status !== "rejected") {
    text.title = "Click to find this on the PDF";
    text.addEventListener("click", () => jumpToPdf(it.text));
  }
  div.appendChild(text);

  const actions = document.createElement("div");
  actions.className = "snip-actions";

  const mkBtn = (cls, html, on, handler) => {
    const b = document.createElement("button");
    b.className = "rev " + cls + (on ? " on" : "");
    b.innerHTML = html;
    b.addEventListener("click", (e) => { e.stopPropagation(); handler(); });
    return b;
  };
  actions.appendChild(mkBtn("approve", "&#10003; Approve", it.status === "approved",
    () => setStatus(it, "approved")));
  actions.appendChild(mkBtn("reject", "&#10007; Reject", it.status === "rejected",
    () => setStatus(it, "rejected")));
  actions.appendChild(mkBtn("edit", "&#9998; Edit", it.status === "edited",
    () => startEdit(div, it, color)));

  const badge = document.createElement("span");
  badge.className = "status-badge " + it.status;
  badge.textContent = it.status === "pending" ? "unreviewed" : it.status;
  actions.appendChild(badge);

  // AI verification verdict (set by the per-category "AI verify" button).
  if (it.verify && it.verify.verdict) {
    const v = it.verify;
    const icon = { correct: "✓", incorrect: "✗", unsure: "?" }[v.verdict] || "?";
    const vb = document.createElement("span");
    vb.className = "verify-badge " + v.verdict;
    vb.textContent = `${icon} ${v.verdict}`;
    vb.title = `${v.reason || ""}${v.model ? "  (verified by " + v.model + ")" : ""}`.trim();
    actions.appendChild(vb);
  }

  if (it.original === null) {   // manual answer -> allow discarding it
    actions.appendChild(mkBtn("del", "&#10006;", false, () => deleteItem(it)));
  }

  div.appendChild(actions);
  return div;
}

/* Inline editor: swap the answer for a textarea + Save/Cancel. */
function startEdit(div, it, color) {
  div.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "snip-edit";
  const ta = document.createElement("textarea");
  ta.value = it.text;

  const acts = document.createElement("div");
  acts.className = "edit-actions";
  const save = document.createElement("button");
  save.className = "rev edit on";
  save.textContent = "Save";
  const cancel = document.createElement("button");
  cancel.className = "rev";
  cancel.textContent = "Cancel";

  save.addEventListener("click", async () => {
    const text = ta.value.trim();
    if (!text) { ta.focus(); return; }
    const page = PdfView.ready() ? PdfView.findPage(text) : null;
    await patchItem(it, { text, page });   // server sets status edited/manual
  });
  cancel.addEventListener("click", () => applyAll());

  acts.appendChild(save);
  acts.appendChild(cancel);
  wrap.appendChild(ta);
  wrap.appendChild(acts);
  div.appendChild(wrap);
  ta.focus();
}

/* Per-label "add answer manually" control (appears at the end of each group). */
function renderAddControl(label) {
  const wrap = document.createElement("div");
  const btn = document.createElement("button");
  btn.className = "add-answer";
  btn.textContent = "+ Add answer";
  btn.addEventListener("click", () => startAdd(wrap, label));
  wrap.appendChild(btn);
  return wrap;
}

function startAdd(wrap, label) {
  wrap.innerHTML = "";
  const form = document.createElement("div");
  form.className = "add-form";
  const ta = document.createElement("textarea");
  ta.placeholder = `Type an answer for "${label}"…`;

  const acts = document.createElement("div");
  acts.className = "edit-actions";
  const save = document.createElement("button");
  save.className = "rev approve on";
  save.textContent = "Add";
  const cancel = document.createElement("button");
  cancel.className = "rev";
  cancel.textContent = "Cancel";

  save.addEventListener("click", async () => {
    const text = ta.value.trim();
    if (!text) { ta.focus(); return; }
    const page = PdfView.ready() ? PdfView.findPage(text) : null;
    await addItem(label, text, page);
  });
  cancel.addEventListener("click", () => applyAll());

  acts.appendChild(save);
  acts.appendChild(cancel);
  form.appendChild(ta);
  form.appendChild(acts);
  wrap.appendChild(form);
  ta.focus();
}

/* ---- review actions (server-backed) ------------------------------------ */
async function setStatus(it, status) {
  // clicking an already-active status reverts to "unreviewed"
  const next = it.status === status ? "pending" : status;
  await patchItem(it, { status: next });
}

async function patchItem(it, body) {
  try {
    const r = await fetch(`/api/review/${state.docId}/${it.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await r.json();
    if (!r.ok) { banner(j.error || "Update failed"); return; }
    Object.assign(it, j);
    refreshReviewUI();
  } catch (e) {
    banner("Update failed: " + e.message);
  }
}

async function addItem(label, text, page) {
  try {
    const r = await fetch(`/api/review/${state.docId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label, text, page }),
    });
    const j = await r.json();
    if (!r.ok) { banner(j.error || "Add failed"); return; }
    state.items.push(j);
    refreshReviewUI();
  } catch (e) {
    banner("Add failed: " + e.message);
  }
}

/* ---- feature: AI verification (per category) --------------------------- */
async function verifyLabel(label, btn) {
  if (!state.docId) { banner("Analyze a document first."); return; }
  const model = el("model").value;
  const orig = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = "&#8987; verifying…";
  banner("");
  try {
    const r = await fetch(`/api/verify/${state.docId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label, model }),
    });
    const j = await r.json();
    if (!r.ok) { banner(j.error || "Verification failed"); return; }
    // merge the returned verdicts back into our items, then re-render
    const byId = {};
    (j.items || []).forEach((u) => (byId[u.id] = u.verify));
    for (const it of state.items) if (byId[it.id]) it.verify = byId[it.id];
    const n = (j.items || []).length;
    if (!n) banner(j.message || "No answers to verify for this label.", "warn");
    refreshReviewUI();
  } catch (e) {
    banner("Verification failed: " + e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = orig;
  }
}

async function deleteItem(it) {
  try {
    const r = await fetch(`/api/review/${state.docId}/${it.id}`, { method: "DELETE" });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      banner(j.error || "Delete failed");
      return;
    }
    state.items = state.items.filter((x) => x.id !== it.id);
    refreshReviewUI();
  } catch (e) {
    banner("Delete failed: " + e.message);
  }
}

/* ---- feature: click an answer -> locate it on the PDF ------------------ */
function jumpToPdf(text) {
  if (!PdfView.ready()) { banner("PDF viewer isn't available."); return; }
  const found = PdfView.scrollTo(text);
  if (!found) {
    banner("Couldn't locate this answer on the PDF — it may cross a page break " +
      "or differ from the PDF's own text.", "warn");
  } else {
    banner("");
  }
}

/* ---- feature: export to Excel ----------------------------------------- */
function onExport() {
  if (!state.docId || !state.hasResults) {
    banner("Analyze a document first, then export.");
    return;
  }
  // attachment download; doesn't navigate away from the page
  window.location.href = `/api/export/${state.docId}`;
}

/* ---- feature: resizable splitter -------------------------------------- */
let currentSplit = 0.5;
function setSplit(r) {
  currentSplit = r;
  const panes = el("panes");
  panes.style.setProperty("--lw", r + "fr");
  panes.style.setProperty("--rw", (1 - r) + "fr");
}
function initSplitter() {
  const panes = el("panes");
  const splitter = el("splitter");
  if (!panes || !splitter) return;

  const saved = parseFloat(localStorage.getItem(SPLIT_KEY));
  if (saved > 0.1 && saved < 0.9) setSplit(saved);

  let dragging = false;
  const moveTo = (clientX) => {
    const rect = panes.getBoundingClientRect();
    let r = (clientX - rect.left) / rect.width;
    r = Math.max(0.18, Math.min(0.82, r));
    setSplit(r);
  };
  const stop = () => {
    if (!dragging) return;
    dragging = false;
    splitter.classList.remove("dragging");
    document.body.classList.remove("col-resizing");
    localStorage.setItem(SPLIT_KEY, String(currentSplit));
    if (PdfView.ready()) PdfView.refit();   // re-render PDF to the new width
  };

  splitter.addEventListener("mousedown", (e) => {
    dragging = true;
    splitter.classList.add("dragging");
    document.body.classList.add("col-resizing");
    e.preventDefault();
  });
  document.addEventListener("mousemove", (e) => { if (dragging) moveTo(e.clientX); });
  document.addEventListener("mouseup", stop);

  splitter.addEventListener("touchstart", () => {
    dragging = true;
    splitter.classList.add("dragging");
  }, { passive: true });
  document.addEventListener("touchmove", (e) => {
    if (dragging && e.touches[0]) moveTo(e.touches[0].clientX);
  }, { passive: true });
  document.addEventListener("touchend", stop);
}

/* ---- feature: PDF zoom in / out --------------------------------------- */
function updateZoomLabel() {
  if (PdfView.ready()) el("zoomLevel").textContent = Math.round(PdfView.getZoom() * 100) + "%";
}

// Coalesce rapid zoom requests (button mashing / ctrl+wheel) into one re-render.
let pendingZoom = null;
let zoomTimer = null;
function queueZoom(target) {
  if (!PdfView.ready()) return;
  pendingZoom = Math.max(0.5, Math.min(4, target));
  el("zoomLevel").textContent = Math.round(pendingZoom * 100) + "%";   // instant feedback
  clearTimeout(zoomTimer);
  zoomTimer = setTimeout(async () => {
    const z = pendingZoom;
    pendingZoom = null;
    await PdfView.setZoom(z);
    updateZoomLabel();
  }, 130);
}
function stepZoom(delta) {
  const base = pendingZoom != null ? pendingZoom : (PdfView.ready() ? PdfView.getZoom() : 1);
  queueZoom(base + delta);
}

function initZoom() {
  el("zoomIn").addEventListener("click", () => stepZoom(0.2));
  el("zoomOut").addEventListener("click", () => stepZoom(-0.2));
  el("zoomReset").addEventListener("click", () => queueZoom(1));
  // Ctrl/Cmd + wheel over the PDF zooms it, like a normal PDF viewer.
  el("pdfPane").addEventListener("wheel", (e) => {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    stepZoom(e.deltaY < 0 ? 0.1 : -0.1);
  }, { passive: false });
}

/* ---- feature: find in PDF (search bar in the left pane) ---------------- */
function setFindCount(res) {
  const c = el("findCount");
  const has = res && res.count > 0;
  const q = el("pdfSearch").value.trim();
  if (!q) { c.textContent = ""; c.classList.remove("none"); }
  else if (has) { c.textContent = `${res.index} / ${res.count}`; c.classList.remove("none"); }
  else { c.textContent = "0 / 0"; c.classList.add("none"); }
  const disabled = !has;
  el("findPrev").disabled = disabled;
  el("findNext").disabled = disabled;
}

function clearFind(refocus) {
  el("pdfSearch").value = "";
  if (PdfView.ready()) PdfView.clearSearch();
  el("findCount").textContent = "";
  el("findCount").classList.remove("none");
  el("findPrev").disabled = true;
  el("findNext").disabled = true;
  if (refocus) el("pdfSearch").focus();
}

function runFind() {
  if (!PdfView.ready()) { banner("PDF viewer isn't available."); return; }
  const q = el("pdfSearch").value.trim();
  if (q.length < 2) {   // need 2+ chars to search; blank the counter, don't say "0/0"
    PdfView.clearSearch();
    el("findCount").textContent = "";
    el("findCount").classList.remove("none");
    el("findPrev").disabled = true;
    el("findNext").disabled = true;
    return;
  }
  setFindCount(PdfView.search(q));
}

function initPdfSearch() {
  const box = el("pdfSearch");
  if (!box) return;
  let timer = null;
  box.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(runFind, 180);   // debounce while typing
  });
  box.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      if (!PdfView.ready()) return;
      // if the query changed since last search, runFind covers it; otherwise step
      clearTimeout(timer);
      if (el("findNext").disabled) { runFind(); return; }
      setFindCount(e.shiftKey ? PdfView.searchPrev() : PdfView.searchNext());
    } else if (e.key === "Escape") {
      clearFind(false);
    }
  });
  el("findNext").addEventListener("click", () => setFindCount(PdfView.searchNext()));
  el("findPrev").addEventListener("click", () => setFindCount(PdfView.searchPrev()));
  el("findClear").addEventListener("click", () => clearFind(true));
}

/* ---- feature: add a custom category (wired to the model) --------------- */
function openCatModal() {
  el("catErr").textContent = "";
  el("catLabel").value = "";
  el("catDesc").value = "";
  el("catAns").value = "";
  el("catModal").classList.add("show");
  el("catLabel").focus();
}
function closeCatModal() { el("catModal").classList.remove("show"); }

async function submitCategory() {
  const label = el("catLabel").value.trim();
  const description = el("catDesc").value.trim();
  const answer_format = el("catAns").value.trim();
  if (!label) { el("catErr").textContent = "Label is required."; el("catLabel").focus(); return; }
  if (!description) {
    el("catErr").textContent = "Description is required — it is used as the model prompt.";
    el("catDesc").focus();
    return;
  }
  el("catSave").disabled = true;
  try {
    const r = await fetch("/api/categories", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label, description, answer_format }),
    });
    const j = await r.json();
    if (!r.ok) { el("catErr").textContent = j.error || "Could not add category."; return; }
    await refreshCategories(j.label);
    closeCatModal();
    banner(`Added category “${j.label}”. Click “Re-run model” to extract and highlight it on this document.`,
      "warn");
  } catch (e) {
    el("catErr").textContent = "Could not add category: " + e.message;
  } finally {
    el("catSave").disabled = false;
  }
}

/* Re-pull the category list after an add/remove and rebuild colours + chips.
   If a subset of labels is stored as visible, make sure `ensureLabel` shows. */
async function refreshCategories(ensureLabel) {
  const cats = await fetch("/api/categories").then((r) => r.json());
  state.categories = cats;
  state.colorByLabel = {};
  cats.forEach((c) => (state.colorByLabel[c.label] = c.color));

  if (ensureLabel) {
    const raw = localStorage.getItem(VISIBLE_KEY);
    if (raw) {   // user has a custom visible subset -> add the new label to it
      try {
        const arr = JSON.parse(raw);
        if (!arr.includes(ensureLabel)) {
          arr.push(ensureLabel);
          localStorage.setItem(VISIBLE_KEY, JSON.stringify(arr));
        }
      } catch { /* ignore malformed storage */ }
    }
    state.selected.add(ensureLabel);
  }
  renderChips();
  if (state.hasResults) applyAll();
}

function initAddCategory() {
  el("addCategory").addEventListener("click", openCatModal);
  el("catCancel").addEventListener("click", closeCatModal);
  el("catSave").addEventListener("click", submitCategory);
  // close on backdrop click or Escape
  el("catModal").addEventListener("click", (e) => {
    if (e.target === el("catModal")) closeCatModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && el("catModal").classList.contains("show")) closeCatModal();
  });
}

/* ---- session persistence (survive page reload) ------------------------- */
function persistSession() {
  if (!state.docId || !state.hasResults) return;
  try {
    localStorage.setItem(SESSION_KEY, JSON.stringify({
      docId: state.docId,
      filename: state.filename,
      result: state.lastResult,
    }));
  } catch { /* localStorage full/blocked -> skip */ }
}

async function restoreSession() {
  let saved;
  try { saved = JSON.parse(localStorage.getItem(SESSION_KEY) || "null"); }
  catch { saved = null; }
  if (!saved || !saved.docId) return;

  try {
    // doc is gone if the server restarted -> clear and bail
    const rv = await fetch(`/api/review/${saved.docId}`);
    if (!rv.ok) { localStorage.removeItem(SESSION_KEY); return; }
    const data = await rv.json();

    state.docId = saved.docId;
    state.filename = saved.filename;
    state.lastResult = saved.result || {};
    state.spansByLabel = (saved.result && saved.result.spans_by_label) || {};
    setItems(data.items || []);

    el("panes").style.display = "grid";
    el("emptyState").style.display = "none";
    el("rerun").classList.remove("hidden");
    banner("Restored your last analysed document.", "warn");

    if (PdfView.ready()) {
      await PdfView.load("/api/pdf/" + state.docId);
      el("pdfNote").textContent = state.filename || "";
      updateZoomLabel();
    }
    onResultsReady(state.lastResult);
  } catch {
    localStorage.removeItem(SESSION_KEY);
  }
}

init();
