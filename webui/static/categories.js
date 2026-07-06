/* CUAD highlighter -- categories reference page.
   Lists all 41 categories (colour, description, answer format, group) and lets
   you tick which ones appear as labels on the main page. The choice is stored
   in localStorage under the same key app.js reads. Default = show all. */

const VISIBLE_KEY = "cuad_visible_categories";

const el = (id) => document.getElementById(id);
const escapeHtml = (s) =>
  String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

let categories = [];

/* visible set: null in storage means "all". Return a Set of labels to show. */
function loadVisible() {
  const raw = localStorage.getItem(VISIBLE_KEY);
  if (!raw) return new Set(categories.map((c) => c.label)); // default all
  try {
    return new Set(JSON.parse(raw));
  } catch {
    return new Set(categories.map((c) => c.label));
  }
}

function saveVisible(set) {
  // if everything is selected, clear the key so "default = all" stays in effect
  if (set.size === categories.length) {
    localStorage.removeItem(VISIBLE_KEY);
  } else {
    localStorage.setItem(VISIBLE_KEY, JSON.stringify([...set]));
  }
}

async function init() {
  categories = await fetch("/api/categories").then((r) => r.json());
  const visible = loadVisible();
  render(visible);

  el("search").addEventListener("input", () => filter(el("search").value));
  el("checkAll").addEventListener("change", () => {
    const v = loadVisible();
    if (el("checkAll").checked) categories.forEach((c) => v.add(c.label));
    else v.clear();
    saveVisible(v);
    render(loadVisible());
  });
  el("reset").addEventListener("click", () => {
    localStorage.removeItem(VISIBLE_KEY);
    el("search").value = "";
    render(loadVisible());
  });

  el("addCat").addEventListener("click", () =>
    toggleAddPanel(el("addCatPanel").style.display === "none"));
  el("ncCancel").addEventListener("click", () => toggleAddPanel(false));
  el("ncSave").addEventListener("click", saveCategory);
  el("addCatPanel").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); saveCategory(); }
    else if (e.key === "Escape") toggleAddPanel(false);
  });
}

function render(visible) {
  const body = el("catBody");
  body.innerHTML = "";
  categories.forEach((c, i) => {
    const tr = document.createElement("tr");
    if (c.custom) tr.classList.add("custom-row");
    tr.dataset.label = c.label.toLowerCase();
    tr.dataset.desc = (c.description || "").toLowerCase();
    const badge = c.custom ? ' <span class="custom-tag">custom</span>' : "";
    tr.innerHTML =
      `<td><span class="swatch" style="background:${c.color}"></span></td>` +
      `<td class="lbl">${escapeHtml(c.label)}${badge}</td>` +
      `<td>${escapeHtml(c.description)}</td>` +
      `<td class="fmt">${escapeHtml(c.answer_format || "—")}</td>` +
      `<td class="grp">${escapeHtml(c.group || "—")}</td>` +
      `<td class="show-col"></td>`;
    const showCell = tr.lastElementChild;
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = visible.has(c.label);
    cb.addEventListener("change", () => {
      const v = loadVisible();
      if (cb.checked) v.add(c.label);
      else v.delete(c.label);
      saveVisible(v);
      syncCheckAll();
    });
    showCell.appendChild(cb);
    if (c.custom) {
      const del = document.createElement("button");
      del.className = "cat-del";
      del.title = "Remove this custom category";
      del.innerHTML = "&times;";
      del.addEventListener("click", () => removeCategory(c.label));
      showCell.appendChild(del);
    }
    body.appendChild(tr);
  });
  syncCheckAll();
}

/* ---- add / remove custom categories ------------------------------------ */
async function reloadCategories() {
  categories = await fetch("/api/categories").then((r) => r.json());
  render(loadVisible());
  filter(el("search").value);
}

function toggleAddPanel(show) {
  el("addCatPanel").style.display = show ? "" : "none";
  el("ncErr").textContent = "";
  if (show) {
    el("ncLabel").value = "";
    el("ncDesc").value = "";
    el("ncAns").value = "";
    el("ncLabel").focus();
  }
}

async function saveCategory() {
  const label = el("ncLabel").value.trim();
  const description = el("ncDesc").value.trim();
  const answer_format = el("ncAns").value.trim();
  if (!label) { el("ncErr").textContent = "Label is required."; el("ncLabel").focus(); return; }
  if (!description) {
    el("ncErr").textContent = "Description is required — it is used as the model prompt.";
    el("ncDesc").focus();
    return;
  }
  el("ncSave").disabled = true;
  try {
    const r = await fetch("/api/categories", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label, description, answer_format }),
    });
    const j = await r.json();
    if (!r.ok) { el("ncErr").textContent = j.error || "Could not add category."; return; }
    // if the user stored a custom visible subset, make sure the new one shows;
    // with no stored subset the default is "show all", so nothing to do.
    const raw = localStorage.getItem(VISIBLE_KEY);
    if (raw) {
      try {
        const arr = JSON.parse(raw);
        if (!arr.includes(j.label)) {
          arr.push(j.label);
          localStorage.setItem(VISIBLE_KEY, JSON.stringify(arr));
        }
      } catch { /* ignore malformed storage */ }
    }
    toggleAddPanel(false);
    await reloadCategories();
  } catch (e) {
    el("ncErr").textContent = "Could not add category: " + e.message;
  } finally {
    el("ncSave").disabled = false;
  }
}

async function removeCategory(label) {
  try {
    const r = await fetch("/api/categories/" + encodeURIComponent(label), { method: "DELETE" });
    if (!r.ok) return;
    const v = loadVisible();
    v.delete(label);
    saveVisible(v);
    await reloadCategories();
  } catch { /* ignore */ }
}

function syncCheckAll() {
  const v = loadVisible();
  el("checkAll").checked = v.size === categories.length;
  el("checkAll").indeterminate = v.size > 0 && v.size < categories.length;
}

function filter(q) {
  q = q.trim().toLowerCase();
  document.querySelectorAll("#catBody tr").forEach((tr) => {
    const hit = !q || tr.dataset.label.includes(q) || tr.dataset.desc.includes(q);
    tr.style.display = hit ? "" : "none";
  });
}

init();
