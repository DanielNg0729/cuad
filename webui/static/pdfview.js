/* PDF viewer + highlighter for the left pane.
   Renders the actual PDF with PDF.js (canvas + a transparent text layer), then
   highlights extracted spans by *searching* the text layer and tinting the
   matching text divs. We rely on PDF.js's own positioning for the text layer,
   so there is no manual coordinate math. Highlighting is at text-item
   granularity (whole word/line items that overlap a match get tinted), which is
   robust for the native-text CUAD PDFs. */

const PdfView = (() => {
  const lib = window.pdfjsLib;
  if (lib) {
    // Worker is served locally too (see static/vendor/), matching pdf.min.js.
    lib.GlobalWorkerOptions.workerSrc = "/static/vendor/pdf.worker.min.js";
  }

  let pages = [];          // [{pageStr, ranges:[[start,end,itemIdx]], textDivs:[el]}]
  let container = null;
  let marked = [];         // currently tinted divs, for fast clearing
  let lastUrl = null;      // for refit() after a splitter resize
  let lastEntries = [];    // last highlight() input, re-applied after refit
  let pdfDoc = null;       // the loaded PDF.js document (re-rendered on zoom/refit)
  let zoom = 1;            // zoom multiplier on top of the fit-to-width scale
  const MIN_ZOOM = 0.5, MAX_ZOOM = 4;
  const clampZoom = (z) => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z));

  // --- in-document search state (the "find in PDF" bar) ---------------------
  let searchQuery = "";    // current find query ("" = no active search)
  let searchMatches = [];  // [{divs:[el,...]}] one per occurrence, in doc order
  let searchIdx = -1;      // index of the currently focused match (-1 = none)
  let searchMarked = [];   // divs tinted by search, for fast clearing

  const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  function hexToRgba(hex, a) {
    const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex || "");
    if (!m) return `rgba(255,221,87,${a})`;
    return `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${a})`;
  }

  async function load(url, onProgress) {
    if (!lib) throw new Error("PDF.js failed to load (static/vendor/pdf.min.js).");
    if (url !== lastUrl) zoom = 1;   // a freshly opened document starts fit-to-width
    lastUrl = url;
    container = document.getElementById("pdfPane");
    pdfDoc = await lib.getDocument({ url }).promise;
    await render(onProgress);
    return pdfDoc.numPages;
  }

  /* (Re)render every page from the loaded document at the current zoom. Called
     on initial load, on zoom changes, and after a splitter resize. Re-rendering
     (rather than CSS-scaling) keeps text and highlights crisp at any zoom. */
  async function render(onProgress) {
    if (!pdfDoc || !container) return;
    pages = [];
    marked = [];
    // remember roughly where we were so zooming doesn't jump to the top
    const frac = container.scrollHeight ? container.scrollTop / container.scrollHeight : 0;
    container.innerHTML = "";

    const fitWidth = Math.max(320, container.clientWidth - 24);
    const dpr = window.devicePixelRatio || 1;

    for (let p = 1; p <= pdfDoc.numPages; p++) {
      const page = await pdfDoc.getPage(p);
      const base = page.getViewport({ scale: 1 });
      const fitScale = Math.min(3, Math.max(0.5, fitWidth / base.width));
      const scale = Math.min(8, fitScale * zoom);   // zoom on top of fit
      const vp = page.getViewport({ scale });

      const wrap = document.createElement("div");
      wrap.className = "pdf-page";
      wrap.style.width = Math.floor(vp.width) + "px";
      wrap.style.height = Math.floor(vp.height) + "px";

      const canvas = document.createElement("canvas");
      canvas.width = Math.floor(vp.width * dpr);
      canvas.height = Math.floor(vp.height * dpr);
      canvas.style.width = Math.floor(vp.width) + "px";
      canvas.style.height = Math.floor(vp.height) + "px";
      wrap.appendChild(canvas);

      const textLayer = document.createElement("div");
      textLayer.className = "textLayer";
      textLayer.style.width = Math.floor(vp.width) + "px";
      textLayer.style.height = Math.floor(vp.height) + "px";
      // PDF.js 3.x positions text-layer spans relative to this scale factor.
      textLayer.style.setProperty("--scale-factor", scale);
      wrap.appendChild(textLayer);

      container.appendChild(wrap);

      await page.render({
        canvasContext: canvas.getContext("2d"),
        viewport: vp,
        transform: dpr !== 1 ? [dpr, 0, 0, dpr, 0, 0] : null,
      }).promise;

      // Text layer drives highlighting. If it ever fails, the canvas above
      // still shows the PDF; highlighting just no-ops for that page.
      let s = "";
      const ranges = [];
      const textDivs = [];
      try {
        const tc = await page.getTextContent();
        // textContentSource replaces the deprecated textContent param (PDF.js 3.x).
        await lib.renderTextLayer({
          textContentSource: tc,
          container: textLayer,
          viewport: vp,
          textDivs,
        }).promise;
        tc.items.forEach((it, i) => {
          const start = s.length;
          s += it.str;
          ranges.push([start, s.length, i]);
        });
      } catch (e) {
        console.warn("text layer failed on page " + p, e);
      }
      pages.push({ pageStr: s, ranges, textDivs });

      if (onProgress) onProgress(p, pdfDoc.numPages);
    }
    // restore the relative scroll position after the new pages are laid out
    container.scrollTop = frac * container.scrollHeight;
  }

  function clearHighlights() {
    marked.forEach((d) => {
      d.style.backgroundColor = "";
      d.classList.remove("hl");
    });
    marked = [];
  }

  function mark(div, color) {
    if (!div) return;
    div.style.backgroundColor = hexToRgba(color, 0.45);
    div.classList.add("hl");
    marked.push(div);
  }

  /* All text-layer divs on `page` that overlap any match of `query`. Returns a
     de-duplicated array (whitespace-tolerant; matches across word/line splits). */
  function matchDivs(page, query) {
    const norm = (query || "").trim().replace(/\s+/g, " ");
    if (norm.length < 2) return [];
    let re;
    try {
      re = new RegExp(norm.split(" ").map(escapeRegex).join("\\s*"), "gi");
    } catch {
      return [];
    }
    const hits = new Set();
    let m, guard = 0;
    while ((m = re.exec(page.pageStr)) && guard++ < 5000) {
      const a = m.index, b = m.index + m[0].length;
      if (b === a) { re.lastIndex++; continue; }
      for (const [rs, re2, idx] of page.ranges) {
        if (rs < b && re2 > a) hits.add(page.textDivs[idx]);
      }
      if (re.lastIndex === m.index) re.lastIndex++;
    }
    return [...hits];
  }

  /* entries: [{color, spans:[str,...]}] -- spans for the currently selected labels */
  function highlight(entries) {
    lastEntries = entries;
    clearHighlights();
    if (!pages.length) return;
    for (const entry of entries) {
      for (const span of entry.spans) {
        for (const page of pages) {
          for (const div of matchDivs(page, span)) mark(div, entry.color);
        }
      }
    }
  }

  /* 1-based page number where `query` first appears on the PDF, or null. */
  function findPage(query) {
    for (let i = 0; i < pages.length; i++) {
      if (matchDivs(pages[i], query).length) return i + 1;
    }
    return null;
  }

  /* Scroll the PDF pane to the first occurrence of `query` and flash it.
     Returns true if it was located, false otherwise. */
  function scrollTo(query) {
    if (!container || !pages.length) return false;
    for (const page of pages) {
      const divs = matchDivs(page, query);
      if (!divs.length) continue;
      const target = divs[0];
      const cRect = container.getBoundingClientRect();
      const dRect = target.getBoundingClientRect();
      // centre the match within the (internally scrolling) PDF pane
      container.scrollTop +=
        (dRect.top - cRect.top) - container.clientHeight / 2 + dRect.height / 2;
      divs.forEach((d) => {
        d.classList.remove("flash-hl");
        void d.offsetWidth;            // restart the CSS animation
        d.classList.add("flash-hl");
        setTimeout(() => d.classList.remove("flash-hl"), 1500);
      });
      return true;
    }
    return false;
  }

  /* --- in-document search ("find in PDF") --------------------------------
     Like a normal PDF viewer's find bar: tint every occurrence of the query,
     focus one at a time, and step through them with next/prev (wrapping). */

  /* Every occurrence of `query`, in document order, as {divs:[el,...]} where
     divs are the text-layer spans that occurrence overlaps (a match can span
     more than one span). Whitespace-tolerant, case-insensitive. */
  function findMatches(query) {
    const norm = (query || "").trim().replace(/\s+/g, " ");
    const out = [];
    if (norm.length < 2) return out;
    let re;
    try {
      re = new RegExp(norm.split(" ").map(escapeRegex).join("\\s*"), "gi");
    } catch {
      return out;
    }
    for (const page of pages) {
      re.lastIndex = 0;
      let m, guard = 0;
      while ((m = re.exec(page.pageStr)) && guard++ < 5000) {
        const a = m.index, b = m.index + m[0].length;
        if (b === a) { re.lastIndex++; continue; }
        const divs = [];
        for (const [rs, re2, idx] of page.ranges) {
          if (rs < b && re2 > a) divs.push(page.textDivs[idx]);
        }
        if (divs.length) out.push({ divs });
        if (re.lastIndex === m.index) re.lastIndex++;
      }
    }
    return out;
  }

  function clearSearchTints() {
    searchMarked.forEach((d) => d.classList.remove("search-hl", "search-current"));
    searchMarked = [];
  }

  /* Recompute + re-tint matches for the current searchQuery without scrolling.
     Keeps the focused match if its index is still in range. Used on (re)search
     and after a re-render (zoom/resize) rebuilds the text layers. */
  function applySearch() {
    clearSearchTints();
    searchMatches = findMatches(searchQuery);
    if (!searchMatches.length) { searchIdx = -1; return; }
    if (searchIdx < 0 || searchIdx >= searchMatches.length) searchIdx = 0;
    searchMatches.forEach((mt, i) => {
      mt.divs.forEach((d) => {
        d.classList.add("search-hl");
        if (i === searchIdx) d.classList.add("search-current");
        searchMarked.push(d);
      });
    });
  }

  function scrollToMatch(i) {
    const mt = searchMatches[i];
    if (!mt || !container) return;
    const target = mt.divs[0];
    const cRect = container.getBoundingClientRect();
    const dRect = target.getBoundingClientRect();
    container.scrollTop +=
      (dRect.top - cRect.top) - container.clientHeight / 2 + dRect.height / 2;
  }

  /* Focus match `i` (wraps around): move the "current" emphasis and scroll to
     it. Returns {count, index} where index is 1-based (0 when there are none). */
  function focusMatch(i) {
    if (!searchMatches.length) return { count: 0, index: 0 };
    const n = searchMatches.length;
    i = ((i % n) + n) % n;
    if (searchIdx >= 0 && searchMatches[searchIdx])
      searchMatches[searchIdx].divs.forEach((d) => d.classList.remove("search-current"));
    searchIdx = i;
    searchMatches[i].divs.forEach((d) => d.classList.add("search-current"));
    scrollToMatch(i);
    return { count: n, index: i + 1 };
  }

  /* Start/refresh a find. Tints all matches and jumps to the first.
     Returns {count, index} for the find bar's "n / N" counter. */
  function search(query) {
    searchQuery = (query || "").trim();
    searchIdx = -1;
    applySearch();
    if (!searchMatches.length) return { count: 0, index: 0 };
    return focusMatch(0);
  }

  function searchNext() { return focusMatch(searchIdx + 1); }
  function searchPrev() { return focusMatch(searchIdx - 1); }

  function clearSearch() {
    searchQuery = "";
    searchIdx = -1;
    searchMatches = [];
    clearSearchTints();
  }

  /* Serialise re-renders (zoom clicks / wheel / resize) so overlapping calls
     can't interleave mid-render and corrupt the page list. */
  let renderChain = Promise.resolve();
  function queueRender() {
    renderChain = renderChain.then(async () => {
      const entries = lastEntries;
      await render();
      highlight(entries);
      if (searchQuery) applySearch();   // re-tint find matches on the new layers
    }).catch((e) => console.warn("pdf re-render failed", e));
    return renderChain;
  }

  /* Re-render at the pane's current width (after a splitter resize). */
  async function refit() {
    if (!lib || !pdfDoc) return;
    await queueRender();
  }

  /* Zoom the PDF (re-render at the new scale; the pane size is unchanged). */
  async function setZoom(z) {
    zoom = clampZoom(z);
    if (lib && pdfDoc) await queueRender();
    return zoom;
  }
  function getZoom() { return zoom; }

  function ready() { return !!lib; }

  return { load, highlight, clearHighlights, findPage, scrollTo, refit,
           setZoom, getZoom, ready,
           search, searchNext, searchPrev, clearSearch };
})();
