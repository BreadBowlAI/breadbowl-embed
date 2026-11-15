(function () {
  "use strict";
  var data = JSON.parse(document.getElementById("report-data").textContent);
  var state = {
    view: "attribution-final",
    head: "mean",
    onlySlot: false,
    slot: { query: data.defaults.query_slot, document: data.defaults.document_slot },
    matrix: "contribution",
    active: [data.defaults.query_slot, data.defaults.document_slot],
    cursor: { query: -1, document: -1 }
  };
  var headCache = {};

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  function rgb(hex) {
    var h = hex.replace("#", "");
    if (h.length === 3) { h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2]; }
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  function mix(a, b, t) {
    return [0, 1, 2].map(function (i) { return Math.round(a[i] + (b[i] - a[i]) * t); });
  }
  function luminance(c) {
    var v = c.map(function (x) {
      x /= 255;
      return x <= 0.03928 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
  }
  function shade(value, scale, diverging, base) {
    var t = scale > 0 && isFinite(value) ? Math.min(1, Math.abs(value) / scale) : 0;
    var target = rgb(cssVar(diverging && value < 0 ? "--neg" : "--pos"));
    var c = mix(base, target, t * 0.9);
    return { bg: "rgb(" + c.join(",") + ")", fg: luminance(c) < 0.3 ? "#ffffff" : "#0b0b0b", t: t };
  }
  function fmt(x) {
    if (x === null || !isFinite(x)) { return "–"; }
    var a = Math.abs(x);
    if (a !== 0 && (a < 1e-3 || a >= 1e4)) { return x.toExponential(2); }
    return x.toFixed(4);
  }
  function el(tag, attrs, text) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    if (text !== undefined) { node.textContent = text; }
    return node;
  }
  function svgEl(tag, attrs) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    return node;
  }

  // ---- attention and attribution ----------------------------------------------------------------
  function headMatrix(side) {
    var key = side + ":" + state.head;
    if (headCache[key]) { return headCache[key]; }
    var att = data[side].attention;
    var m;
    if (state.head === "mean") {
      m = att[0].map(function (row, s) {
        return row.map(function (_, t) {
          var sum = 0;
          for (var h = 0; h < att.length; h++) { sum += att[h][s][t]; }
          return sum / att.length;
        });
      });
    } else {
      m = att[Number(state.head)];
    }
    headCache[key] = m;
    return m;
  }
  function tokenValues(side) {
    var m = headMatrix(side);
    var slot = state.slot[side];
    var headLabel = state.head === "mean" ? "heads averaged" : "head " + state.head;
    if (state.view === "attention") {
      return { values: m[slot], diverging: false,
        label: side + " slot " + slot + " attention (" + headLabel + ")" };
    }
    var component = state.view.split("-")[1];
    var contribution = data.contributions[side][component];
    // Rows sum to 1 only up to rounding (bfloat16, 5 stored digits): renormalise as the library does.
    var sums = m.map(function (row) { return row.reduce(function (a, b) { return a + b; }, 0) || 1; });
    m = m.map(function (row, s) { return row.map(function (v) { return v / sums[s]; }); });
    var values;
    if (state.onlySlot) {
      values = m[slot].map(function (a) { return a * contribution[slot]; });
    } else {
      values = m[0].map(function (_, t) {
        var sum = 0;
        for (var s = 0; s < m.length; s++) { sum += contribution[s] * m[s][t]; }
        return sum;
      });
    }
    return { values: values, diverging: true,
      label: component + " attribution" + (state.onlySlot ? " through slot " + slot : "") + " (" + headLabel + ")" };
  }
  function describeToken(side, t, value, label) {
    var tok = data[side].tokens[t];
    return side + " token #" + t + " " + JSON.stringify(tok[0]) + " [" + tok[1] + "] " + label + ": " + fmt(value);
  }
  function renderTokens(side) {
    var box = document.getElementById(side + "-tokens");
    var result = tokenValues(side);
    var scale = 0;
    result.values.forEach(function (v) { if (isFinite(v)) { scale = Math.max(scale, Math.abs(v)); } });
    var base = rgb(cssVar("--surface"));
    box.textContent = "";
    data[side].tokens.forEach(function (tok, t) {
      var span = el("span", { "class": "tok seg-" + tok[1], "data-index": String(t) }, tok[0]);
      var v = result.values[t];
      var c = shade(v, scale, result.diverging, base);
      if (c.t > 0.01) {
        span.style.background = c.bg;
        span.style.color = c.fg;
      }
      span.title = "#" + t + " " + tok[2] + " [" + tok[1] + "] " + fmt(v);
      if (t === state.cursor[side]) { span.classList.add("cursor"); }
      box.appendChild(span);
    });
    if (data[side].truncated) {
      box.appendChild(el("span", { "class": "seg-special" }, " …[truncated at the token limit]"));
    }
    var legend = document.getElementById(side + "-legend");
    legend.textContent = "";
    var bar = el("span", { "class": "bar" });
    var neg = cssVar("--neg"), pos = cssVar("--pos"), surface = cssVar("--surface");
    bar.style.background = result.diverging ?
      "linear-gradient(90deg," + neg + "," + surface + "," + pos + ")" :
      "linear-gradient(90deg," + surface + "," + pos + ")";
    legend.appendChild(el("span", {}, result.diverging ? fmt(-scale) : "0"));
    legend.appendChild(bar);
    legend.appendChild(el("span", {}, fmt(scale)));
    legend.appendChild(el("span", {}, "· " + result.label));
    box.dataset.label = result.label;
    box._values = result.values;
  }
  function renderPickers() {
    ["query", "document"].forEach(function (side) {
      var picker = document.getElementById(side + "-slots");
      Array.prototype.forEach.call(picker.querySelectorAll("button"), function (b) {
        var on = Number(b.dataset.slot) === state.slot[side];
        b.setAttribute("aria-checked", on ? "true" : "false");
        b.tabIndex = on ? 0 : -1;
      });
    });
  }
  function buildPicker(side) {
    var picker = document.getElementById(side + "-slots");
    for (var s = 0; s < data.slots[side]; s++) {
      var b = el("button", { type: "button", role: "radio", "data-slot": String(s),
        "aria-label": side + " slot " + s }, String(s));
      picker.appendChild(b);
    }
    picker.addEventListener("click", function (e) {
      var b = e.target.closest("button");
      if (!b) { return; }
      selectSlot(side, Number(b.dataset.slot), true);
    });
    picker.addEventListener("keydown", function (e) {
      var n = data.slots[side], s = state.slot[side], next = null;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") { next = (s + 1) % n; }
      if (e.key === "ArrowLeft" || e.key === "ArrowUp") { next = (s - 1 + n) % n; }
      if (e.key === "Home") { next = 0; }
      if (e.key === "End") { next = n - 1; }
      if (next !== null) { e.preventDefault(); selectSlot(side, next, true); }
    });
  }
  function selectSlot(side, s, focus) {
    state.slot[side] = s;
    state.active = [state.slot.query, state.slot.document];
    renderPickers();
    renderTokens(side);
    renderMatrix();
    if (focus) {
      var b = document.querySelector("#" + side + "-slots button[data-slot='" + s + "']");
      if (b) { b.focus(); }
    }
  }
  function attachTokenKeys(side) {
    var box = document.getElementById(side + "-tokens");
    var readout = document.getElementById(side + "-readout");
    function show(t) {
      readout.textContent = describeToken(side, t, box._values[t], box.dataset.label);
    }
    box.addEventListener("mouseover", function (e) {
      var span = e.target.closest(".tok");
      if (span) { show(Number(span.dataset.index)); }
    });
    box.addEventListener("keydown", function (e) {
      var n = data[side].tokens.length, t = state.cursor[side];
      if (e.key === "ArrowRight") { t = Math.min(n - 1, t + 1); }
      else if (e.key === "ArrowLeft") { t = Math.max(0, t - 1); }
      else if (e.key === "Home") { t = 0; }
      else if (e.key === "End") { t = n - 1; }
      else { return; }
      e.preventDefault();
      state.cursor[side] = t;
      renderTokens(side);
      var span = box.querySelector(".tok[data-index='" + t + "']");
      if (span && span.scrollIntoView) { span.scrollIntoView({ block: "nearest" }); }
      show(t);
    });
    box.addEventListener("focus", function () {
      if (state.cursor[side] < 0) { state.cursor[side] = 0; renderTokens(side); }
      show(state.cursor[side]);
    });
  }

  // ---- slot-pair matrix -------------------------------------------------------------------------
  var CELL = 22, LEFT = 34, TOP = 30;
  function renderMatrix() {
    var spec = data.matrices[state.matrix];
    var m = spec.data;
    var rows = m.length, cols = m[0].length;
    var svg = document.getElementById("matrix");
    svg.textContent = "";
    var w = LEFT + cols * CELL + 4, h = TOP + rows * CELL + 4;
    svg.setAttribute("viewBox", "0 0 " + w + " " + h);
    svg.setAttribute("width", String(w));
    svg.setAttribute("height", String(h));
    var scale = 0;
    m.forEach(function (row) { row.forEach(function (v) { if (isFinite(v)) { scale = Math.max(scale, Math.abs(v)); } }); });
    var base = rgb(cssVar(spec.diverging ? "--mid" : "--surface"));
    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) {
        var c = shade(m[i][j], scale, spec.diverging, base);
        var rect = svgEl("rect", { x: LEFT + j * CELL, y: TOP + i * CELL, width: CELL, height: CELL,
          fill: c.bg, "class": "cell", "data-i": i, "data-j": j });
        if (i === state.slot.query && j === state.slot.document) { rect.classList.add("sel-row"); }
        if (i === state.active[0] && j === state.active[1]) { rect.classList.add("active"); }
        svg.appendChild(rect);
      }
    }
    // Draw the active cell last so its outline is on top.
    var active = svg.querySelector("rect.active");
    if (active) { svg.appendChild(active); }
    for (var r = 0; r < rows; r++) {
      var tr = svgEl("text", { x: LEFT - 6, y: TOP + r * CELL + CELL * 0.68, "text-anchor": "end" });
      tr.textContent = String(r);
      svg.appendChild(tr);
    }
    for (var q = 0; q < cols; q++) {
      var tc = svgEl("text", { x: LEFT + q * CELL + CELL / 2, y: TOP - 8, "text-anchor": "middle" });
      tc.textContent = String(q);
      svg.appendChild(tc);
    }
    var axis = svgEl("text", { x: LEFT, y: 10 });
    axis.textContent = "document slot →";
    svg.appendChild(axis);
    var legend = document.getElementById("matrix-legend");
    legend.textContent = "";
    var bar = el("span", { "class": "bar" });
    bar.style.background = spec.diverging ?
      "linear-gradient(90deg," + cssVar("--neg") + "," + cssVar("--mid") + "," + cssVar("--pos") + ")" :
      "linear-gradient(90deg," + cssVar("--surface") + "," + cssVar("--pos") + ")";
    legend.appendChild(el("span", {}, spec.diverging ? fmt(-scale) : "0"));
    legend.appendChild(bar);
    legend.appendChild(el("span", {}, fmt(scale)));
    legend.appendChild(el("span", {}, "· " + spec.label + " (rows: query slots)"));
    document.getElementById("matrix-note").textContent = spec.note;
    describeCell();
  }
  function describeCell() {
    var spec = data.matrices[state.matrix];
    var i = state.active[0], j = state.active[1];
    document.getElementById("matrix-readout").textContent =
      "query slot " + i + " × document slot " + j + ": " + spec.label + " " + fmt(spec.data[i][j]) +
      " (selected pair: " + state.slot.query + " × " + state.slot.document + ")";
  }
  function attachMatrix() {
    var svg = document.getElementById("matrix");
    svg.addEventListener("mouseover", function (e) {
      var r = e.target.closest("rect.cell");
      if (!r) { return; }
      state.active = [Number(r.dataset.i), Number(r.dataset.j)];
      describeCell();
    });
    svg.addEventListener("click", function (e) {
      var r = e.target.closest("rect.cell");
      if (!r) { return; }
      choosePair(Number(r.dataset.i), Number(r.dataset.j));
    });
    document.getElementById("matrix-focus").addEventListener("keydown", function (e) {
      var rows = data.slots.query, cols = data.slots.document;
      var i = state.active[0], j = state.active[1];
      if (e.key === "ArrowRight") { j = Math.min(cols - 1, j + 1); }
      else if (e.key === "ArrowLeft") { j = Math.max(0, j - 1); }
      else if (e.key === "ArrowDown") { i = Math.min(rows - 1, i + 1); }
      else if (e.key === "ArrowUp") { i = Math.max(0, i - 1); }
      else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choosePair(i, j); return; }
      else { return; }
      e.preventDefault();
      state.active = [i, j];
      renderMatrix();
    });
  }
  function choosePair(i, j) {
    state.slot.query = i;
    state.slot.document = j;
    state.active = [i, j];
    renderPickers();
    renderTokens("query");
    renderTokens("document");
    renderMatrix();
  }

  // ---- controls and theme -----------------------------------------------------------------------
  function renderAll() {
    renderPickers();
    renderTokens("query");
    renderTokens("document");
    renderMatrix();
  }
  function effectiveDark() {
    var set = document.documentElement.getAttribute("data-theme");
    if (set) { return set === "dark"; }
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function syncThemeButton() {
    var b = document.getElementById("theme");
    var dark = effectiveDark();
    b.setAttribute("aria-pressed", dark ? "true" : "false");
    b.textContent = dark ? "Light theme" : "Dark theme";
  }
  function init() {
    document.getElementById("controls-js").hidden = false;
    document.getElementById("interactive").hidden = false;
    var head = document.getElementById("head");
    for (var h = 0; h < data.heads; h++) {
      head.appendChild(el("option", { value: String(h) }, "head " + h));
    }
    buildPicker("query");
    buildPicker("document");
    attachTokenKeys("query");
    attachTokenKeys("document");
    attachMatrix();
    document.getElementById("view").addEventListener("change", function (e) { state.view = e.target.value; renderAll(); });
    head.addEventListener("change", function (e) { state.head = e.target.value; renderAll(); });
    document.getElementById("only-slot").addEventListener("change", function (e) {
      state.onlySlot = e.target.checked; renderAll();
    });
    document.getElementById("matrix-kind").addEventListener("change", function (e) {
      state.matrix = e.target.value; renderMatrix();
    });
    document.getElementById("theme").addEventListener("click", function () {
      document.documentElement.setAttribute("data-theme", effectiveDark() ? "light" : "dark");
      syncThemeButton();
      renderAll();
    });
    if (window.matchMedia) {
      var mq = window.matchMedia("(prefers-color-scheme: dark)");
      var onChange = function () { syncThemeButton(); renderAll(); };
      if (mq.addEventListener) { mq.addEventListener("change", onChange); }
    }
    syncThemeButton();
    renderAll();
  }
  init();
})();
