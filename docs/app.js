/* Interactive demo of "Learning-Rate Sensitivity of Backpropagation". No dependencies.
 *
 * Data: window.SWEEP (docs/data/sweep.js, from src/demo_data.py) holds the per-epoch validation
 * accuracy of the main sweep; window.DEMO_MODELS (docs/data/models.js, from src/export_demo.py)
 * holds the Optdigits networks, which run here in the browser.
 * ?record=1 exposes window.__tour.render(t) so that a script can capture the guided tour frame by frame.
 */
(function () {
  "use strict";

  const SW = window.SWEEP;
  const DM = window.DEMO_MODELS || null;
  const LRS = SW.lrs;
  const NLR = LRS.length;
  const METHODS = ["bp_sgd", "prox_sgd", "prox_all"];
  const STYLE = {
    bp_sgd: { name: "BP+SGD", color: "#c8553d", marker: "circle" },
    prox_sgd: { name: "ProxProp", color: "#1a5ca8", marker: "square" },
    prox_all: { name: "ProxProp-all", color: "#0a2a55", marker: "diamond" },
  };
  const DSN = { drybean: "Dry Bean", optdigits: "Optdigits", har: "HAR" };
  const ARCHN = { shallow: "1 hidden layer", medium: "2 hidden layers", deep: "3 hidden layers" };
  const SVGNS = "http://www.w3.org/2000/svg";
  const RECORD_AT = location.hash.match(/^#record(?:@([\d.]+))?$/); // #record, or #record@40 to preview second 40
  const RECORD = new URLSearchParams(location.search).has("record") || !!RECORD_AT;
  const $ = (s) => document.querySelector(s);

  // ------------------------------------------------------------------ helpers
  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴" };
  function lrLabel(v) {
    if (v >= 0.01) return String(+v.toPrecision(3));
    const e = Math.floor(Math.log10(v) + 1e-9);
    const m = Math.round(v / Math.pow(10, e));
    const p = "10" + String(e).split("").map((c) => SUP[c]).join("");
    return m === 1 ? p : m + "·" + p;
  }
  function svgEl(tag, attrs, parent) {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function text(parent, x, y, str, cls, anchor, extra) {
    const t = svgEl("text", Object.assign({ x, y, class: cls, "text-anchor": anchor || "start" }, extra || {}), parent);
    t.textContent = str;
    return t;
  }
  function marker(g, shape, x, y, r, fill, stroke, sw) {
    const a = { fill, stroke, "stroke-width": sw };
    if (shape === "circle") return svgEl("circle", Object.assign({ cx: x, cy: y, r }, a), g);
    if (shape === "square") {
      const h = r * 0.88;
      return svgEl("rect", Object.assign({ x: x - h, y: y - h, width: 2 * h, height: 2 * h }, a), g);
    }
    const d = r * 1.22;
    return svgEl("path", Object.assign({ d: `M${x},${y - d}L${x + d},${y}L${x},${y + d}L${x - d},${y}Z` }, a), g);
  }
  function cross(g, x, y, r, color, sw) {
    svgEl("path", { d: `M${x - r},${y - r}L${x + r},${y + r}M${x - r},${y + r}L${x + r},${y - r}`, stroke: color,
      "stroke-width": sw, "stroke-linecap": "round", fill: "none" }, g);
  }
  function swatch(m) {
    const st = STYLE[m];
    const s = svgEl("svg", { viewBox: "0 0 30 14", "aria-hidden": "true" });
    svgEl("line", { x1: 1, x2: 29, y1: 7, y2: 7, stroke: st.color, "stroke-width": m === "prox_all" ? 3 : 2.4 }, s);
    marker(s, st.marker, 15, 7, 4.4, st.color, "#fff", 1.2);
    return s;
  }
  function pct(permille) { return (permille / 10).toFixed(1); }
  function ticksInto(box) {
    box.textContent = "";
    LRS.forEach((v, i) => {
      const s = document.createElement("span");
      s.textContent = i % 2 === 0 ? lrLabel(v) : "";
      box.appendChild(s);
    });
  }
  function windowText(usable) {
    const idx = usable.map((u, i) => (u ? i : -1)).filter((i) => i >= 0);
    if (!idx.length) return "none";
    const lo = LRS[idx[0]], hi = LRS[idx[idx.length - 1]];
    return lo === hi ? lrLabel(lo) : lrLabel(lo) + " – " + lrLabel(hi);
  }

  // ------------------------------------------------------------------ explorer
  const ex = { ds: "har", arch: "medium", lrPos: 8 };
  const exIndex = () => Math.max(0, Math.min(NLR - 1, Math.round(ex.lrPos)));
  const setting = () => SW.settings[ex.ds + "/" + ex.arch];

  function buildLegend() {
    const box = $("#legend");
    box.textContent = "";
    for (const m of METHODS) {
      const s = document.createElement("span");
      s.appendChild(swatch(m));
      s.appendChild(document.createTextNode(STYLE[m].name));
      box.appendChild(s);
    }
    const notes = [
      ["filled", "filled marker: usable"],
      ["hollow", "hollow: outside the usable range"],
      ["cross", "×: numerical failure in a seed"],
    ];
    for (const [kind, label] of notes) {
      const s = document.createElement("span");
      const v = svgEl("svg", { viewBox: "0 0 14 14", style: "width:14px;height:14px", "aria-hidden": "true" });
      if (kind === "cross") cross(v, 7, 7, 4, "#48505c", 2);
      else marker(v, "circle", 7, 7, 4.6, kind === "filled" ? "#48505c" : "#fff", "#48505c", 1.6);
      s.appendChild(v);
      s.appendChild(document.createTextNode(label));
      box.appendChild(s);
    }
  }

  function frameAxes(g, W, H, m, X, Y, xTicks, xLabel, yLabel) {
    for (const a of [0, 0.25, 0.5, 0.75, 1]) {
      svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(a), y2: Y(a), class: "grid" }, g);
      text(g, m.l - 8, Y(a) + 4, Math.round(a * 100) + "%", "tick", "end");
    }
    for (const [v, lab] of xTicks) text(g, X(v), H - m.b + 18, lab, "tick", "middle");
    svgEl("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, class: "frame" }, g);
    text(g, (m.l + W - m.r) / 2, H - 6, xLabel, "axis", "middle");
    const cy = (m.t + H - m.b) / 2;
    text(g, 15, cy, yLabel, "axis", "middle", { transform: `rotate(-90 15 ${cy})` });
  }

  function drawCurveChart() {
    const svg = $("#curve-chart");
    svg.textContent = "";
    const W = 560, H = 300, m = { l: 56, r: 16, t: 26, b: 44 };
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const X = (i) => m.l + (i / (NLR - 1)) * (W - m.l - m.r);
    const Y = (a) => m.t + (1 - a) * (H - m.t - m.b);
    const s = setting(), ref = s.ref / 1000, i0 = exIndex();
    const g = svgEl("g", {}, svg);
    svgEl("rect", { x: m.l, y: Y(ref), width: W - m.l - m.r, height: Y(ref - SW.delta) - Y(ref), fill: "#dfe3e8" }, g);
    for (let i = 0; i < NLR; i++) svgEl("line", { x1: X(i), x2: X(i), y1: m.t, y2: H - m.b, class: "grid minor" }, g);
    frameAxes(g, W, H, m, X, Y, [[0, lrLabel(1e-4)], [4, lrLabel(1e-2)], [8, "1"], [12, "100"]],
      "Learning rate η", "Final val. accuracy");
    text(g, W - m.r - 8, Y(ref - SW.delta) + 14, "gray band: within 2 points of the best", "band-label", "end");
    for (const meth of METHODS) {
      const r = s.methods[meth], st = STYLE[meth];
      svgEl("polyline", { points: r.final.map((v, i) => `${X(i)},${Y(v / 1000)}`).join(" "), fill: "none",
        stroke: st.color, "stroke-width": meth === "prox_all" ? 2.6 : 2, "stroke-linejoin": "round" }, g);
    }
    for (const meth of METHODS) {
      const r = s.methods[meth], st = STYLE[meth];
      r.final.forEach((v, i) => {
        const x = X(i), y = Y(v / 1000), big = i === i0;
        if (r.runs[i].fails.length) cross(g, x, y, big ? 6.5 : 4.4, st.color, big ? 2.8 : 2.1);
        else marker(g, st.marker, x, y, big ? 7 : 4.3, r.usable[i] ? st.color : "#fff", st.color, big ? 2.4 : 1.7);
      });
    }
    const cx = X(ex.lrPos);
    svgEl("line", { x1: cx, x2: cx, y1: m.t, y2: H - m.b, class: "cursor" }, g);
    text(g, Math.max(m.l + 30, Math.min(W - m.r - 30, cx)), m.t - 9, "η = " + lrLabel(LRS[i0]), "cursor-label", "middle");
  }

  function drawTrainChart() {
    const svg = $("#train-chart");
    svg.textContent = "";
    const W = 560, H = 300, m = { l: 56, r: 16, t: 26, b: 44 }, EP = SW.epochs;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const X = (e) => m.l + ((e - 1) / (EP - 1)) * (W - m.l - m.r);
    const Y = (a) => m.t + (1 - a) * (H - m.t - m.b);
    const s = setting(), ref = s.ref / 1000, i = exIndex();
    const g = svgEl("g", {}, svg);
    svgEl("rect", { x: m.l, y: Y(ref), width: W - m.l - m.r, height: Y(ref - SW.delta) - Y(ref), fill: "#dfe3e8" }, g);
    frameAxes(g, W, H, m, X, Y, [[1, "1"], [25, "25"], [50, "50"], [75, "75"], [100, "100"]], "Epoch", "Val. accuracy");
    text(g, (m.l + W - m.r) / 2, m.t - 9, `${DSN[ex.ds]}, ${ARCHN[ex.arch]}, η = ${lrLabel(LRS[i])}`, "cursor-label", "middle");
    for (const meth of METHODS) {
      const run = s.methods[meth].runs[i], st = STYLE[meth];
      const upper = run.hi.map((v, e) => `${X(e + 1)},${Y(v / 1000)}`);
      const lower = run.lo.map((v, e) => `${X(e + 1)},${Y(v / 1000)}`).reverse();
      svgEl("polygon", { points: upper.concat(lower).join(" "), fill: st.color, opacity: 0.13 }, g);
    }
    for (const meth of METHODS) {
      const run = s.methods[meth].runs[i], st = STYLE[meth];
      svgEl("polyline", { points: run.mean.map((v, e) => `${X(e + 1)},${Y(v / 1000)}`).join(" "), fill: "none",
        stroke: st.color, "stroke-width": meth === "prox_all" ? 2.8 : 2.2, "stroke-linejoin": "round" }, g);
      const seen = new Set();
      for (const f of run.fails) {
        if (seen.has(f.epoch)) continue;
        seen.add(f.epoch);
        const e = Math.min(EP, Math.max(1, f.epoch));
        cross(g, X(e), Y(run.mean[e - 1] / 1000), 5, st.color, 2.4);
      }
    }
  }

  function methodCell(meth) {
    const td = document.createElement("td");
    td.className = "method-cell";
    td.appendChild(swatch(meth));
    td.appendChild(document.createTextNode(STYLE[meth].name));
    return td;
  }

  function drawCards() { // one table row per update rule
    const s = setting(), i = exIndex(), body = $("#method-cards");
    body.textContent = "";
    for (const meth of METHODS) {
      const r = s.methods[meth], nf = r.runs[i].fails.length;
      let status;
      if (r.usable[i]) status = '<span class="status-good">usable</span>';
      else if (nf) status = `<span class="status-bad">numerical failure in ${nf} of ${SW.seeds} seeds</span>`;
      else status = '<span class="status-warn">outside the usable range</span>';
      const tr = document.createElement("tr");
      tr.appendChild(methodCell(meth));
      tr.insertAdjacentHTML("beforeend", `<td>${pct(r.final[i])}%</td><td>${status}</td><td>${windowText(r.usable)}</td>`);
      body.appendChild(tr);
    }
  }

  function renderExplorer() {
    for (const [id, key] of [["#ds-seg", "ds"], ["#arch-seg", "arch"]])
      for (const btn of document.querySelectorAll(id + " button")) btn.classList.toggle("is-selected", btn.dataset.v === ex[key]);
    $("#lr-slider").value = exIndex();
    $("#lr-value").textContent = lrLabel(LRS[exIndex()]);
    drawCurveChart();
    drawTrainChart();
    drawCards();
  }

  let sweepTimer = null;
  function stopSweep() {
    if (sweepTimer) cancelAnimationFrame(sweepTimer);
    sweepTimer = null;
    $("#sweep-btn").innerHTML = "Sweep &eta;";
  }
  function startSweep() {
    const t0 = performance.now(), dur = 7000;
    $("#sweep-btn").innerHTML = "Stop";
    const step = (now) => {
      const p = Math.min(1, (now - t0) / dur);
      ex.lrPos = p * (NLR - 1);
      renderExplorer();
      if (p < 1) sweepTimer = requestAnimationFrame(step);
      else stopSweep();
    };
    sweepTimer = requestAnimationFrame(step);
  }

  function initExplorer() {
    buildLegend();
    ticksInto($("#lr-ticks"));
    for (const [id, key] of [["#ds-seg", "ds"], ["#arch-seg", "arch"]])
      $(id).addEventListener("click", (e) => {
        const b = e.target.closest("button");
        if (!b) return;
        ex[key] = b.dataset.v;
        renderExplorer();
      });
    $("#lr-slider").addEventListener("input", (e) => {
      stopSweep();
      ex.lrPos = +e.target.value;
      renderExplorer();
    });
    $("#sweep-btn").addEventListener("click", () => (sweepTimer ? stopSweep() : startSweep()));
    renderExplorer();
  }

  // ------------------------------------------------------------------ digits: models
  const MODELS = {};
  function decode(arr) { // little-endian float32, or float64 for the huge weights of some large-rate runs
    const bin = atob(arr.data), n = bin.length, buf = new Uint8Array(n);
    for (let i = 0; i < n; i++) buf[i] = bin.charCodeAt(i);
    return arr.dtype === "f8" ? new Float64Array(buf.buffer) : new Float32Array(buf.buffer);
  }
  function loadModels() {
    if (!DM) return false;
    for (const m of DM.models) {
      if (!m.failed) for (const L of m.layers) { L.Wf = decode(L.W); L.bf = decode(L.b); }
      MODELS[m.method + "@" + DM.lrs.indexOf(m.lr)] = m;
    }
    return true;
  }
  function predict(model, raw) {
    if (model.failed) return model.const.slice();
    let h = raw.map((v, j) => (v - DM.mean[j]) / DM.std[j]);
    const n = model.layers.length;
    model.layers.forEach((L, li) => {
      const out = L.shape[0], inp = L.shape[1], z = new Array(out);
      for (let o = 0; o < out; o++) {
        let s = L.bf[o];
        const row = o * inp;
        for (let k = 0; k < inp; k++) s += L.Wf[row + k] * h[k];
        z[o] = li < n - 1 ? Math.max(0, s) : s;
      }
      h = z;
    });
    const mx = Math.max.apply(null, h), ex2 = h.map((v) => Math.exp(v - mx)), sum = ex2.reduce((a, b) => a + b, 0);
    return ex2.map((v) => v / sum);
  }

  // ------------------------------------------------------------------ digits: input
  const PAD = 270, BRUSH = 22;
  const dg = { strokes: [], example: null, lr: 9 };
  let drawing = null;

  function catmull(points, sub) {  // smooth a polyline (normalized coordinates) for drawing
    if (points.length < 3) return points;
    const out = [];
    for (let i = 0; i < points.length - 1; i++) {
      const p0 = points[Math.max(0, i - 1)], p1 = points[i], p2 = points[i + 1], p3 = points[Math.min(points.length - 1, i + 2)];
      for (let k = 0; k < sub; k++) {
        const t = k / sub, t2 = t * t, t3 = t2 * t;
        out.push([0, 1].map((d) => 0.5 * (2 * p1[d] + (-p0[d] + p2[d]) * t + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * t2 +
          (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * t3)));
      }
    }
    out.push(points[points.length - 1]);
    return out;
  }
  function strokeOn(ctx, strokes, size, width) {
    ctx.strokeStyle = "#111";
    ctx.fillStyle = "#111";
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const s of strokes) {
      if (!s.length) continue;
      if (s.length === 1) {
        ctx.beginPath();
        ctx.arc(s[0][0] * size, s[0][1] * size, width / 2, 0, 2 * Math.PI);
        ctx.fill();
        continue;
      }
      ctx.beginPath();
      ctx.moveTo(s[0][0] * size, s[0][1] * size);
      for (let k = 1; k < s.length; k++) ctx.lineTo(s[k][0] * size, s[k][1] * size);
      ctx.stroke();
    }
  }
  function canvas(w, h) {
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    return c;
  }
  // Optdigits preprocessing: size-normalize the ink into a 32x32 bitmap, count the on pixels of each 4x4 block
  function featuresFromStrokes(strokes) {
    const S0 = PAD, c = canvas(S0, S0), x = c.getContext("2d", { willReadFrequently: true });
    x.fillStyle = "#fff";
    x.fillRect(0, 0, S0, S0);
    strokeOn(x, strokes, S0, BRUSH);
    const d = x.getImageData(0, 0, S0, S0).data;
    let x0 = S0, y0 = S0, x1 = -1, y1 = -1;
    for (let yy = 0; yy < S0; yy++)
      for (let xx = 0; xx < S0; xx++)
        if (d[(yy * S0 + xx) * 4] < 128) {
          if (xx < x0) x0 = xx;
          if (xx > x1) x1 = xx;
          if (yy < y0) y0 = yy;
          if (yy > y1) y1 = yy;
        }
    const f = new Array(64).fill(0);
    if (x1 < 0) return f;
    const bw = x1 - x0 + 1, bh = y1 - y0 + 1, sc = 32 / Math.max(bw, bh);
    const dw = Math.max(1, Math.round(bw * sc)), dh = Math.max(1, Math.round(bh * sc));
    const s = canvas(32, 32), sx = s.getContext("2d", { willReadFrequently: true });
    sx.fillStyle = "#fff";
    sx.fillRect(0, 0, 32, 32);
    sx.imageSmoothingEnabled = true;
    sx.imageSmoothingQuality = "high";
    sx.drawImage(c, x0, y0, bw, bh, Math.floor((32 - dw) / 2), Math.floor((32 - dh) / 2), dw, dh);
    const e = sx.getImageData(0, 0, 32, 32).data;
    for (let yy = 0; yy < 32; yy++)
      for (let xx = 0; xx < 32; xx++)
        if (e[(yy * 32 + xx) * 4] < 160) f[Math.floor(yy / 4) * 8 + Math.floor(xx / 4)]++;
    return f;
  }
  function currentFeatures() {
    if (dg.example) return dg.example.x.slice();
    return featuresFromStrokes(dg.strokes.map((s) => (s.smooth ? catmull(s, 6) : s)));
  }

  const pad = $("#pad"), pctx = pad.getContext("2d");
  function drawBlocks(ctx, x, size) {  // 8x8 block counts as a gray image
    const c = size / 8;
    for (let r = 0; r < 8; r++)
      for (let q = 0; q < 8; q++) {
        const g = Math.round(255 * (1 - x[r * 8 + q] / 16));
        ctx.fillStyle = `rgb(${g},${g},${g})`;
        ctx.fillRect(q * c, r * c, c + 0.5, c + 0.5);
      }
  }
  function redrawPad() {
    pctx.fillStyle = "#fff";
    pctx.fillRect(0, 0, PAD, PAD);
    if (dg.example) {
      drawBlocks(pctx, dg.example.x, PAD);
      return;
    }
    strokeOn(pctx, dg.strokes.map((s) => (s.smooth ? catmull(s, 6) : s)), PAD, BRUSH);
    if (!dg.strokes.length) {
      pctx.fillStyle = "#9aa6b5";
      pctx.font = "600 15px Inter, Segoe UI, sans-serif";
      pctx.textAlign = "center";
      pctx.fillText("Draw a digit here", PAD / 2, PAD / 2);
    }
  }
  function drawFeatures(x) {
    const c = $("#feat"), ctx = c.getContext("2d"), S = c.width, cell = S / 8;
    ctx.clearRect(0, 0, S, S);
    for (let r = 0; r < 8; r++)
      for (let q = 0; q < 8; q++) {
        const v = x[r * 8 + q], a = v / 16;
        const col = [255 + (10 - 255) * a, 255 + (42 - 255) * a, 255 + (85 - 255) * a].map(Math.round);
        ctx.fillStyle = `rgb(${col.join(",")})`;
        ctx.fillRect(q * cell, r * cell, cell, cell);
        ctx.fillStyle = a > 0.45 ? "#fff" : "#6b7280";
        ctx.font = "600 11px Inter, Segoe UI, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(v), q * cell + cell / 2, r * cell + cell / 2 + 0.5);
      }
    ctx.strokeStyle = "#1a1a1a";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(0.75, 0.75, S - 1.5, S - 1.5);
  }

  // A network with low accuracy learned too little at a small rate, or broke down at a large one.
  const SMALL_LR = LRS.indexOf(0.1);
  function weakness(model) {
    return DM.lrs.indexOf(model.lr) <= SMALL_LR ? "learned too little" : "collapsed in training";
  }
  function modelStatus(model) {
    const acc = (100 * model.val_acc).toFixed(1);
    if (model.failed) return `<span class="status-bad">diverged at epoch ${model.epoch}; constant predictor</span>`;
    if (model.val_acc < 0.5) return `<span class="status-warn">${weakness(model)}; val. accuracy ${acc}%</span>`;
    return `<span class="status-good">val. accuracy ${acc}%</span>`;
  }
  function renderDigits() {
    redrawPad();
    $("#digit-lr").value = dg.lr;
    $("#digit-lr-value").textContent = lrLabel(LRS[dg.lr]);
    const x = currentFeatures();
    drawFeatures(x);
    $("#truth").innerHTML = dg.example ? `Test digit with true label <b>${dg.example.label}</b>.` : "";
    for (const c of document.querySelectorAll("#examples canvas")) c.classList.toggle("on", dg.example === c._ex);
    const body = $("#outs");
    body.textContent = "";
    const blank = x.every((v) => v === 0);
    for (const meth of METHODS) { // one table row per network: prediction, class probabilities, training outcome
      const tr = document.createElement("tr");
      tr.appendChild(methodCell(meth));
      const model = MODELS[meth + "@" + dg.lr];
      if (!model) {
        tr.insertAdjacentHTML("beforeend", '<td colspan="3">network not available</td>');
      } else if (blank) {
        tr.insertAdjacentHTML("beforeend", `<td>&ndash;</td><td class="has-text-grey">waiting for an input</td><td>${modelStatus(model)}</td>`);
      } else {
        const p = predict(model, x), top = p.indexOf(Math.max.apply(null, p));
        let hist = '<div class="hist">';
        p.forEach((v, k) => {
          hist += `<div class="hcol${k === top ? " top" : ""}"><div class="hbar" style="height:${Math.max(1, 100 * v).toFixed(1)}%` +
            `${k === top ? ";background:" + STYLE[meth].color : ""}"></div><span>${k}</span></div>`;
        });
        hist += "</div>";
        tr.insertAdjacentHTML("beforeend", `<td><span class="pred">${top}</span><span class="conf">${(100 * p[top]).toFixed(1)}%</span></td>` +
          `<td>${hist}</td><td>${modelStatus(model)}</td>`);
      }
      body.appendChild(tr);
    }
  }

  function initDigits() {
    const dpr = window.devicePixelRatio || 1;
    pad.width = Math.round(PAD * dpr);
    pad.height = Math.round(PAD * dpr);
    pctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ticksInto($("#digit-ticks"));
    if (!loadModels()) {
      $("#outs").innerHTML = '<tr><td colspan="4">The networks (data/models.js) are not available.</td></tr>';
    }
    const pos = (e) => {
      const r = pad.getBoundingClientRect();
      return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
    };
    let raf = 0;
    const schedule = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; renderDigits(); }); };
    pad.addEventListener("pointerdown", (e) => {
      if (Tour.playing()) return;
      pad.setPointerCapture(e.pointerId);
      if (dg.example) { dg.example = null; dg.strokes = []; }
      drawing = [pos(e)];
      dg.strokes.push(drawing);
      schedule();
    });
    pad.addEventListener("pointermove", (e) => {
      if (!drawing) return;
      drawing.push(pos(e));
      schedule();
    });
    const end = () => { drawing = null; schedule(); };
    pad.addEventListener("pointerup", end);
    pad.addEventListener("pointercancel", end);
    $("#clear-btn").addEventListener("click", () => { dg.strokes = []; dg.example = null; renderDigits(); });
    $("#random-btn").addEventListener("click", () => {
      if (!DM) return;
      dg.example = DM.examples[Math.floor(Math.random() * DM.examples.length)];
      dg.strokes = [];
      renderDigits();
    });
    $("#digit-lr").addEventListener("input", (e) => { dg.lr = +e.target.value; renderDigits(); });
    if (DM) {
      const strip = $("#examples");
      for (const e of DM.examples) {
        const c = canvas(32, 32);
        drawBlocks(c.getContext("2d"), e.x, 32);
        c.title = "Test digit " + e.label;
        c._ex = e;
        c.addEventListener("click", () => { dg.example = e; dg.strokes = []; renderDigits(); });
        strip.appendChild(c);
      }
    }
    renderDigits();
  }

  // ------------------------------------------------------------------ guided tour
  // A "7" for the tour, in pad coordinates (0..1); the tour draws it stroke by stroke.
  const TOUR_DIGIT = [[[0.24, 0.17], [0.5, 0.16], [0.76, 0.17], [0.66, 0.34], [0.55, 0.53], [0.47, 0.72], [0.43, 0.88]]];
  const TOUR_DIGIT_LABEL = 7;

  const Tour = (function () {
    const cap = $("#tour-caption"), card = $("#tour-card"), bar = $("#tour-progress");
    const DUR = 57.5;
    let playing = false, start = 0, raf = 0, cardKind = "";
    const clamp01 = (v) => Math.max(0, Math.min(1, v));
    const seg = (t, a, b) => clamp01((t - a) / (b - a));
    const ease = (p) => (p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2);
    const page = $("#page");
    // position of an element inside the page, independent of the current scroll or translation
    const yOf = (el) => el.getBoundingClientRect().top - page.getBoundingClientRect().top - 18;
    // live: scroll the window; recording: translate the page, which is exact for every frame
    const scrollPage = (y) => {
      if (RECORD) page.style.transform = `translateY(${-y}px)`;
      else window.scrollTo(0, y);
    };

    function predictionsText(lrIdx) {
      if (!DM) return "";
      const x = featuresFromStrokes(TOUR_DIGIT.map((s) => catmull(s, 6)));
      return METHODS.map((m) => {
        const model = MODELS[m + "@" + lrIdx], p = predict(model, x), top = p.indexOf(Math.max.apply(null, p));
        return `${STYLE[m].name} → ${top} ${top === TOUR_DIGIT_LABEL ? "✓" : "✗"}`;
      }).join(" ");
    }
    function statusWord(m, lrIdx) {
      const model = MODELS[m + "@" + lrIdx];
      if (!model) return "";
      if (model.failed) return "diverged in training";
      if (model.val_acc < 0.5) return `${weakness(model)} (val. accuracy ${(100 * model.val_acc).toFixed(0)}%)`;
      return `trained well (val. accuracy ${(100 * model.val_acc).toFixed(0)}%)`;
    }
    function caption(t) {
      if (t < 7 || t >= 51.5) return ""; // the title card already asks the question
      if (t < 9) return "Same network, same data: only the learning rate η changes. (HAR, 2 hidden layers)";
      if (t < 29) {
        const i = Math.round(ex.lrPos);
        if (i <= 3) return "Small η: BP+SGD learns, but ProxProp's damped steps are still too small.";
        if (i <= 6) return "BP+SGD trains well only for η ≈ 0.03–0.1.";
        if (i <= 10) return "BP+SGD collapses here. ProxProp works: its usable range has moved up.";
        return "ProxProp now fails numerically; ProxProp-all still trains.";
      }
      if (t < 33) return "Shift, not widen: ProxProp moves the range up; ProxProp-all also widens it.";
      if (t < 36) return "Input → model → output: three networks trained on Optdigits at η = 3.";
      if (t < 41.5) return `At η = 3, BP+SGD ${statusWord("bp_sgd", 9)}. ${t > 38.5 ? predictionsText(9) : ""}`;
      if (t < 46.5) return `At η = 0.01, ProxProp-all ${statusWord("prox_all", 4)}. ${predictionsText(4)}`;
      return `At η = 10, both proximal networks read the digit. ${predictionsText(10)}`;
    }
    function showCard(kind, alpha) {
      if (kind !== cardKind) {
        cardKind = kind;
        if (kind === "intro") {
          card.innerHTML = '<div class="venue">ECS 171 Machine Learning, Final Project</div>' +
            '<div class="card-title">Learning-Rate Sensitivity of Backpropagation</div>' +
            '<div class="card-sub">Does Proximal Backpropagation widen the range of learning rates that train well, or only move it?</div>' +
            '<div class="card-authors">Zekai Yu, Tingshuo Fan, Jiacheng Chen</div>' +
            '<div class="card-sub">University of California, Davis</div>';
        } else if (kind === "outro") {
          card.innerHTML = '<div class="card-title">Shift, not widening</div><ul>' +
            "<li>ProxProp moves the usable learning-rate range up by half a decade to two decades, but widens it in only 3 of 9 settings.</li>" +
            "<li>Applying the proximal step to the output layer too (ProxProp-all) widens the range in 8 of 9 settings, with no numerical failure in 585 runs.</li>" +
            "<li>ProxProp-all raises test accuracy on HAR by 1.1 to 1.9 points over BP+SGD.</li></ul>" +
            '<div class="url">fantasy-creator.github.io/Learning-Rate-Sensitivity-of-Backpropagation</div>';
        }
      }
      card.style.display = kind ? "flex" : "none";
      card.style.opacity = String(alpha);
    }

    function render(t) {
      t = Math.max(0, Math.min(DUR, t));
      // title cards
      if (t < 3.5) showCard("intro", Math.min(seg(t, 0, 0.5), 1 - seg(t, 3.0, 3.5)));
      else if (t >= 51.5) showCard("outro", seg(t, 51.5, 52.1));
      else showCard("", 0);
      // explorer state
      ex.ds = "har";
      ex.arch = "medium";
      ex.lrPos = (NLR - 1) * seg(t, 9, 29);
      // digits state
      dg.example = null;
      const draw = seg(t, 36, 39.5);
      if (t < 36) dg.strokes = [];
      else {
        const pts = catmull(TOUR_DIGIT[0], 6), n = Math.max(1, Math.round(draw * pts.length));
        dg.strokes = [pts.slice(0, n)];
      }
      dg.lr = t < 41.5 ? 9 : t < 46.5 ? 4 : 10;
      // scrolling
      // keep the controls and results below the caption bar at the top of the screen
      const yE = yOf($("#explorer .controls")) - 72, yD = yOf($("#digits .digit-grid")) - 108;
      let y;
      if (t < 5) y = 0;
      else if (t < 7.5) y = yE * ease(seg(t, 5, 7.5));
      else if (t < 33) y = yE;
      else if (t < 35.5) y = yE + (yD - yE) * ease(seg(t, 33, 35.5));
      else y = yD;
      scrollPage(Math.round(y));
      $("#explorer").classList.toggle("spot", t >= 7.5 && t < 33);
      $("#digits").classList.toggle("spot", t >= 35.5 && t < 51.5);
      renderExplorer();
      renderDigits();
      const c = caption(t);
      cap.style.display = c ? "block" : "none";
      cap.innerHTML = c + (RECORD ? "" : '<span class="hint">Guided tour &middot; press Esc to stop</span>');
      bar.style.display = "block";
      bar.style.width = (100 * t) / DUR + "%";
    }

    function stop() {
      playing = false;
      cancelAnimationFrame(raf);
      cap.style.display = "none";
      bar.style.display = "none";
      showCard("", 0);
      $("#explorer").classList.remove("spot");
      $("#digits").classList.remove("spot");
    }
    function play() {
      stopSweep();
      playing = true;
      start = performance.now();
      const step = (now) => {
        if (!playing) return;
        const t = (now - start) / 1000;
        if (t >= DUR) { stop(); return; }
        render(t);
        raf = requestAnimationFrame(step);
      };
      raf = requestAnimationFrame(step);
    }
    return { play, stop, render, DUR, playing: () => playing };
  })();

  // ------------------------------------------------------------------ start
  initExplorer();
  initDigits();
  $("#tour-btn").addEventListener("click", () => (Tour.playing() ? Tour.stop() : Tour.play()));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && Tour.playing()) Tour.stop(); });
  $("#tour-caption").addEventListener("click", () => Tour.playing() && Tour.stop());
  if (RECORD) {
    document.body.classList.add("recording");
    const v = $("#teaser-video"); // a still image instead of the (previously recorded) tour video
    if (v) {
      const img = document.createElement("img");
      img.src = v.getAttribute("poster");
      img.alt = "Study overview";
      img.className = "teaser-still";
      v.replaceWith(img);
    }
    window.__tour = { render: Tour.render, duration: Tour.DUR };
    if (RECORD_AT && RECORD_AT[1]) window.addEventListener("load", () => Tour.render(+RECORD_AT[1]));
  }
})();
