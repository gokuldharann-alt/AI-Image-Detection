/* Veritas frontend — click / drag-drop / paste upload + animated result + log streaming. No build step. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const dz = $("dropzone"), fileInput = $("fileInput"), analyzeBtn = $("analyzeBtn");
  const previewImg = $("previewImg"), dzEmpty = $("dzEmpty"), dzPreviewWrap = $("dzPreviewWrap");
  const fileMeta = $("fileMeta"), terminal = $("terminal");
  const resultCard = $("resultCard"), verdictEl = $("verdict");
  const confNum = $("confNum"), ringFg = $("ringFg");
  const realPct = $("realPct"), fakePct = $("fakePct"), realBar = $("realBar"), fakeBar = $("fakeBar");
  const teleGrid = $("teleGrid"), resultSkeleton = $("resultSkeleton");
  const engineDot = $("engineDot"), engineText = $("engineText");

  let currentFile = null;
  let analyzing = false;
  const RING_C = 263.9;

  /* ── terminal helpers ─────────────────────────────────────────── */
  function term(html, cls = "") {
    const p = document.createElement("p");
    p.className = `term-line ${cls}`;
    p.innerHTML = html;
    terminal.appendChild(p);
    terminal.scrollTop = terminal.scrollHeight;
    return p;
  }
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  /** Stream backend log lines one-by-one for the "live AI thought" feel. */
  async function streamLogs(lines, minGap = 90) {
    for (const line of lines) {
      term(esc(line), "term-dim");
      await new Promise((r) => setTimeout(r, minGap + Math.random() * 120));
    }
  }

  /* ── health / model info ──────────────────────────────────────── */
  async function boot() {
    try {
      const r = await fetch("/api/health");
      const h = await r.json();
      engineDot.className = "h-2 w-2 rounded-full bg-spotify";
      engineText.textContent =
        `${h.device || "CPU"} · ${h.engine || "pytorch"} · ${h.backbone || "efficientnet_b0"}`;
      term(`<span class="t-tag">✓</span> backend online — <span class="term-ok">${esc(h.device || "?")}</span> · engine=<b>${esc(h.engine || "?")}</b> · params=${Number(h.params || 0).toLocaleString()}`, "");
      if (!h.fine_tuned) term(`<span class="t-tag">!</span> <span class="term-warn">demo weights (ImageNet base) — run <b>python train.py</b> on CIFAKE for real accuracy.</span>`);
    } catch {
      engineDot.className = "h-2 w-2 rounded-full bg-red-500";
      engineText.textContent = "backend offline";
      term(`<span class="t-tag">✗</span> cannot reach backend. Is <b>uvicorn</b> running on :8000?`, "term-warn");
    }
  }

  /* ── file intake (shared by all 3 upload methods) ─────────────── */
  function acceptFile(file) {
    if (!file || analyzing) return;
    if (!file.type.startsWith("image/")) { term(`<span class="t-tag">✗</span> not an image: ${esc(file.name || "?")}`, "term-warn"); return; }
    if (file.size > 15 * 1024 * 1024) { term(`<span class="t-tag">✗</span> file &gt; 15 MB rejected.`, "term-warn"); return; }
    currentFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    dzEmpty.classList.add("hidden");
    dzPreviewWrap.classList.remove("hidden");
    fileMeta.textContent = `${file.name || "pasted-image.png"} · ${(file.size / 1024).toFixed(1)} KB · ${file.type}`;
    analyzeBtn.disabled = false;
    term(`<span class="t-tag">→</span> image loaded: <b>${esc(file.name || "clipboard.png")}</b> (${(file.size / 1024).toFixed(1)} KB) — hit <b>Analyze</b>.`);
  }

  // 1) click → file explorer
  dz.addEventListener("click", (e) => { if (e.target.id !== "changeBtn") fileInput.click(); });
  $("changeBtn").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
  fileInput.addEventListener("change", () => acceptFile(fileInput.files[0]));

  // 2) drag & drop with smooth hover animation
  let dragDepth = 0;
  window.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; dz.classList.add("dragging"); });
  window.addEventListener("dragleave", (e) => { e.preventDefault(); if (--dragDepth <= 0) { dragDepth = 0; dz.classList.remove("dragging"); } });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault(); dragDepth = 0; dz.classList.remove("dragging");
    const f = e.dataTransfer?.files?.[0];
    if (f) acceptFile(f);
  });

  // 3) copy/paste — intercept clipboard image data anywhere on the page
  window.addEventListener("paste", (e) => {
    const items = e.clipboardData?.items || [];
    for (const it of items) {
      if (it.type.startsWith("image/")) {
        e.preventDefault();
        const f = it.getAsFile();
        if (f) { term(`<span class="t-tag">⎘</span> pasted image from clipboard detected.`); acceptFile(f); }
        return;
      }
    }
  });

  /* ── analyze ──────────────────────────────────────────────────── */
  analyzeBtn.addEventListener("click", analyze);
  async function analyze() {
    if (!currentFile || analyzing) return;
    analyzing = true;
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = `<span class="inline-block animate-spin">◌</span>&nbsp; Analyzing…`;

    resultCard.classList.remove("hidden");
    resultSkeleton.classList.remove("hidden");
    teleGrid.innerHTML = "";
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    term(`<br><span class="text-spotify font-bold">$</span> veritas analyze <b>${esc(currentFile.name || "clipboard.png")}</b>`, "");

    const t0 = performance.now();
    try {
      const fd = new FormData();
      fd.append("file", currentFile, currentFile.name || "pasted.png");
      const res = await fetch("/api/predict", { method: "POST", body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const clientMs = performance.now() - t0;

      // skeleton → streamed logs → animated verdict (feels instantaneous yet alive)
      await streamLogs(data.logs || []);
      resultSkeleton.classList.add("hidden");
      renderResult(data, clientMs);
      term(`<span class="t-tag">✓</span> done in <b>${data.telemetry.total_ms} ms</b> server / ${clientMs.toFixed(0)} ms round-trip — verdict: <b>${data.label}</b> ${(data.confidence * 100).toFixed(1)}%`, data.label === "REAL" ? "term-ok" : "");
    } catch (err) {
      resultSkeleton.classList.add("hidden");
      term(`<span class="t-tag">✗</span> analysis failed: ${esc(err.message)}`, "term-warn");
    } finally {
      analyzing = false;
      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = "▶&nbsp; Analyze image";
    }
  }

  /* ── result rendering ─────────────────────────────────────────── */
  let lastResult = null;
  async function sendFeedback(trueLabel) {
    const msg = $("fbMsg");
    if (!currentFile || !lastResult) {
      if (msg) msg.textContent = "Analyze an image first, then give feedback.";
      return;
    }
    msg.textContent = "sending…";
    try {
      const fd = new FormData();
      fd.append("file", currentFile, currentFile.name || "pasted.png");
      fd.append("true_label", trueLabel);
      fd.append("pred_label", lastResult.label || "");
      fd.append("confidence", String(lastResult.confidence || 0));
      const res = await fetch("/api/feedback", { method: "POST", body: fd });
      const j = await res.json();
      if (!res.ok) throw new Error(j.detail || `HTTP ${res.status}`);
      msg.textContent = `✓ saved as ${trueLabel} (R=${j.totals.REAL} F=${j.totals.FAKE}). Thank you — retrain after 50+ per class.`;
      term(`<span class="t-tag">♥</span> feedback saved: true=<b>${trueLabel}</b> was=${lastResult.label} — totals R=${j.totals.REAL} F=${j.totals.FAKE}`);
    } catch (e) { msg.textContent = "✗ " + e.message; }
  }
  $("fbCorrect").addEventListener("click", () => lastResult && sendFeedback(lastResult.label));
  $("fbReal").addEventListener("click", () => sendFeedback("REAL"));
  $("fbFake").addEventListener("click", () => sendFeedback("FAKE"));
  function renderResult(d, clientMs) {
    lastResult = d;
    $("fbMsg").textContent = "";
    const isReal = d.label === "REAL";
    verdictEl.textContent = d.label === "REAL" ? "● REAL PHOTO" : "◆ AI-GENERATED";
    verdictEl.className = `mt-1 text-3xl font-black tracking-tight ${isReal ? "verdict-real" : "verdict-fake"}`;
    ringFg.style.stroke = isReal ? "#1DB954" : "#e879f9";

    // count-up animation for confidence
    const target = d.confidence * 100;
    const tStart = performance.now();
    (function tick(now) {
      const k = Math.min(1, (now - tStart) / 900);
      const eased = 1 - Math.pow(1 - k, 3);
      confNum.textContent = (target * eased).toFixed(1) + "%";
      ringFg.style.strokeDashoffset = RING_C * (1 - (d.confidence * eased));
      if (k < 1) requestAnimationFrame(tick);
    })(tStart);

    realPct.textContent = (d.prob_real * 100).toFixed(1) + "%";
    fakePct.textContent = (d.prob_fake * 100).toFixed(1) + "%";
    requestAnimationFrame(() => {
      realBar.style.width = d.prob_real * 100 + "%";
      fakeBar.style.width = d.prob_fake * 100 + "%";
    });

    const t = d.telemetry || {};
    const s = t.stages_ms || {};
    const chips = [
      ["⏱ Total", `${t.total_ms ?? "—"} ms`],
      ["🖥 Device", t.device || "—"],
      ["⚙ Engine", t.engine || "—"],
      ["🔢 Dtype", (t.dtype || "—").toUpperCase()],
      ["📐 Input", t.input_shape || "—"],
      ["📊 Logit", d.logit],
      ["🚀 Throughput", `${t.throughput_img_s ?? "—"} img/s`],
      ["🌐 Round-trip", `${clientMs.toFixed(0)} ms`],
    ];
    teleGrid.innerHTML = chips.map(([k, v]) =>
      `<div class="tele-chip"><p>${k}</p><p>${esc(v)}</p></div>`).join("");

    if (d.activations && d.activations.channels) {
      term(`<span class="t-tag">◉</span> <span class="term-dim">last-conv ${esc(JSON.stringify(d.activations.feature_map))} · sparsity ${(d.activations.sparsity * 100).toFixed(1)}% · top channels ${esc(JSON.stringify(d.activations.top_channels))}</span>`);
    }
  }

  boot();
})();
