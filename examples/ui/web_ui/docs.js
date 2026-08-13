// GAJE docs landing — interactividad (reveal, terminal, copy, bitácora)
(function () {
  "use strict";

  // Animaciones reveal al hacer scroll
  var revealEls = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) {
            e.target.classList.add("visible");
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.12 }
    );
    revealEls.forEach(function (el) { io.observe(el); });
  } else {
    revealEls.forEach(function (el) { el.classList.add("visible"); });
  }

  // --- Terminal simulador ---
  var termBody = document.getElementById("term");
  var termTabs = document.querySelectorAll(".terminal-tabs .tab-btn");

  var TERM_OUTPUT = {
    load: [
      "$ gaje-cli --load models/qwen2_5_1_5b_q4_0_q8_0_embd.gaje.flat",
      "🧬 [Zero-Copy Mmap] Formato .gaje.flat v2 detectado.",
      "✅ Carga instantánea por mmap en 0.75 ms.",
      "🧬 [ArchitectureDescriptor] Arquitectura Qwen2_5.",
      "✅ Organismo cargado y listo en memoria.",
      "> "
    ].join("\n"),
    status: [
      "$ gaje-cli --status",
      "Versión: 1.6.0-alpha",
      "Cuantización: Q4_0 (cuerpo) + FP32 (embeddings)",
      "SIMD: AVX2/FMA/AVX/SSE4.2",
      "Island Model (.gmem): activo, 512 tok de presupuesto",
      "Estado: LISTO",
      "> "
    ].join("\n"),
    benchmark: [
      "$ gaje-cli --benchmark",
      "qwen2_5_1_5b · 15 tokens · 1.8 s",
      "  → 8.3 tok/s (CPU nativa, release)",
      "smollm2_4bit · 95 tokens · 3.4 s",
      "  → 28 tok/s",
      "Zero-copy mmap load: 0.75 ms",
      "> "
    ].join("\n"),
    dni: [
      "$ gaje-cli --dni-verify",
      "✅ Ingesta Neuronal Directa (DNI) operativa.",
      "✅ Coherencia de pesos F32 reconstruidos > 0.99.",
      "✅ Paridad FP32 certificada en docs/reports/.",
      "> "
    ].join("\n")
  };

  function setTerm(tab, output) {
    termBody.textContent = output;
    termTabs.forEach(function (t) {
      t.classList.toggle("active", t.getAttribute("data-tab") === tab);
    });
  }

  termTabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      setTerm(tab.getAttribute("data-tab"), TERM_OUTPUT[tab.getAttribute("data-tab")]);
    });
  });

  if (termBody && TERM_OUTPUT.load) termBody.textContent = TERM_OUTPUT.load;

  // --- Botones copiar (terminal y code-box) ---
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    var ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
    return Promise.resolve();
  }

  document.querySelectorAll(".copy-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var targetId = btn.getAttribute("data-target");
      var src =
        targetId === "term"
          ? termBody
          : document.getElementById(targetId);
      if (!src) return;
      var old = btn.textContent;
      copyText(src.textContent).then(function () {
        btn.textContent = "✓ Copiado";
        setTimeout(function () { btn.textContent = old; }, 1600);
      });
    });
  });

  // --- Bitácora: filtros y búsqueda ---
  var entries = document.querySelectorAll(".log-entry");
  var searchInput = document.getElementById("bitacora-search");

  function applyFilters() {
    var active = document.querySelector(".filter-btn.active");
    var cat = active ? active.getAttribute("data-filter") : "all";
    var q = searchInput ? searchInput.value.trim().toLowerCase() : "";
    entries.forEach(function (entry) {
      var catOk = cat === "all" || entry.getAttribute("data-category") === cat;
      var textOk = !q || entry.textContent.toLowerCase().indexOf(q) !== -1;
      entry.style.display = catOk && textOk ? "" : "none";
    });
  }

  document.querySelectorAll(".filter-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      document.querySelectorAll(".filter-btn").forEach(function (b) { b.classList.remove("active"); });
      btn.classList.add("active");
      applyFilters();
    });
  });

  if (searchInput) searchInput.addEventListener("input", applyFilters);

  // --- Simulador de cuantización (slider) ---
  var slider = document.getElementById("weight-slider");
  var weightVal = document.getElementById("weight-val");
  var modeBtns = document.querySelectorAll(".sim-toggle-btn");

  var STATES = {
    "00": { sym: "A", label: "A (Adenina)", meaning: "Zona Inhibidora Fuerte" },
    "01": { sym: "C", label: "C (Citosina)", meaning: "Zona Inhibidora Suave" },
    "11": { sym: "G", label: "G (Guanina)", meaning: "Zona Activadora Positiva Suave" },
    "10": { sym: "T", label: "T (Timina)", meaning: "Zona Activadora Fuerte" }
  };
  var MODES = { auto: "Normal (Cuantizado 2-bit)", anchor: "Ancla de Estabilidad (Top 1%)" };
  var bitsEl = document.getElementById("genomic-bits");
  var meaningEl = document.getElementById("genomic-meaning");
  var precisionEl = document.getElementById("sim-precision");
  var symbolEl = document.getElementById("genomic-symbol");
  var modeLabel = null;

  function codeFor(v) {
    // 2-bit mapeo por umbrales
    if (v > 1.5) return "10";
    if (v > 0.25) return "11";
    if (v > -1.5) return "01";
    return "00";
  }

  function renderSim() {
    var v = parseFloat(slider ? slider.value : 0.45);
    if (weightVal) weightVal.textContent = (v > 0 ? "+" : "") + v.toFixed(3);
    var code = codeFor(v);
    var st = STATES[code];
    if (symbolEl) {
      symbolEl.textContent = code + " · " + st.sym;
    }
    if (bitsEl) bitsEl.textContent = "2 bits (" + code + ")";
    if (meaningEl) meaningEl.textContent = st.meaning;
    if (precisionEl) precisionEl.textContent = "2 bits (16× compresión)";
  }

  if (slider) {
    slider.addEventListener("input", renderSim);
    renderSim();
  }

  if (modeBtns.length) {
    modeBtns.forEach(function (b) {
      b.addEventListener("click", function () {
        modeBtns.forEach(function (x) { x.classList.remove("active"); });
        b.classList.add("active");
        var m = b.getAttribute("data-mode");
        if (precisionEl) precisionEl.textContent = m === "anchor" ? "Ancla Top 1% (F16 preservado)" : "2 bits (16× compresión)";
      });
    });
  }
})();