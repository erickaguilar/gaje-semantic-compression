/* ==========================================================================
   GAJE — Interactive Application Script
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {

  // --------------------------------------------------------------------------
  // 1. Terminal Simulator Logic
  // --------------------------------------------------------------------------
  const terminalCommands = {
    load: [
      "$ ./target/release/gaje-cli --load silver_adult_sovereign.gaje",
      "🧬 Despertando organismo GAJE desde base de datos...",
      "⚡ SIMD NEON JIT descompresión inicializada (ARM64).",
      "✅ Reconstructed 30 transformer blocks (2.1GB F32 -> 168MB .gaje).",
      "[*] Reconstrucción de tensores finalizada en 3.05s",
      "[STATUS] Infraestructura: verificada — similitud de pesos > 0.99",
      "[STATUS] Capa semántica: en evaluación activa — PPL bajo investigación"
    ],
    status: [
      "$ ./target/release/gaje-cli --status",
      "📊 GAJE Engine System Diagnostic v1.0.0-alpha (Silver Adult)",
      "-------------------------------------------------------------",
      "• Memory Footprint: 32MB (O(1) continuous mmap allocation)",
      "• Rust Core Sovereign: 100% Zero-GIL / Pure Rust C-ABI",
      "• Bit-Depth Resolution: Attention 4-bit / FFN 2-bit (Mixed-bit)",
      "• Stability Anchors: Top 1.0% precision preserved (Γ = 8.0)",
      "• GQA Head Symmetry: GQA 8:1 ratio aligned (0 crash state)"
    ],
    benchmark: [
      "$ ./target/release/gaje-cli --benchmark --tokens 512",
      "🔬 Running Latency & Memory Throughput Benchmarks...",
      "-------------------------------------------------------------",
      "• Token generation speed: 42.8 tokens/sec (ARM Cortex-A78)",
      "• Quantization throughput: 1.84 GB/s vector SIMD unpacking",
      "• Memory bandwidth utilization: 8.2% (ultra-low power consumption)",
      "• Cosine similarity (Orig F32 vs Reconstructed): 0.9942"
    ],
    dni: [
      "$ ./target/release/gaje-cli --dni-verify --dataset wikitext-2",
      "🧪 Direct Neuronal Ingestion (DNI) Evaluation...",
      "-------------------------------------------------------------",
      "• Layer-by-layer Fisher Information Metric: Heterogeneous",
      "• Entropy metric H(h_norm): 4.12 bits",
      "• Dynamic RNA strands: Active (up to 4-bit precision fallback)",
      "• Current PPL: ~572.0 (Resampling phase in progress)"
    ]
  };

  const termEl = document.getElementById('term');
  const tabBtns = document.querySelectorAll('.tab-btn');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let currentTab = 'load';
  let typeTimeout = null;

  function renderStaticTerminal(lines) {
    if (!termEl) return;
    termEl.textContent = lines.join('\n');
  }

  function typeTerminal(lines) {
    if (!termEl) return;
    if (typeTimeout) clearTimeout(typeTimeout);

    if (reducedMotion) {
      renderStaticTerminal(lines);
      return;
    }

    let lineIdx = 0;
    let charIdx = 0;
    let accumulatedText = '';

    function step() {
      if (lineIdx >= lines.length) {
        termEl.innerHTML = accumulatedText + '<span class="caret"></span>';
        return;
      }

      const currentLine = lines[lineIdx];
      if (charIdx <= currentLine.length) {
        termEl.innerHTML = accumulatedText + currentLine.slice(0, charIdx) + '<span class="caret"></span>';
        charIdx++;
        typeTimeout = setTimeout(step, 10 + Math.random() * 18);
      } else {
        accumulatedText += currentLine + '\n';
        lineIdx++;
        charIdx = 0;
        typeTimeout = setTimeout(step, 200);
      }
    }

    step();
  }

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentTab = btn.getAttribute('data-tab');
      typeTerminal(terminalCommands[currentTab] || terminalCommands.load);
    });
  });

  // Initial terminal animation
  typeTerminal(terminalCommands.load);


  // --------------------------------------------------------------------------
  // 2. Interactive Quantization Simulator Logic
  // --------------------------------------------------------------------------
  const weightSlider = document.getElementById('weight-slider');
  const weightValEl = document.getElementById('weight-val');
  const genomicSymbolEl = document.getElementById('genomic-symbol');
  const genomicBitsEl = document.getElementById('genomic-bits');
  const genomicMeaningEl = document.getElementById('genomic-meaning');
  const simPrecisionEl = document.getElementById('sim-precision');
  const simEnergyEl = document.getElementById('sim-energy');
  const simToggles = document.querySelectorAll('.sim-toggle-btn');

  let isAnchorMode = false;

  function updateQuantization() {
    if (!weightSlider || !weightValEl) return;

    const val = parseFloat(weightSlider.value);
    weightValEl.textContent = (val >= 0 ? '+' : '') + val.toFixed(3);

    // If anchor protection mode is forced or magnitude > 2.2
    if (isAnchorMode || Math.abs(val) > 2.2) {
      genomicSymbolEl.className = 'genomic-symbol state-anchor';
      genomicSymbolEl.innerHTML = '<span>⚡ ANCLA</span>';
      genomicBitsEl.textContent = 'F32 (32-bit Preservado)';
      genomicMeaningEl.textContent = 'Top 1% Estabilidad (sin cuantizar)';
      simPrecisionEl.textContent = '100% (F32 Nativo)';
      simEnergyEl.textContent = 'Γ = 8.0 (Invariante)';
      return;
    }

    // 2-bit Genomic Alphabet Mapping:
    // [-3.0, -1.0) -> A (00)
    // [-1.0, 0.0)  -> C (01)
    // [0.0, 1.0)   -> G (11)
    // [1.0, 3.0]   -> T (10)
    if (val < -1.0) {
      genomicSymbolEl.className = 'genomic-symbol state-a';
      genomicSymbolEl.innerHTML = '<span>00</span> · A (Adenina)';
      genomicBitsEl.textContent = '2 bits (00)';
      genomicMeaningEl.textContent = 'Zona Represora Negativa Alta';
      simPrecisionEl.textContent = '2 bits (16× compresión)';
      simEnergyEl.textContent = '0.041 (Plástico)';
    } else if (val < 0.0) {
      genomicSymbolEl.className = 'genomic-symbol state-c';
      genomicSymbolEl.innerHTML = '<span>01</span> · C (Citosina)';
      genomicBitsEl.textContent = '2 bits (01)';
      genomicMeaningEl.textContent = 'Zona Represora Negativa Suave';
      simPrecisionEl.textContent = '2 bits (16× compresión)';
      simEnergyEl.textContent = '0.088 (Plástico)';
    } else if (val < 1.0) {
      genomicSymbolEl.className = 'genomic-symbol state-g';
      genomicSymbolEl.innerHTML = '<span>11</span> · G (Guanina)';
      genomicBitsEl.textContent = '2 bits (11)';
      genomicMeaningEl.textContent = 'Zona Activadora Positiva Suave';
      simPrecisionEl.textContent = '2 bits (16× compresión)';
      simEnergyEl.textContent = '0.092 (Plástico)';
    } else {
      genomicSymbolEl.className = 'genomic-symbol state-t';
      genomicSymbolEl.innerHTML = '<span>10</span> · T (Timina)';
      genomicBitsEl.textContent = '2 bits (10)';
      genomicMeaningEl.textContent = 'Zona Activadora Positiva Alta';
      simPrecisionEl.textContent = '2 bits (16× compresión)';
      simEnergyEl.textContent = '0.045 (Plástico)';
    }
  }

  if (weightSlider) {
    weightSlider.addEventListener('input', updateQuantization);
  }

  simToggles.forEach(btn => {
    btn.addEventListener('click', () => {
      simToggles.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      isAnchorMode = (btn.getAttribute('data-mode') === 'anchor');
      updateQuantization();
    });
  });

  updateQuantization();


  // --------------------------------------------------------------------------
  // 3. Bitácora Filtering & Searching
  // --------------------------------------------------------------------------
  const filterBtns = document.querySelectorAll('.filter-btn');
  const searchInput = document.getElementById('bitacora-search');
  const logEntries = document.querySelectorAll('.log-entry');

  let activeFilter = 'all';
  let searchQuery = '';

  function filterLogEntries() {
    logEntries.forEach(entry => {
      const category = entry.getAttribute('data-category');
      const text = entry.textContent.toLowerCase();

      const matchesCategory = (activeFilter === 'all' || category === activeFilter);
      const matchesSearch = (searchQuery === '' || text.includes(searchQuery));

      if (matchesCategory && matchesSearch) {
        entry.style.display = 'grid';
      } else {
        entry.style.display = 'none';
      }
    });
  }

  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeFilter = btn.getAttribute('data-filter');
      filterLogEntries();
    });
  });

  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      searchQuery = e.target.value.toLowerCase().trim();
      filterLogEntries();
    });
  }


  // --------------------------------------------------------------------------
  // 4. Copy Code Snippets
  // --------------------------------------------------------------------------
  const copyBtns = document.querySelectorAll('.copy-btn');

  copyBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-target');
      let textToCopy = '';

      if (targetId) {
        const el = document.getElementById(targetId);
        if (el) textToCopy = el.textContent;
      } else if (btn.getAttribute('data-copy')) {
        textToCopy = btn.getAttribute('data-copy');
      } else if (currentTab && terminalCommands[currentTab]) {
        textToCopy = terminalCommands[currentTab].join('\n');
      }

      if (textToCopy) {
        navigator.clipboard.writeText(textToCopy).then(() => {
          const originalText = btn.textContent;
          btn.textContent = '¡Copiado!';
          btn.style.borderColor = 'var(--signal)';
          btn.style.color = 'var(--signal)';
          setTimeout(() => {
            btn.textContent = originalText;
            btn.style.borderColor = '';
            btn.style.color = '';
          }, 2000);
        }).catch(err => {
          console.error('Copy failed', err);
        });
      }
    });
  });


  // --------------------------------------------------------------------------
  // 5. Scroll Reveal Animations
  // --------------------------------------------------------------------------
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('in');
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  document.querySelectorAll('.reveal').forEach(el => observer.observe(el));

});
