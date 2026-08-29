# 🧬 GAJE Semantic Compression — Global Development Protocol & AI Agent Guide

[![Language: Spanish](https://img.shields.io/badge/Language-Espa%C3%B1ol-yellow.svg)](AGENTS.md) [![Language: Chinese](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red.svg)](AGENTS.zh.md)

This document defines the **global project description**, repository architecture, and the **mandatory operational and technical standards** for human developers and Artificial Intelligence agents (Antigravity, Claude, Gemini, Copilot, etc.).

---

## 1. Global Project Description

**GAJE (Genetic Adaptive Joint Embedding / DNA Semantic Compression)** is a hybrid framework for semantic compression and neuronal genetic memory. Its objective is to compress and retrieve dense semantic representations at a genomic/neuronal level with ultra-low latency, combining genetic algorithms, zero-copy mmap memory, and lateral K-WTA inhibition.

### Key Components:
* **Native Core (Rust):** High-efficiency implementation for compression/decompression, vector search, mmap, and mathematical kernels.
* **Research Layer (Python):** Interface, embeddings, PyTorch/HuggingFace integrations, and experimental tooling.
* **Web Interface (Web UI & Theme System):** Interactive panel featuring chat, HUD telemetry, architecture visualizer, and live documentation.

---

## 2. Repository Map and Architecture

* **`/src`**: Native Rust core (`gaje-core` library, `gaje-cli` binary, SIMD kernels, data structures).
* **`/python`**: Python module `gaje` and bindings to the native core.
* **`/examples/ui/web_ui`**:
  * `index.html`: Interactive chat and compression telemetry HUD.
  * `docs.html`: Interactive documentation center.
  * `architecture.html`: System graph and visualizer.
  * `server.py`: Native HTTP backend server (Python stdlib `http.server`) for the Web UI.
  * `static/`: Stylesheets (`css/base.css`, `css/chat.css`), scripts, and Y2K sprite icons (`static/icons/y2k/sprite.svg`).
* **`/docs`**: Classified documentation (`guides/`, `plans/`, `bdd/`, `reports/`, `meta/`, `research/`).
* **`/tests`**: Unit, integration, and metrics test suites (`pytest`, `cargo test`).
* **`/benchmarks`**: Evaluation suites for latency, throughput, and entropy.
* **`/data`**: Models, datasets, logs, and genomic artifacts.

---

## 3. Golden Rules for Developers & AI Agents

### A. Core Sovereignty & Memory Management (Rust & Python)
1. **Memory Efficiency:** Unnecessary mass tensor pre-allocations in hot critical loops are strictly prohibited. Always prioritize raw pointers, references, and zero-copy shared memory (`Arc<Vec<u8>>`, `mmap`).
2. **Native Sovereignty:** High-performance features or administrative CLI tools must be implemented as commands in `gaje-cli` (Rust), avoiding disposable monolithic scripts.
3. **No Module Collisions:** In `python/gaje/`, do not create directory structures that collide with compiled binary extensions (`_impl`).

### B. Y2K Design System & Dual-Theme Architecture

#### 1. Theme Definitions & Philosophy:
* **`y2k-dark = 'HIG-APPLE'` (Default Dark Theme):**
  * **Concept:** Intersection between Apple Human Interface Guidelines (HIG) dark materials and Web 1.0 Y2K cyberpunk futurism. Creates an advanced genomic research console atmosphere with zero visual fatigue.
  * **Background & Panels:** Pure black background (`#000000`), translucent carbon panels (`#1c1c1e`, `#2c2c2e`), and `backdrop-filter: blur(20px)` (OS-grade glassmorphism).
  * **Neon & DNA Accents:** Electric Blue (`#0a84ff`), Violet (`#5e5ce6` / `#a78bfa`), Terminal Cyan (`#22d3ee`), Neon Pink (`#f472b6`), and Matrix Green (`#30d158`). DNA Bases: A (`#ff453a`), C (`#0a84ff`), G (`#30d158`), T (`#ffd60a`).
  * **Retro Layers:** CSS-generated CRT scanlines (`repeating-linear-gradient`), 115° reflective sheen, and interactive blinking terminal cursor (`brand-underscore`).
* **`y2k-light = 'SCANDINAVIAN-DESIGN'` (Light Theme via `[data-theme="light"]`):**
  * **Concept:** Based on Nordic/Scandinavian design principles (democratic functionalism, warm *hygge* minimalism, connection with nature, and light maximization), fused with the **Field & Research Laboratory Notebook** paradigm. Visually portrays the model as a living organism engaged in continuous learning and semantic memory consolidation.
  * **Square Geometry (0px radius):** In `y2k-light`, all interface components (note cards, response bubbles, buttons, containers, dropdowns, and badges) are strictly rectangular with 90-degree corners (`border-radius: 0px`), evoking technical archive index cards and engineering notebooks.
  * **Background & Paper:** Soft ivory / clean parchment background (`#f6f5f3` / `#edebe9`) with subtle 24px dot-grid and amber/crimson margin guide lines; response panels styled as bound lab cards in structured ivory white (`#ffffff`).
  * **Organic & DNA Accents:** Forest / Deep Jade Green (`#2c5234`), Slate (`#2c3539`), Botanical Amber (`#b45309`), and Crimson (`#b91c1c`). Thought disclosure module styled as an engineering field notes memorandum.
  * **UX & Ergonomics:** Clean geometric typography (*Inter* / *Plus Jakarta Sans*), high-legibility neutral micro-borders, inverted high-contrast icons, and zero eye strain in daylight environments.

#### 2. Frontend Technical Rules for Agents & Developers:
* **Golden Overflow Rule:** `overflow: hidden` is **strictly prohibited** on `.y2k-header` to avoid clipping dropdown menus and floating tooltips.
* **Z-Index Hierarchy:**
  * `z-index: 1` and `2`: CRT scanline and glass sheen pseudo-elements (`::before` / `::after`).
  * `z-index: 3`: Main bar content `.wrap`.
  * `z-index: 200`: Menu dropdowns (`.y2k-menu-dropdown`) and modals (`.y2k-apple-modal`).
* **3D Bevel Buttons:** Maintain Web 1.0 tactile depth with interior shadows (`box-shadow: inset 1px 1px 0 rgba(255,255,255,.18), inset -1px -1px 0 rgba(0,0,0,.35)`) that invert on press (`:active`).

### C. Empirical Truth and Certification
1. **Compilation does not equal semantic success:** Code compiling successfully does not certify compression accuracy. Perplexity (PPL) and semantic distance validations must be formally proven.
2. **Development Lifecycle:** Design under SDD (specifications) -> BDD (*Given-When-Then* scenarios) -> TDD (unit/integration tests).

### D. Long-Running Processes & Strict Token Conservation
1. **Computation-Intensive Nature:** Tasks in this repository (Rust `--release` compilations, `.flat` weight quantization and export, perplexity/entropy benchmarks, and large model downloads) are compute-heavy operations that may take seconds or minutes depending on hardware.
2. **Absolute Prohibition of Active Polling (`manage_task status`):**
   * Each invocation of `manage_task` to query the status of an ongoing task consumes both output tokens (*tool call*) and input tokens (*logs, stdout/stderr injected back into context*).
   * Polling in loops prematurely exhausts the context window and the session's token budget.
3. **Mandatory Reactive Pattern (*Reactive Wakeup*):**
   * After dispatching an asynchronous command with `run_command`, AI agents **must immediately stop calling tools** and yield the turn to the environment.
   * The environment automatically notifies and wakes up the agent when the background process finishes with its exit code and output.
4. **Parameter `WaitMsBeforeAsync`:**
   * For compute-heavy operations, set `WaitMsBeforeAsync` appropriately (e.g. 5000–10000 ms) and allow long processes to run in the background without intermediate polling calls.

---

## 4. Frequent Development Commands

```bash
# Compile native Rust core in optimized release mode
cargo build --release

# Run Rust test suite
cargo test

# Run Python test suite
pytest tests/

# Launch Web UI local server
python examples/ui/web_ui/server.py --port 8080
```

---

## 5. Commit Standards
Use **Conventional Commits**:
* `feat(module):` New feature
* `fix(module):` Bug fix
* `perf(module):` Performance optimization
* `docs(module):` Documentation updates
* `style(ui):` Visual or styling adjustments (Y2K / CSS)
* `refactor(module):` Code refactoring without behavior change
