# 🧬 GAJE Protocol: Semantic Adaptation & Genomic Compression (v1.7.4-alpha)

[![Version](https://img.shields.io/badge/version-1.7.4--alpha_Helix_Ecosystem-purple)](docs/meta/EMPIRICAL_TRUTH_STATE.md)
[![Engine](https://img.shields.io/badge/Engine-Pure_Rust_PyO3_WASM-orange.svg)](src/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Format](https://img.shields.io/badge/Format-Zero--Copy_GAJE_mmap-brightgreen.svg)](docs/plans/UNIFIED_GAJE_ADAPTIVE_FORMAT_PLAN.md)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20Hub-yellow)](https://huggingface.co/eaguilar/gaje-models)
[![Language: Spanish](https://img.shields.io/badge/Language-Espa%C3%B1ol-yellow.svg)](README.md)
[![Language: Chinese](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red.svg)](README.zh.md)

**GAJE (Genomic Adaptive Joint Embedding)** is an ultra-high-density research and computing protocol designed for the execution and compression of Large Language Models (LLMs). The protocol quantizes parameter spaces down to a discrete **4-bit per weight** representation (16 optimized centroids) and **2-bit per weight** (experimental neuromorphic front), integrating the sovereign **`gaje-server` (Zero-Python Runtime)** with real-time SSE token-by-token streaming and concurrent hot-swap, persistent zero-copy memory (**Island Model `.gmem`**), dynamic self-describing headers (**`ArchitectureDescriptor`**), in-place adaptive centroid mutations with lineage tracking, instant memory-mapped file loading (**unified `.gaje` v2**), and in-browser **WebAssembly inference (Zero-Server)**.

---

## 📦 Certified Model Catalog (Hugging Face Hub)

Official pre-packaged models are available at the [Official Hugging Face Hub (`eaguilar/gaje-models`)](https://huggingface.co/eaguilar/gaje-models):

| Organism / Model | Format | Size | Optimal Runtime | Capabilities |
| :--- | :---: | :---: | :---: | :--- |
| **`gaje_nano_1.5b.gaje`** | `.gaje` v2 | **1.23 GB** | WebAssembly (Mobile / Web) | Ultra-fast, minimal RAM footprint, ideal for phones. |
| **`gaje_prime_3b.gaje`** | `.gaje` v2 | **2.24 GB** | WASM Desktop / Cloud | Balanced, high general reasoning and contextual depth. |
| **`gaje_ultra_7b.gaje`** | `.gaje` v2 | **4.88 GB** | Server / Cloud Native | Deep reasoning, multi-turn coding and complex analysis. |

---

## 🔬 Empirical Status & Scientific Diagnosis (v1.7.3-alpha)

Following the principle of **Empirical Truth** ([`docs/meta/EMPIRICAL_TRUTH_STATE.md`](docs/meta/EMPIRICAL_TRUTH_STATE.md)), the system presents the following certified functional state:

### 🏆 1. A/B Parity Control Experiment (GAJE Q4_0 vs. HuggingFace PyTorch FP32)

We executed an A/B parity trial comparing the original FP32 model (`Qwen/Qwen2-0.5B-Instruct`) in **PyTorch** against the native **GAJE 4-bit `.gaje`** engine on an **AMD Ryzen 7 5800H** CPU:

| Inference Engine | Format / Precision | Exact Generated Response | Real E2E Throughput | RAM Consumption |
| :--- | :---: | :--- | :---: | :--- |
| **HuggingFace PyTorch** | **FP32 Original (Alibaba)** | *"El planeta más grande del Sistema Solar es la Tierra, con una"* | **`1.38 tok/s`** | $1,980\text{ MB}$ |
| **GAJE Native Engine (`.gaje`)** | **4-bit Genomic Zero-Copy** | *"El planeta más grande del Sistema Solar es la Tierra."* | **`19.2 - 23.0 tok/s`** | **`448 MB` (RSS, ~77% vs FP32)** |

---

### ⚡ 2. Certified Multimodel Production Performance (Ryzen 7 5800H)

| Model / Architecture | Binary Format | Certified Factual Response | CPU Throughput | Cold Start Load Time | Live RAM RSS |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Qwen2.5 1.5B Instruct** | **`.gaje` (Hybrid v2)** | Spanish: *"La capital de Francia es París."* | **`11.31 - 12.13 tok/s`** | **`< 0.75 ms` (mmap)** | **`2.6 GB` (Virtual)** |
| **Qwen2 0.5B Instruct** | **`.gaje` (Hybrid v2)** | Chinese: *"木星"* (Jupiter) / Spanish: *"París"* | **`19.20 - 23.00 tok/s`** | **`< 0.75 ms` (mmap)** | **`~498 MiB` (~74% vs FP32)** |
| **SmolLM2 135M Instruct** | **`.gaje` (Zero-Copy)** | English: *"Berlin."* / *"100°C"* | **`28.28 - 32.10 tok/s`** | **`< 0.75 ms` (mmap)** | **`~472 MB` (Q4_0 body + FP32 embeddings)** |

> [!IMPORTANT]
> **Hybrid .gaje v2 Layout**: To preserve semantic representation fidelity and avoid vocabulary collapse in high-density languages (like Chinese/Arabic), the `.gaje` format (with `.flat` retained as a backward-compatible alias) stores critical semantic layers (`token_embd` and `lm_head`) in **FP32** (4 bytes/weight), while the transformer body (attention and FFN projections) is quantized to **Q4_0** (4-bits) or **Q2_0** (2-bits).

---

### 🏝️ 3. Island Model (.gmem): Sub-Millisecond Hippocampal RAG

The system integrates real-time contextual memory retrieval and persistence through 64-byte aligned flat binary indices (`.gmem`):

* **Vector Retrieval Latency (RAG)**: **`< 0.5 ms`** per multi-niche query via sovereign *Weighted Mean Pooling* over input token embeddings ($W_E$).
* **Entropy Gap Gating ($\Delta_{top} \ge 0.12$)**: K-WTA competitive lateral inhibition and mathematical rejection of ambiguous or competitive trap queries.
* **Satellite Whitening Calibration**: Anisotropic centering ($\boldsymbol{\mu} \in \mathbb{R}^D$) persisted under `data/calibration/` with graceful fallback (Option C).
* **Canonical 6-State Telemetry**: Real-time monitoring across SSE and server metrics (`memory_injected`, `rejected_low_similarity`, `rejected_entropy_gap`, etc.).
* **Context Budget**: Protected injection strictly within the system prompt block (`ChatML` / `Llama3`) preserving turn fidelity.

---

## ⚡ Quick Start Guide — Sovereign Single-Binary (`gaje-cli`)

GAJE Helix runs as a **self-contained standalone native Rust executable** requiring zero external dependencies or Python runtime in production.

### 1. Build the Release Binary
```bash
cargo build --release --bin gaje-cli
```

### 2. Core `gaje-cli` Commands

```bash
# Run HTTP SSE streaming server with embedded in-memory Web UI
./target/release/gaje-cli serve --port 8080

# Interactive Chat REPL session
./target/release/gaje-cli chat --model models/production/gaje_pico_135m.gaje

# Pull models directly from Hugging Face
./target/release/gaje-cli pull pico

# Inspect and verify local models catalog
./target/release/gaje-cli models list
./target/release/gaje-cli models inspect models/production/gaje_pico_135m.gaje

# In-place genomic centroid mutation (.gaje)
./target/release/gaje-cli mutate --model models/production/gaje_pico_135m.gaje --rate 0.05

# Inspect genetic lineage and adaptive generations
./target/release/gaje-cli history --model models/production/gaje_pico_135m.gaje

# Export any model (.gguf, .gaje) to zero-copy flat v2
./target/release/gaje-cli export-flat models/source/model.gguf -o models/production/model.gaje

# Throughput (TPS), TTFT and Perplexity (PPL) benchmark suite
./target/release/gaje-cli benchmark --model models/production/gaje_pico_135m.gaje --tokens 64

# Audit tensor weights (zero NaN / zero Inf certificate)
./target/release/gaje-cli audit models/production/gaje_pico_135m.gaje

# Diagnostic hardware extensions (AVX2, AVX512, NEON, FMA)
./target/release/gaje-cli doctor
```

### 3. Native Test Suites
```bash
# Run native Rust unit & integration test suites
cargo test --lib
cargo test --test cli_standalone_test
```

---

## 🛠️ Architectural Foundations

### 1. Dynamic Self-Describing Flat Headers (`.gaje` v2)
The **`FlatHeaderV2`** binary structure implements an autodescriptive **`ArchitectureDescriptor`**. Model dimensions, RoPE parameters, and attention permutation patterns ($Q/K$) are dynamically parsed and written to the fixed 4096-byte header, including cryptographic lineage tracking and mutation counters.

### 2. Quantization-Aware Training (QAT) Stabilization
GAJE supports native local adaptation algorithms. Centroid updates in Rust (`storage.rs`) are normalized by dividing the accumulated gradients by the actual activation count (`centroid_counts`), preventing gradient explosion (`NaN` / `Inf`) and guaranteeing mathematical convergence of quantization loss.

### 3. Minimum Action Lagrangian Sampling
Token generation is modeled as a dynamic system governed by the Principle of Least Action, evaluating kinetic energy $T$ (semantic mobility) and potential energy $V$ (grammatical constraint):

$$\mathcal{L} = T - V$$

---

## 📂 Repository Organization (`v1.7.2-alpha`)

```text
gaje-semantic-compression/
├── src/                    # Rust Native Core (SIMD Kernels, LLM Engine, KV-Cache, Mmap Loader)
│   ├── bin/gaje-cli.rs     # Sovereign single binary CLI
│   ├── cli/                # Terminal presentation and subcommands (models, tools)
│   ├── compute/            # Pure numeric compute (sintergic, sampling, SIMD kernels, quantize)
│   ├── core/               # Base structures (gtok, tokenizers, session ring buffer, DNI)
│   ├── io/                 # Zero-copy mmap I/O (adaptive, flat_reader, flat_writer, gguf/, gmem)
│   ├── nn/                 # Neural layers (WeightStorage, GenomicLLM, Spiking, Distiller)
│   └── server/             # Embedded native HTTP server with SSE streaming (tiny_http)
├── python/gaje/            # PyO3 Bridge and Native Python Infrastructure Wrappers
├── examples/               # Core demos, Web UI, notebooks and Rust utilities
│   └── ui/web_ui/          # Web UI Frontend (http://localhost:8080) and server.py server
├── tests/                  # Verification Suite (unit, integration, metrics, training, ui_e2e)
├── scripts/                # Maintenance scripts and transmuters
├── models/production/      # Quantized Production Models (.gaje flat v2)
└── docs/                   # Scientific papers, blueprints, and reports
    ├── reports/            # Verified empirical results (parity reports and benchmarks)
    ├── guides/             # Operational manuals (GAJE CLI, workflows)
    ├── plans/              # Roadmaps and strategic plans
    ├── meta/               # Governance and empirical truth state
    └── archive/            # Exploratory research and legacy versions
```

> **Consolidation note:** experimental content (exploratory Rust binaries, research notes and prior-stage demos) is fully preserved under `legacy/` and `docs/archive/`. The main tree only keeps operational, verified components.

---

## ⚡ Quick Start Guide & Web UI Deployment

### 1. Installation and Compilation (PyO3)
```bash
# Create virtual environment
uv venv && source .venv/bin/activate

# Build Rust native core with CPU target optimizations
maturin develop --release --features python
```

### 2. Sovereign Production HTTP Server (`gaje-server`) & CLI
```bash
# Compile optimized production binary
cargo build --release --bin gaje-cli

# Start sovereign native HTTP server (Zero-Python, real-time SSE streaming, embedded UI)
./target/release/gaje-cli serve --port 8080

# Interactive terminal Chat REPL
./target/release/gaje-cli chat --model models/production/gaje_pico_135m.gaje

# Or optionally run lightweight Python development server:
python examples/ui/web_ui/server.py
```
Open `http://localhost:8080` in your browser to interact with the streaming chat and real-time HUD telemetry.

### 3. Run the Native Verification Suite
```bash
# Execute pure Rust verification suite
cargo test --lib
cargo test --test cli_standalone_test

# Execute python tests
pytest tests/
```

---

## ⚖️ License & Governance
Licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [LICENSE](LICENSE) for more details.

---
*GAJE-Flow Protocol v1.7.3-alpha (Silver Adult) — Toward Sovereign Edge Ultra-High-Density Inference.*
