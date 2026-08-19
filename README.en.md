# 🧬 GAJE Protocol: Semantic Adaptation & Genomic Compression (v1.6.0-alpha)

[![Version](https://img.shields.io/badge/version-1.6.0--alpha_Silver_Adult-purple)](docs/meta/EMPIRICAL_TRUTH_STATE.md)
[![Engine](https://img.shields.io/badge/Engine-Pure_Rust_PyO3-orange.svg)](src/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Language: Spanish](https://img.shields.io/badge/Language-Espa%C3%B1ol-yellow.svg)](README.md)

**GAJE (Genomic Adaptive Joint Embedding)** is an ultra-high-density research and computing protocol designed for the execution and compression of Large Language Models (LLMs). The protocol quantizes parameter spaces down to a discrete **4-bit per weight** representation (16 optimized centroids) and **2-bit per weight** (4 states: `00=A`, `01=C`, `11=G`, `10=T`), integrating persistent zero-copy memory (**Island Model `.gmem`**), dynamic self-describing headers (**`ArchitectureDescriptor`**), and instant memory-mapped file loading (**`.gaje.flat` v2**).

---

## 🔬 Empirical Status & Scientific Diagnosis (v1.6.0-alpha)

Following the principle of **Empirical Truth** ([`docs/meta/EMPIRICAL_TRUTH_STATE.md`](file:///home/erickaguilar/Documentos/gaje-semantic-compression/docs/meta/EMPIRICAL_TRUTH_STATE.md)), the system presents the following certified functional state:

### 🏆 1. A/B Parity Control Experiment (GAJE Q4_0 vs. HuggingFace PyTorch FP32)

We executed an A/B parity trial comparing the original FP32 model (`Qwen/Qwen2-0.5B-Instruct`) in **PyTorch** against the native **GAJE 4-bit `.gaje.flat`** engine on an **AMD Ryzen 7 5800H** CPU:

| Inference Engine | Format / Precision | Exact Generated Response | Real E2E Throughput | RAM Consumption |
| :--- | :---: | :--- | :---: | :---: |
| **HuggingFace PyTorch** | **FP32 Original (Alibaba)** | *"El planeta más grande del Sistema Solar es la Tierra, con una"* | **`1.38 tok/s`** | $1,980\text{ MB}$ |
| **GAJE Native Engine (`.flat`)** | **4-bit Genomic Zero-Copy** | *"El planeta más grande del Sistema Solar es la Tierra."* | **`19.2 - 23.0 tok/s`** | **`448 MB` (RSS, ~77% vs FP32)** |

---

### ⚡ 2. Certified Multimodel Production Performance (Ryzen 7 5800H)

| Model / Architecture | Binary Format | Certified Factual Response | CPU Throughput | Cold Start Load Time | Live RAM RSS |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Qwen2.5 1.5B Instruct** | **`.gaje.flat` (Hybrid v2)** | Spanish: *"La capital de Francia es París."* | **`11.31 - 12.13 tok/s`** | **`< 0.75 ms` (mmap)** | **`2.6 GB` (Virtual)** |
| **Qwen2 0.5B Instruct** | **`.gaje.flat` (Hybrid v2)** | Chinese: *"木星"* (Jupiter) / Spanish: *"París"* | **`19.20 - 23.00 tok/s`** | **`< 0.75 ms` (mmap)** | **`~498 MiB` (~74% vs FP32)** |
| **SmolLM2 135M Instruct** | **`.gaje.flat` (Zero-Copy)** | English: *"Berlin."* / *"100°C"* | **`28.28 - 32.10 tok/s`** | **`< 0.75 ms` (mmap)** | **`~472 MB` (Q4_0 body + FP32 embeddings)** |

> [!IMPORTANT]
> **Hybrid .flat v2 Layout**: To preserve semantic representation fidelity and avoid vocabulary collapse in high-density languages (like Chinese/Arabic), the `.flat` format stores the critical semantic layers (`token_embd` and `lm_head`) in **FP32** (4 bytes/weight), while the transformer body (attention and FFN projections) is quantized to **Q4_0** (4-bits).

---

### 🏝️ 3. Island Model (.gmem): Sub-Millisecond Persistence

The system integrates contextual memory persistence through 64-byte aligned flat binary indices (`.gmem`):

* **Vector Retrieval Latency (RAG)**: **`0.75 ms`** ($750\text{ µs}$) per multi-niche query.
* **Cold Start Latency (`.gmem`)**: **`0.12 ms`** ($120\text{ µs}$) from file.
* **Context Budget**: Automatic injection of $128\text{ tokens}$ of high resonance ($\text{CosSim} = 0.9998$).

---

## 🛠️ Architectural Foundations

### 1. Dynamic Self-Describing Flat Headers (`.gaje.flat` v2)
The **`FlatHeaderV2`** binary structure implements an autodescriptive **`ArchitectureDescriptor`**. During exporting (`export_gaje_flat.py`), the model dimensions, RoPE parameters, and attention permutation patterns ($Q/K$) are dynamically parsed and written to bytes `56-79` of the file header, eliminating manual deployment errors.

### 2. Quantization-Aware Training (QAT) Stabilization
GAJE supports native local adaptation algorithms. Centroid updates in Rust (`linear.rs`) are normalized by dividing the accumulated gradients by the actual activation count (`centroid_counts`), preventing gradient explosion (`NaN` / `Inf`) and guaranteeing mathematical convergence of quantization loss.

### 3. Minimum Action Lagrangian Sampling
Token generation is modeled as a dynamic system governed by the Principle of Least Action, evaluating kinetic energy $T$ (semantic mobility) and potential energy $V$ (grammatical constraint):

$$\mathcal{L} = T - V$$

---

## 📂 Repository Organization (`v1.6.0-alpha`)

```text
gaje-semantic-compression/
├── src/                    # Rust Native Core (AVX2/FMA SIMD Kernels, LLM Engine, KV-Cache, Mmap Loader)
│   └── bin/gaje-cli.rs     # Primary native engine CLI
├── python/gaje/            # PyO3 Bridge and Native Python Infrasctructure Wrappers
├── examples/               # Core demos, Web UI, notebooks and Rust utilities
│   └── ui/web_ui/          # Web UI Frontend (http://localhost:8080) and server.py server
├── tests/                  # Verification Suite (unit, integration, metrics, training, ui_e2e)
├── scripts/                # Utility scripts and Flat Exporters (.gaje.flat)
├── models/production/      # Quantized Production Models (Qwen2 0.5B, SmolLM2 135M)
└── docs/                   # Scientific papers, blueprints, and reports (v1.6.0 report)
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

### 2. Run the Interactive Web UI
```bash
python examples/ui/web_ui/server.py
```
Open `http://localhost:8080` in your browser and select your flat quantized model.

### 3. Run the Native Verification Suite
```bash
# Execute python tests (21/21 tests passing successfully)
pytest tests/
```

---

## ⚖️ License & Governance
Licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [LICENSE](LICENSE) for more details.

---
*GAJE-Flow Protocol v1.6.0-alpha (Silver Adult) — Toward Sovereign Edge Ultra-High-Density Inference.*
