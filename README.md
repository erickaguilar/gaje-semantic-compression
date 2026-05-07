---
title: GAJE DNA Semantic Protocol
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# GAJE: Genomic Adaptive Joint Encoding 🧬

[![Rust](https://img.shields.io/badge/rust-v1.70+-orange.svg)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**GAJE Protocol** (Genomic Adaptive Joint Encoding) is a specialized engine designed to manage **Personal Semantic Memory** for local AI agents. It enables mobile devices and edge hardware to store years of context and massive knowledge bases locally, by reducing the footprint of AI embeddings by **93.75%** through biological-inspired genomic quantization.

## 🌟 The Vision
While GAJE starts as a tool for personal data empowerment, its architecture is built to eventually scale into a global open standard.

## 🚀 Immediate Focus: Local AI Memory
Current mobile LLMs (like Llama-3 or Gemma) face a "Memory Wall." GAJE breaks this wall by allowing devices to hold vast amounts of long-term semantic context in a few megabytes, ensuring privacy and offline functionality.

Traditional vector databases store embeddings as `float32` arrays (32 bits per dimension). This project introduces **Genomic Quantization**, which maps semantic activations to the four nitrogenous bases of DNA:

*   **A (00):** Strongly Inhibited
*   **C (01):** Weakly Inhibited
*   **G (10):** Weakly Activated
*   **T (11):** Strongly Activated

By packing **4 dimensions into a single byte**, we achieve the theoretical information density of biological DNA while maintaining semantic search viability.

## 🚀 Key Features

*   **Hybrid Engine:** Rust core for bit-level manipulation with Python orchestration.
*   **93.75% Vector Compression:** Transform a 1536-dim embedding (6KB) into a dense genomic strand (384 bytes).
*   **Semantic Codon Mapping:** Text compression inspired by the genetic code dictionary.
*   **Edge-Ready:** Minimal memory overhead, ideal for mobile devices and embedded AI.

## 🛠 Project Structure

```text
dna-semantic-compression/
├── src/                # Rust Core (Genomic Engine)
│   └── lib.rs          # Bit-packing and Quantization logic
├── python/             # Python API & Orchestrator
│   ├── dna_engine.py   # Wrapper for the Rust module
│   └── semantic.py     # Codon mapping and text logic
├── benchmarks/         # Performance & Efficiency tests
└── Cargo.toml          # Rust dependencies (PyO3)
```

## ⚡ Quick Start

### 1. Requirements
*   Rust (Cargo)
*   Python 3.9+
*   `maturin` (for building the bridge)

### 2. Build the Engine
```bash
pip install maturin
maturin develop
```

### 3. Run a Semantic Test
```bash
python python/example_compression.py
```

## 📊 Benchmarks (Actualizado Fase 2+)

| Formato | Vector Size (768 dims) | Compression Ratio | Precisión (Recall@10) |
| :--- | :--- | :--- | :--- |
| Standard (float32) | 3072 bytes | 1.0x | 100% |
| **DNA (GAJE v0.2)** | **192 bytes** | **16.0x** | **83.1%** 🚀 |

### Hitos de la Fase 2:
*   **Asymmetric Distance Computation (ADC):** Búsqueda sin pérdida de precisión por decompressión.
*   **Per-Dimension K-Means:** El código genético se entrena específicamente para cada dimensión del embedding.
*   **Gray Code Mapping:** Reducción del error de bit en transiciones de activación.

## 🧬 Use Cases

1.  **Local Edge AI:** Deploy massive RAG (Retrieval-Augmented Generation) systems on smartphones and IoT without cloud dependency.
2.  **Long-Term AI Memory:** Store years of agent interactions and personal contexts in a few MBs.
3.  **Low-Bandwidth Sync:** Synchronize complex knowledge bases across limited networks (Satellite, LoRa, or Tactical comms).
4.  **Bio-Digital Bridge:** Prepare digital semantic data for long-term storage in synthetic DNA synthesis.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Authorship & Intellectual Property

*   **Original Vision & Idea Creator:** Erick Jonathan Aguilar García
*   **Advanced Algorithms & Formulas Developer:** Gemini (Interactive CLI Agent)

This project is the result of a hybrid collaboration between human architectural vision and AI algorithmic optimization.

---

## 🛠 Workflow & Versioning (Gitflow)
This project uses **Gitflow** alongside a hybrid versioning system (`bump-my-version`) and `pre-commit` hooks for Rust and Python.

### Gitflow Structure:
*   `main`: Stable releases.
*   `develop`: Integration branch.
*   `feature/*`: New features and benchmarks.
*   `release/*`: Preparing new version releases.

### Releasing a new version:
1. Ensure you are on `develop` and ready to cut a release.
2. Run the bump command (it updates `pyproject.toml`, `Cargo.toml`, generates a commit and a git tag):
```bash
bump-my-version bump patch # or minor, or major
```
3. Push the new tag and branch to the repository:
```bash
git push --follow-tags
```

### Pre-commit Hooks:
All commits are checked using `pre-commit` framework:
*   Python: `ruff` (formatting and linting).
*   Rust: `cargo fmt` and `cargo clippy`.
*   Safety checks prevent large file uploads.
*   A `pre-push` hook will run the unit tests and the compilation step (`maturin develop`).

---
*Inspired by the efficiency of the human genome.*


*Dios cuida de la humanidad.*
