# 🧬 Scripts & Workflow Directory — GAJE Semantic Compression

> **IMPORTANTE — SOBERANÍA DEL NÚCLEO NATIVO (Rust Single Binary):**  
> A partir de la versión **GAJE Helix 1.7.0**, todos los flujos de trabajo de producción, descarga, inferencia, exportación, benchmarks, construcción de datasets y auditoría estructural han sido migrados a comandos nativos de alto rendimiento dentro de **`gaje-cli` (Rust)**.

---

## 🗺️ Matriz de Equivalencias (Python $\rightarrow$ `gaje-cli`)

| Flujo de Trabajo | Script Histórico (Legacy Python) | Comando Nativo Soberano (`gaje-cli`) |
|---|---|---|
| **Chat Interactivo (CLI)** | `python/examples/chat_stream.py` | `gaje-cli chat` |
| **Inferencia Unitaria** | `python/examples/inference_oneshot.py` | `gaje-cli --model <m> --prompt "<p>"` |
| **Servidor Web & Chat UI** | `examples/ui/web_ui/server.py` | `gaje-cli serve --port 8080` |
| **Descarga de Modelos** | `scripts/maintenance/download_model.py` | `gaje-cli pull pico|nano|prime|ultra` |
| **Descarga Hugging Face** | `scripts/setup/download_hf_model.py` | `gaje-cli pull <repo/archivo>` |
| **Catálogo de Modelos** | `scripts/maintenance/scan_models.py` | `gaje-cli models list` |
| **Inspección Estructural** | `scripts/maintenance/native_sovereignty_cert.py` | `gaje-cli models inspect <model.flat>` |
| **Verificación de Integridad**| `scripts/maintenance/scan_models.py` | `gaje-cli models verify <model.flat>` |
| **Exportación Zero-Copy** | `scripts/export/export_gaje_flat.py` | `gaje-cli export-flat <input> -o <out.flat>` |
| **Exportación Qwen / Smol**| `scripts/export/export_smollm2_flat.py` | `gaje-cli export-flat <input> -o <out.flat>` |
| **Benchmark & Perplejidad**| `scripts/benchmarks/engine_benchmark.py` | `gaje-cli benchmark --model <m> [--corpus <c>]` |
| **Construcción de Datasets**| `scripts/data_processing/create_hybrid_dataset.py` | `gaje-cli dataset-build <inputs...> -o <out.jsonl>` |
| **Auditoría y Chequeo NaN**| `scripts/archive/monte_carlo/coherence_test_mc.py`| `gaje-cli audit <model.flat>` |
| **Diagnóstico de Hardware** | `scripts/maintenance/check_simd.py` | `gaje-cli doctor` |
| **Épocas de Memoria .gmem** | `scripts/maintenance/flash_resonance.py` | `gaje-cli epoch --action list|rollback` |

---

## 🔬 Scripts de Investigación y Prototipado en Python (`scripts/research/`, `scripts/training/`)

Los scripts en Python que utilizan `_impl` / PyO3 para investigación exploratoria (como Monte Carlo Tree Search, algoritmos genéticos experimentales con PyTorch o visualización gráfica) se mantienen en `scripts/` como herramientas de laboratorio, pero **no forman parte de la cadena crítica de producción**.
