# 🗺️ Índice Maestro: GAJE Semantic Compression (v1.7.0-alpha)

Mapa central del repositorio consolidado. Define la estructura lógica actual, la documentación estratégica y la suite de validación.

---

## 📂 1. Documentación (`docs/`)

La documentación se organiza por función. La **investigación exploratoria** y las **versiones heredadas** se conservan íntegramente en `docs/archive/`.

### ⚖️ Gobernanza y Verdad Empírica (`docs/meta/`)
* **[EMPIRICAL_TRUTH_STATE.md](meta/EMPIRICAL_TRUTH_STATE.md)**: **DOCUMENTO CRÍTICO.** Estado real que contrasta código vs. resultados empíricos.
* **[VALIDATION_PROTOCOLS.md](meta/VALIDATION_PROTOCOLS.md)**: Protocolos de validación y gates de certificación.
* **[CONSTITUTION_OF_BIRTH.md](meta/vision/CONSTITUTION_OF_BIRTH.md)**: Principios éticos y técnicos del protocolo.
* **[RESPONSIBLE_POLICY_AND_GOVERNANCE.md](meta/RESPONSIBLE_POLICY_AND_GOVERNANCE.md)**: Posicionamiento de IA Constitucional Local.
* **[FINDINGS_AND_DIAGNOSTICS_2026.md](meta/FINDINGS_AND_DIAGNOSTICS_2026.md)**: Registro de éxitos y fracasos del motor.
* **[LIFECYCLE_FLOW.md](meta/LIFECYCLE_FLOW.md)**: Ciclo de vida del organismo y gates de certificación.
* **[EMBRYO_10MB_DISTILLATION_STRATEGY.md](meta/EMBRYO_10MB_DISTILLATION_STRATEGY.md)**: Hallazgo empírico (preservación de ranking) y propuesta de destilación para un embrión de ~10 MB.

### 🔬 Investigación (`docs/research/`)
* **[QUANTUM_GENOMIC_TOKENIZATION_FINDINGS.md](research/QUANTUM_GENOMIC_TOKENIZATION_FINDINGS.md)**: Fundamento matemático del isomorfismo cuántico-genómico (2-qubit Hilbert space a bases de ADN de 2-bits) y matrices de densidad $\rho$.
* **[FROZEN_BODY_CAUSAL_AB_PROTOCOL.md](research/FROZEN_BODY_CAUSAL_AB_PROTOCOL.md)**: Protocolo A/B causal del cuerpo congelado — aislamiento de causas sobre por qué los cuerpos cuantizados no deben recibir QAT post-cuantización.
* **[CE_VS_GENERATION.md](research/CE_VS_GENERATION.md)**: Hallazgo central de la Fase 4b — la CE media no correlaciona con la calidad de generación; la evaluación generativa es la métrica de éxito.
* **[BODY_QAT_06B_PROTOCOL.md](research/BODY_QAT_06B_PROTOCOL.md)**: Protocolo Técnica A vs B en Qwen2-0.5B — el QAT del cuerpo Q4_0 destruye la generación (100%/95% degeneradas vs 0% base); el punto dulce de SmolLM2 no transfiere.
* **[temporal_4bit_fase1_test.py](research/temporal_4bit_fase1_test.py)**: Fase 1 del plan de emulación temporal 2-bit→4-bit — refutación numérica: el Enfoque 1 almacena 4 bits (sin ahorro, doble latencia) y el Enfoque 2 (2-bit real) no recupera la precisión perdida.
* **[Q2_0_2BIT_SPATIAL_EXPERIMENT.md](research/Q2_0_2BIT_SPATIAL_EXPERIMENT.md)**: **Veredicto del experimento Q2_0 (NEGATIVO)**.

### 📊 Resultados y Certificaciones Oficiales (`docs/reports/` y `docs/certifications/`)
* **[GPU_ACCELERATION_CERTIFICATION.md](certifications/GPU_ACCELERATION_CERTIFICATION.md)**: **CERTIFICACIÓN OFICIAL DE ACELERACIÓN GPU (Vulkan / WGPU)** — Despacho masivo de capas tensoriales sobre AMD Radeon Vega con memoria unificada (UMA) y concordancia matemática exacta.
* **[QUANTUM_EMBEDDING_INFERENCE_CERTIFICATION.md](certifications/QUANTUM_EMBEDDING_INFERENCE_CERTIFICATION.md)**: **CERTIFICACIÓN OFICIAL DE INFERENCIA CUÁNTICA (.qemb)** — Cero regresión semántica y reducción del 91.1% en RAM.
* **[BENCHMARK_OFFICIAL_v1_6.md](reports/BENCHMARK_OFFICIAL_v1_6.md)**: **BENCHMARK CIENTÍFICO OFICIAL v1.6.0** — 125 evaluaciones en 5 modelos certificados (velocidad, recall semántico, degradación 0%, compresión 8.0x).
* **[GTOK_CERTIFICATION_REPORT.md](reports/GTOK_CERTIFICATION_REPORT.md)**: **CERTIFICACIÓN OFICIAL GTOK v1.0** — Formato binario nativo de tokenización zero-dependency, incrustable en cabecera `.flat`.
* **[QUANTUM_CODEBOOK_BENCHMARK.md](reports/QUANTUM_CODEBOOK_BENCHMARK.md)**: **BENCHMARK OFICIAL DE CODEBOOK CUÁNTICO (.qemb)** — 94.4% de reducción en tablas de embeddings de vocabulario masivo mediante superposición dispersa.
* **[QUANTUM_GENOMIC_TOKENIZER_PROTOTYPE.md](reports/QUANTUM_GENOMIC_TOKENIZER_PROTOTYPE.md)**: **INFORME DE PROTOTIPO CUÁNTICO-GENÓMICO** — Estados de superposición, pureza $\gamma=1.0$ y colapso contextual vía regla de Born.
* **[ECOSYSTEM_INTEROPERABILITY_AND_ARENA_FINDINGS.md](reports/ECOSYSTEM_INTEROPERABILITY_AND_ARENA_FINDINGS.md)**: **HALLAZGOS DE ARENA E INTEROPERABILIDAD** — Comparativa con Ollama, llama.cpp y vLLM.
* **[GRADIENT_DYNAMICS_AND_FLOW_MODULATION.md](reports/GRADIENT_DYNAMICS_AND_FLOW_MODULATION.md)**: Dinámica de gradientes en flujos de compresión.
* **[HNSW_STEP2_SPIKE_FINDINGS.md](reports/HNSW_STEP2_SPIKE_FINDINGS.md)**: Hallazgos de indexación vectorial aproximada con HNSW.
* **[BENCHMARKS.md](reports/BENCHMARKS.md)**: Benchmark histórico de compresión, memoria, velocidad y PPL por modelo.
* **[SCIENTIFIC_BENCHMARK_v097.md](reports/SCIENTIFIC_BENCHMARK_v097.md)**: Benchmark científico de rendimiento y paridad.
* **[findings_v1.6.0_phase_0_to_3.md](reports/findings_v1.6.0_phase_0_to_3.md)**: Reporte consolidado de hallazgos de las fases 0 a 3.
* **[session_findings_v1.6.0_phase_3.1.md](reports/session_findings_v1.6.0_phase_3.1.md)**: Hallazgos de la fase 3.1 (formato `.flat`).
* **[session_findings_v1.6.0_phase_3.2.md](reports/session_findings_v1.6.0_phase_3.2.md)**: Hallazgos de la fase 3.2.
* **[factual_audit_phase_3.3.md](reports/factual_audit_phase_3.3.md)**: Auditoría factual de la fase 3.3.
* **[smollm2_fp32_parity.md](reports/smollm2_fp32_parity.md)**: Paridad SmolLM2 FP32.
* **[qwen2_distillation_report.md](reports/qwen2_distillation_report.md)**: Reporte de destilación Qwen2.
* **[SPEED_AND_ARCHITECTURE_BREAKTHROUGH_2026.md](reports/SPEED_AND_ARCHITECTURE_BREAKTHROUGH_2026.md)**: Hitos de velocidad y arquitectura 2026.
* **[STEEL_SOUL_MVP_CREATION.md](reports/STEEL_SOUL_MVP_CREATION.md)**: Registro de creación del MVP Steel Soul.

### 🛠️ Guías Operativas (`docs/guides/`)
* **[GAJE_SERVE_DEPLOYMENT_GUIDE.md](guides/GAJE_SERVE_DEPLOYMENT_GUIDE.md)**: **DESPLIEGUE EN PRODUCCIÓN (7B ULTRA).** Guía oficial para Hugging Face Spaces (Docker), VPS Cloud y Túneles Cloudflare.
* **[FAST_MODEL_DOWNLOAD_AND_DNF_TECHNIQUES.md](guides/FAST_MODEL_DOWNLOAD_AND_DNF_TECHNIQUES.md)**: **TÉCNICAS DNF Y MULTI-STREAM.** Protocolo de aceleración de red estilo Fedora librepo / hf_transfer.
* **[FAST_MODEL_DOWNLOAD_GUIDE.md](guides/FAST_MODEL_DOWNLOAD_GUIDE.md)**: Guía práctica de descarga acelerada de modelos desde Hugging Face Hub.
* **[HYBRID_DEPLOYMENT_AND_MODEL_DISTRIBUTION.md](guides/HYBRID_DEPLOYMENT_AND_MODEL_DISTRIBUTION.md)**: Estrategia de despliegue híbrido dual (Edge WASM + Cloud Service) y distribución.
* **[GAJE_CLI_GUIDE.md](guides/GAJE_CLI_GUIDE.md)**: **MANUAL PRINCIPAL.** Comandos y parámetros del motor nativo Rust.
* **[GAJE_CLI_CAPABILITIES_AND_LIMITS.md](guides/GAJE_CLI_CAPABILITIES_AND_LIMITS.md)**: Capacidades operativas y límites del CLI nativo.
* **[GAJE_MODEL_CAPABILITIES_AND_LIMITS.md](guides/GAJE_MODEL_CAPABILITIES_AND_LIMITS.md)**: Matriz de capacidades por familia de modelos genómicos.
* **[AUTOMATION_SUITE_GUIDE.md](guides/AUTOMATION_SUITE_GUIDE.md)**: Manual de ejecución de la suite de pruebas automatizada de regresión e integración continua.
* **[ARCHITECTURE.md](guides/ARCHITECTURE.md)**: Arquitectura global del motor.
* **[OPERATIONAL_WORKFLOWS_V1.1.0.md](guides/OPERATIONAL_WORKFLOWS_V1.1.0.md)**: Flujos de trabajo del protocolo GAJE-Flow.
* **[UNIX_TIME_ARCHITECTURE.md](guides/UNIX_TIME_ARCHITECTURE.md)**: Estándar y arquitectura de marcas de tiempo Unix POSIX.
* **[USER_GUIDE.md](guides/USER_GUIDE.md)**: Manual de usuario del ecosistema.
* **[BREEDING_AND_BORN_GUIDE.md](guides/BREEDING_AND_BORN_GUIDE.md)**: Guía de crianza y generación.
* **[GAJE_PUBLISHING_GUIDE.md](guides/GAJE_PUBLISHING_GUIDE.md)**: Guía de publicación.
* **[SEMANTIC_ENRICHMENT_GUIDE.md](guides/SEMANTIC_ENRICHMENT_GUIDE.md)**: Enriquecimiento semántico.

### 🗺️ Planes Activos y Roadmap Estratégico (`docs/plans/`)
* **[MASTER_ROADMAP_2026.md](plans/MASTER_ROADMAP_2026.md)**: Visión estratégica a largo plazo y arquitectura del motor.
* **[PHASE_4_AGENTIC_GRAPH_EXECUTION_PLAN.md](plans/PHASE_4_AGENTIC_GRAPH_EXECUTION_PLAN.md)**: Plan maestro de ejecución de grafos agénticos soberanos en Rust.
* **[AGENTIC_GRAPH_RUST.md](plans/AGENTIC_GRAPH_RUST.md)**: Tipos `AgentState`, `AgentNode` y orquestación cíclica sin dependencias externas.
* **[DISTILLATION_DEEPSEEK_GEMMA_STRATEGY.md](plans/DISTILLATION_DEEPSEEK_GEMMA_STRATEGY.md)**: Estrategia de destilación y transferencia de conocimiento desde maestros Qwen2.5 / DeepSeek / Gemma.
* **[NATIVE_SEMANTIC_RAG_PLAN.md](plans/NATIVE_SEMANTIC_RAG_PLAN.md)**: Plan de RAG semántico nativo e indexación HNSW zero-copy.
* **[GAJE_32MB_PLAN.md](plans/GAJE_32MB_PLAN.md)**: Plan de compresión extrema para modelos ultra-ligeros de 32 MB.
* **[STRATEGIC_OPPORTUNITIES_AND_NEXT_STEPS.md](plans/STRATEGIC_OPPORTUNITIES_AND_NEXT_STEPS.md)**: Oportunidades estratégicas de alto rendimiento (WebGPU, sub-4bit y ecosistema).
* **[OPPORTUNITIES_FROM_CACTUS_NEEDLE.md](plans/OPPORTUNITIES_FROM_CACTUS_NEEDLE.md)**: Lecciones y oportunidades arquitectónicas aprendidas de Cactus Needle.
* **[GAJE_BENCHMARK_SUITE_PLAN.md](plans/GAJE_BENCHMARK_SUITE_PLAN.md)**: Especificación de la suite continua de benchmarks.
* **[QUALITY_EVAL_PROTOCOL.md](plans/QUALITY_EVAL_PROTOCOL.md)**: Protocolo de evaluación de calidad semántica y perplejidad.
* **[QUALITY_EXPORT_PLAN.md](plans/QUALITY_EXPORT_PLAN.md)**: Plan de exportación calibrada con preservación de gradientes.
* **[NEXT_STEPS_2026.md](plans/NEXT_STEPS_2026.md)**: Próximos pasos operativos del ciclo de desarrollo.

### ✅ Planes Implementados y Certificados (`docs/plans/completed/`)
* **[GAJE_CLI_IMPROVEMENTS_PLAN.md](plans/completed/GAJE_CLI_IMPROVEMENTS_PLAN.md)**: `clap` v4, REPL interactivo y herramienta `gaje-cli doctor`.
* **[BOTTLENECK_OPTIMIZATION_DNF_TECHNIQUES_PLAN.md](plans/completed/BOTTLENECK_OPTIMIZATION_DNF_TECHNIQUES_PLAN.md)**: Multi-Stream WASM, exportación tensorial paralela en Rust y zero-copy mmap.
* **[GAJE_CLI_SERVE_NATIVE_PLAN.md](plans/completed/GAJE_CLI_SERVE_NATIVE_PLAN.md)**: Servidor HTTP nativo en Rust con SSE y compatibilidad API OpenAI.
* **[RUST_SINGLE_BINARY_PLAN.md](plans/completed/RUST_SINGLE_BINARY_PLAN.md)**: Binario único autocontenido con Web UI incrustada vía `rust-embed`.
* **[DEEPSEEK_GEMMA_SUPPORT_PLAN.md](plans/completed/DEEPSEEK_GEMMA_SUPPORT_PLAN.md)**: Soporte nativo para DeepSeek (MLA/MoE), Gemma y Qwen2.5.
* **[GGUF_WRITER_PLAN.md](plans/completed/GGUF_WRITER_PLAN.md)**: Serializador binario GGUF v3 nativo para compatibilidad con Ollama y `llama.cpp`.
* **[MEMORY_EPOCHS_PLAN.md](plans/completed/MEMORY_EPOCHS_PLAN.md)**: Épocas `.gmem` v2, rollback determinista bit a bit (0.10 ms) y snapshots.
* **[GPU_ACCELERATION_BACKEND_PLAN.md](plans/completed/GPU_ACCELERATION_BACKEND_PLAN.md)**: Backend GPU Vulkan / WGPU, shaders WGSL, layer offloading y telemetría en UI.
* **[QUANTUM_INFERENCE_LOOP_INTEGRATION_PLAN.md](plans/completed/QUANTUM_INFERENCE_LOOP_INTEGRATION_PLAN.md)**: Loop de inferencia de embeddings cuánticos `.qemb` (>98% ahorro RAM).
* **[QUANTUM_META_TOKEN_CODEBOOK_PLAN.md](plans/completed/QUANTUM_META_TOKEN_CODEBOOK_PLAN.md)**: Codebook cuántico de 8,192 meta-tokens con amplitudes en esfera unitaria.
* **[ZERO_ORDER_NATIVE_TRAINING_PLAN.md](plans/completed/ZERO_ORDER_NATIVE_TRAINING_PLAN.md)**: Entrenamiento nativo SPSA discreto sobre centroides (speedup 2.24× y nichos 21.56×).
* **[GTOK_BINARY_TOKENIZER_SPEC_PLAN.md](plans/completed/GTOK_BINARY_TOKENIZER_SPEC_PLAN.md)**: Formato binario nativo GTOK incrustable en `.flat`.
* **[WASM_BRAINSTEM_PLAN.md](plans/completed/WASM_BRAINSTEM_PLAN.md)**: Motor WASM client-side como tronco encefálico soberano offline.
* **[WEB_UI_IMPROVEMENT_PLAN.md](plans/completed/WEB_UI_IMPROVEMENT_PLAN.md)**: Plataforma Web UI dual Y2K/Zen, telemetría HUD, streaming SSE y OPFS cache.
* **[TESTING_AND_VERIFICATION_PLAN.md](plans/completed/TESTING_AND_VERIFICATION_PLAN.md)**: Suite de pruebas de regresión, certificación de hardware y purga de memoria.
* **[DOCUMENTATION_CONSOLIDATION_PLAN.md](plans/completed/DOCUMENTATION_CONSOLIDATION_PLAN.md)**: Consolidación y sincronización documental global.

### 🏷️ Registro y Nomenclatura (`docs/registry/`)
* **[MODELS_NOMENCLATURE_AND_GAJE_CONVENTION.md](registry/MODELS_NOMENCLATURE_AND_GAJE_CONVENTION.md)**: Regla estricta de nomenclatura (`.gaje` solo para organismos nacidos, `.flat` para modelos transmutados).
* **[MODELS_REGISTRY_AND_REPRODUCTION_RECIPES.md](registry/MODELS_REGISTRY_AND_REPRODUCTION_RECIPES.md)**: Recetas de reproducción y registro de modelos en producción.

### 🏛️ Especificaciones Técnicas (`docs/sdd/` y `docs/bdd/`)
* **[ARCHITECTURE_CORE.md](sdd/ARCHITECTURE_CORE.md)**: Arquitectura del núcleo.
* **[QAT_IMPLEMENTATION_DETAILS.md](sdd/QAT_IMPLEMENTATION_DETAILS.md)**: Detalles de QAT.
* **[MIXED_BIT_ARCHITECTURE.md](sdd/MIXED_BIT_ARCHITECTURE.md)**: Arquitectura de bits mixtos.
* **[BORN_GENOMIC_FLOW.md](bdd/BORN_GENOMIC_FLOW.md)**: Flujo genómico.
* **[CERTIFICATION_REPORT_V1.5.md](certifications/CERTIFICATION_REPORT_V1.5.md)**: Reporte de certificación.
* **[QUANTUM_EMBEDDING_INFERENCE_CERTIFICATION.md](certifications/QUANTUM_EMBEDDING_INFERENCE_CERTIFICATION.md)**: **CERTIFICACIÓN DE INFERENCIA CUÁNTICA (.qemb)** — Validación de compresión del 98.9%, descompresión SIMD y pruebas end-to-end.

---

## 🧪 2. Suite de Validación (`tests/`)
Estructura compatible con `pytest` y `cargo test` para asegurar la integridad del motor.

* **`tests/unit/`**: Pruebas de bajo nivel (kernels Rust vs NumPy, alineación RoPE, normas, redb, swiglu drift).
* **`tests/integration/`**: Flujos completos de sistema (integración GajeIndex ADC + inferencia LLM, RAG nativo, épocas de memoria).
* **`tests/metrics/`**: Evaluación de calidad (perplexity, precisión, coherencia, DNI interference, HNSW).
* **`tests/training/`**: Convergencia de entrenamiento (IQAT, balanceo de pesos sinápticos).
* **`tests/ui_e2e/`**: Pruebas end-to-end de interfaz web con Playwright.
* **`tests/fixtures/`**: Datos de prueba compartidos (ej. `small_corpus.txt`).

---

## 🧬 3. Gestión de Modelos (`models/`)
* **`models/production/`**: Modelos soberanos y estables (Qwen2 0.5B, SmolLM2 135M).
* **`models/research/`**: Experimentos activos.
* **`models/gguf/`**: Modelos maestros de referencia.
* **`models/archive/`**: Histórico de versiones experimentales.
* *Nota:* estos artefactos son binarios grandes y no se versionan en git (ver `.gitignore`).

---

## 🛠️ 4. Ecosistema de Herramientas (`scripts/`)
* **`scripts/maintenance/`**: Respaldo, escaneo de modelos y sincronización con Hugging Face.
* **`scripts/data_processing/`**: Generación de datasets sintéticos y balanceo semántico.
* **`scripts/benchmarks/`**: Suite de evaluación de rendimiento y recuperación (Needle Test).
* **`scripts/export/`**: Transmutación y empaquetado de modelos binarios planos `.flat` y `.qemb`.
* **`scripts/debug/`**: Trazas tensoriales capa por capa y auditoría de paridad.
* **`scripts/training/`**: Destilación genómica y entrenamiento de centroides.

---

## 📊 5. Benchmarks de Rendimiento (`benchmarks/`)
* **`benchmarks/performance/bench_decode.py`**: Benchmark de decode por fase y tendencia KV-cache.
* **`benchmarks/performance/gaje_flat_benchmark.py`**: Benchmark del formato `.flat`.
* **`benchmarks/FINDINGS_WSL2_BENCHMARK_2026.md`**: Hallazgos de benchmarks WSL2.
* **`benchmarks/research/`** y **`benchmarks/logs/`**: experimentos exploratorios y resultados serializados.

---

## 🧬 6. Código Nativo (`src/` y `python/`)
* **`src/`**: Núcleo en Rust (kernels SIMD, LLM Engine, KV-Cache, Mmap Loader, Downloader multi-stream, Island Model). Solo se compila el CLI principal (`gaje-cli`) por defecto.
* **`python/gaje/`**: Puente PyO3 y wrappers de inferencia nativos.

---

## 📦 7. Contenido Archivado (`legacy/` y `docs/archive/`)
* **`legacy/archive/rust_bins/`**: Bins Rust exploratorios (trainers, breeders, MCTS) conservados como referencia.
* **`legacy/archive/scripts/`** y **`legacy/archive/scratch/`**: Scripts de prueba transitorios.
* **`docs/archive/research/`**: Notas de investigación exploratoria.
* **`docs/archive/legacy_versions/`**, **`docs/archive/plans/`** y **`docs/archive/reports/`**: Documentación de etapas previas (incluye `PLAN_2BIT_ANCHORED_QUANTIZATION.md`, frente 2-bit congelado).

---
*Documento consolidado bajo el protocolo GAJE-Flow (Agosto 2026).*
