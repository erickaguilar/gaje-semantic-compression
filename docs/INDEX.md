# 🗺️ Índice Maestro: GAJE Semantic Compression (v1.6.0-alpha)

Mapa central del repositorio consolidado. Define la estructura lógica actual, la documentación estratégica y la suite de validación.

---

## 📂 1. Documentación (`docs/`)

La documentación se organiza por función. La **investigación exploratoria** y las **versiones heredadas** se conservan íntegramente en `docs/archive/`.

### ⚖️ Gobernanza y Verdad Empírica (`docs/meta/`)
*   **[EMPIRICAL_TRUTH_STATE.md](docs/meta/EMPIRICAL_TRUTH_STATE.md)**: **DOCUMENTO CRÍTICO.** Estado real que contrasta código vs. resultados empíricos.
*   **[VALIDATION_PROTOCOLS.md](docs/meta/VALIDATION_PROTOCOLS.md)**: Protocolos de validación y gates de certificación.
*   **[CONSTITUTION_OF_BIRTH.md](docs/meta/vision/CONSTITUTION_OF_BIRTH.md)**: Principios éticos y técnicos del protocolo.
*   **[RESPONSIBLE_POLICY_AND_GOVERNANCE.md](docs/meta/RESPONSIBLE_POLICY_AND_GOVERNANCE.md)**: Posicionamiento de IA Constitucional Local.
*   **[FINDINGS_AND_DIAGNOSTICS_2026.md](docs/meta/FINDINGS_AND_DIAGNOSTICS_2026.md)**: Registro de éxitos y fracasos del motor.
*   **[LIFECYCLE_FLOW.md](docs/meta/LIFECYCLE_FLOW.md)**: Ciclo de vida del organismo y gates de certificación.
*   **[EMBRYO_10MB_DISTILLATION_STRATEGY.md](docs/meta/EMBRYO_10MB_DISTILLATION_STRATEGY.md)**: Hallazgo empírico (preservación de ranking) y propuesta de destilación para un embrión de ~10 MB.

### 🔬 Investigación (`docs/research/`)
*   **[QUANTUM_GENOMIC_TOKENIZATION_FINDINGS.md](docs/research/QUANTUM_GENOMIC_TOKENIZATION_FINDINGS.md)**: Fundamento matemático del isomorfismo cuántico-genómico (2-qubit Hilbert space a bases de ADN de 2-bits) y matrices de densidad $\rho$.
*   **[FROZEN_BODY_CAUSAL_AB_PROTOCOL.md](docs/research/FROZEN_BODY_CAUSAL_AB_PROTOCOL.md)**: Protocolo A/B causal del cuerpo congelado — aislamiento de causas sobre por qué los cuerpos cuantizados no deben recibir QAT post-cuantización.
*   **[CE_VS_GENERATION.md](docs/research/CE_VS_GENERATION.md)**: Hallazgo central de la Fase 4b — la CE media no correlaciona con la calidad de generación; la evaluación generativa es la métrica de éxito.
*   **[BODY_QAT_06B_PROTOCOL.md](docs/research/BODY_QAT_06B_PROTOCOL.md)**: Protocolo Técnica A vs B en Qwen2-0.5B — el QAT del cuerpo Q4_0 destruye la generación (100%/95% degeneradas vs 0% base); el punto dulce de SmolLM2 no transfiere.
*   **[temporal_4bit_fase1_test.py](docs/research/temporal_4bit_fase1_test.py)**: Fase 1 del plan de emulación temporal 2-bit→4-bit — refutación numérica: el Enfoque 1 almacena 4 bits (sin ahorro, doble latencia) y el Enfoque 2 (2-bit real) no recupera la precisión perdida.

### 📊 Resultados y Certificaciones Oficiales (`docs/reports/`)
*   **[BENCHMARK_OFFICIAL_v1_6.md](docs/reports/BENCHMARK_OFFICIAL_v1_6.md)**: **BENCHMARK CIENTÍFICO OFICIAL v1.6.0** — 125 evaluaciones en 5 modelos certificados (velocidad, recall semántico, degradación 0%, compresión 8.0x).
*   **[GTOK_CERTIFICATION_REPORT.md](docs/reports/GTOK_CERTIFICATION_REPORT.md)**: **CERTIFICACIÓN OFICIAL GTOK v1.0** — Formato binario nativo de tokenización zero-dependency, incrustable en cabecera `.flat`.
*   **[QUANTUM_CODEBOOK_BENCHMARK.md](docs/reports/QUANTUM_CODEBOOK_BENCHMARK.md)**: **BENCHMARK OFICIAL DE CODEBOOK CUÁNTICO (.qemb)** — 94.4% de reducción en tablas de embeddings de vocabulario masivo mediante superposición dispersa.
*   **[QUANTUM_GENOMIC_TOKENIZER_PROTOTYPE.md](docs/reports/QUANTUM_GENOMIC_TOKENIZER_PROTOTYPE.md)**: **INFORME DE PROTOTIPO CUÁNTICO-GENÓMICO** — Estados de superposición, pureza $\gamma=1.0$ y colapso contextual vía regla de Born.
*   **[BENCHMARKS.md](docs/reports/BENCHMARKS.md)**: Benchmark histórico de compresión, memoria, velocidad y PPL por modelo.
*   **[SCIENTIFIC_BENCHMARK_v097.md](docs/reports/SCIENTIFIC_BENCHMARK_v097.md)**: Benchmark científico de rendimiento y paridad.
*   **[session_findings_v1.6.0_phase_3.1.md](docs/reports/session_findings_v1.6.0_phase_3.1.md)**: Hallazgos de la fase 3.1 (formato `.flat`).
*   **[session_findings_v1.6.0_phase_3.2.md](docs/reports/session_findings_v1.6.0_phase_3.2.md)**: Hallazgos de la fase 3.2.
*   **[factual_audit_phase_3.3.md](docs/reports/factual_audit_phase_3.3.md)**: Auditoría factual de la fase 3.3.
*   **[smollm2_fp32_parity.md](docs/reports/smollm2_fp32_parity.md)**: Paridad SmolLM2 FP32.
*   **[qwen2_distillation_report.md](docs/reports/qwen2_distillation_report.md)**: Reporte de destilación Qwen2.

### 🛠️ Guías Operativas (`docs/guides/`)
*   **[AUTOMATION_SUITE_GUIDE.md](docs/guides/AUTOMATION_SUITE_GUIDE.md)**: Manual de ejecución de la suite de pruebas automatizada de regresión e integración continua.
*   **[GAJE_CLI_GUIDE.md](docs/guides/GAJE_CLI_GUIDE.md)**: **MANUAL PRINCIPAL.** Comandos y parámetros del motor nativo Rust.
*   **[ARCHITECTURE.md](docs/guides/ARCHITECTURE.md)**: Arquitectura del motor.
*   **[OPERATIONAL_WORKFLOWS_V1.1.0.md](docs/guides/OPERATIONAL_WORKFLOWS_V1.1.0.md)**: Flujos de trabajo del protocolo GAJE-Flow.
*   **[USER_GUIDE.md](docs/guides/USER_GUIDE.md)**: Manual de usuario del ecosistema.
*   **[BREEDING_AND_BORN_GUIDE.md](docs/guides/BREEDING_AND_BORN_GUIDE.md)**: Guía de crianza y generación.
*   **[GAJE_PUBLISHING_GUIDE.md](docs/guides/GAJE_PUBLISHING_GUIDE.md)**: Guía de publicación.
*   **[SEMANTIC_ENRICHMENT_GUIDE.md](docs/guides/SEMANTIC_ENRICHMENT_GUIDE.md)**: Enriquecimiento semántico.

### 🗺️ Planes y Roadmap (`docs/plans/`)
*   **[GPU_ACCELERATION_BACKEND_PLAN.md](docs/plans/GPU_ACCELERATION_BACKEND_PLAN.md)**: **PLAN DE BACKEND DE ACELERACIÓN GPU (Vulkan / WGPU)** — Offload masivo paralelo de capas lineales y SwiGLU para AMD Radeon / GPUs integradas y discretas.
*   **[QUANTUM_INFERENCE_LOOP_INTEGRATION_PLAN.md](docs/plans/QUANTUM_INFERENCE_LOOP_INTEGRATION_PLAN.md)**: **PLAN DE INTEGRACIÓN END-TO-END DEL LOOP CUÁNTICO** — Integración nativa de `.qemb` en `GenomicLLM` con lookup SIMD $< 0.1\ \mu\text{s}$ y reducción del 91.1% de RAM.
*   **[QUANTUM_META_TOKEN_CODEBOOK_PLAN.md](docs/plans/QUANTUM_META_TOKEN_CODEBOOK_PLAN.md)**: **PLAN DE CODEBOOK CUÁNTICO (8,192 Meta-Tokens)** — Compresión de tablas de embeddings del 94.4% mediante superposición dispersa.
*   **[WASM_BRAINSTEM_PLAN.md](docs/plans/WASM_BRAINSTEM_PLAN.md)**: GAJE-WASM — El motor como tronco encefálico (build WASM + API sensorio-motora).
*   **[MEMORY_EPOCHS_PLAN.md](docs/plans/MEMORY_EPOCHS_PLAN.md)**: Épocas de memoria — conocimiento flexible versionado sobre cuerpo congelado (`.gmem` v2).
*   **[ZERO_ORDER_NATIVE_TRAINING_PLAN.md](docs/plans/ZERO_ORDER_NATIVE_TRAINING_PLAN.md)**: Entrenamiento nativo de orden cero — SPSA discreto sobre centroides.
*   **[TESTING_AND_VERIFICATION_PLAN.md](docs/plans/TESTING_AND_VERIFICATION_PLAN.md)**: Plan maestro de pruebas, verificación de hardware, SSE y purga de memoria.
*   **[GAJE_BENCHMARK_SUITE_PLAN.md](docs/plans/GAJE_BENCHMARK_SUITE_PLAN.md)**: Especificación de la suite de benchmarks y dataset estandarizado de evaluación.
*   **[GTOK_BINARY_TOKENIZER_SPEC_PLAN.md](docs/plans/GTOK_BINARY_TOKENIZER_SPEC_PLAN.md)**: Especificación del formato binario nativo GTOK y roadmap de 4 fases.
*   **[MASTER_ROADMAP_2026.md](docs/plans/MASTER_ROADMAP_2026.md)**: Visión estratégica a largo plazo.
*   **[NEXT_STEPS_2026.md](docs/plans/NEXT_STEPS_2026.md)**: Próximos pasos operativos.
*   **[NATIVE_SEMANTIC_RAG_PLAN.md](docs/plans/NATIVE_SEMANTIC_RAG_PLAN.md)**: Plan de RAG semántico nativo.
*   **[OPERATION_REBIRTH.md](docs/plans/OPERATION_REBIRTH.md)**: Metodología de transmutación.
*   **[DOCUMENTATION_CONSOLIDATION_PLAN.md](docs/plans/DOCUMENTATION_CONSOLIDATION_PLAN.md)**: Sincronización documental.
*   **[Q2_0_SPATIAL_2BIT_EXPERIMENT.md](docs/plans/Q2_0_SPATIAL_2BIT_EXPERIMENT.md)**: Experimento completo de cuantización espacial por bloque a 2 bits/peso (Q2_0) vs Q4_0.
*   **[Q2_0_2BIT_SPATIAL_EXPERIMENT.md](docs/research/Q2_0_2BIT_SPATIAL_EXPERIMENT.md)**: **Veredicto del experimento Q2_0 (NEGATIVO)**.

### 🏷️ Registro y Nomenclatura (`docs/registry/`)
*   **[MODELS_NOMENCLATURE_AND_GAJE_CONVENTION.md](docs/registry/MODELS_NOMENCLATURE_AND_GAJE_CONVENTION.md)**: Regla estricta de nomenclatura (`.gaje` solo para organismos nacidos, `.flat` para modelos transmutados).
*   **[MODELS_REGISTRY_AND_REPRODUCTION_RECIPES.md](docs/registry/MODELS_REGISTRY_AND_REPRODUCTION_RECIPES.md)**: Recetas de reproducción y registro de modelos en producción.

### 🏛️ Especificaciones Técnicas (`docs/sdd/` y `docs/bdd/`)
*   **[docs/sdd/ARCHITECTURE_CORE.md](docs/sdd/ARCHITECTURE_CORE.md)**: Arquitectura del núcleo.
*   **[docs/sdd/QAT_IMPLEMENTATION_DETAILS.md](docs/sdd/QAT_IMPLEMENTATION_DETAILS.md)**: Detalles de QAT.
*   **[docs/sdd/MIXED_BIT_ARCHITECTURE.md](docs/sdd/MIXED_BIT_ARCHITECTURE.md)**: Arquitectura de bits mixtos.
*   **[docs/bdd/BORN_GENOMIC_FLOW.md](docs/bdd/BORN_GENOMIC_FLOW.md)**: Flujo genómico.
*   **[docs/certifications/CERTIFICATION_REPORT_V1.5.md](docs/certifications/CERTIFICATION_REPORT_V1.5.md)**: Reporte de certificación.
*   **[docs/certifications/QUANTUM_EMBEDDING_INFERENCE_CERTIFICATION.md](docs/certifications/QUANTUM_EMBEDDING_INFERENCE_CERTIFICATION.md)**: **CERTIFICACIÓN DE INFERENCIA CUÁNTICA (.qemb)** — Validación de compresión del 98.9%, descompresión SIMD y pruebas end-to-end.

---

## 🧪 2. Suite de Validación (`tests/`)
Estructura compatible con `pytest` para asegurar la integridad del motor.

*   **`tests/unit/`**: Pruebas de bajo nivel (kernels Rust vs NumPy, alineación RoPE, normas, redb, swiglu drift).
*   **`tests/integration/`**: Flujos completos de sistema (integración GajeIndex ADC + inferencia LLM, RAG nativo).
*   **`tests/metrics/`**: Evaluación de calidad (perplexity, precisión, coherencia, DNI interference, HNSW).
*   **`tests/training/`**: Convergencia de entrenamiento (IQAT, balanceo de pesos sinápticos).
*   **`tests/fixtures/`**: Datos de prueba compartidos (ej. `small_corpus.txt`).

---

## 🧬 3. Gestión de Modelos (`models/`)
*   **`models/production/`**: Modelos soberanos y estables (Qwen2 0.5B, SmolLM2 135M).
*   **`models/research/`**: Experimentos activos.
*   **`models/gguf/`**: Modelos maestros de referencia.
*   **`models/archive/`**: Histórico de versiones experimentales.
*   *Nota:* estos artefactos son binarios grandes y no se versionan en git (ver `.gitignore`).

---

## 🛠️ 4. Ecosistema de Herramientas (`scripts/`)
*   **`scripts/maintenance/`**: Respaldo, consolidación de datasets y limpieza.
*   **`scripts/data_processing/`**: Generación de datasets sintéticos y balanceo semántico.
*   **`scripts/benchmarks/`**: Suite de evaluación de rendimiento y recuperación (Needle Test).

## 📊 5. Benchmarks de Rendimiento (`benchmarks/`)
*   **`benchmarks/performance/bench_decode.py`**: Benchmark de decode por fase y tendencia KV-cache.
*   **`benchmarks/performance/gaje_flat_benchmark.py`**: Benchmark del formato `.flat`.
*   **`benchmarks/FINDINGS_WSL2_BENCHMARK_2026.md`**: Hallazgos de benchmarks WSL2.
*   **`benchmarks/research/`** y **`benchmarks/logs/`**: experimentos exploratorios y resultados serializados.

---

## 🧬 6. Código Nativo (`src/` y `python/`)
*   **`src/`**: Núcleo en Rust (kernels SIMD, LLM Engine, KV-Cache, Mmap Loader, Island Model). Solo se compila el CLI principal (`gaje-cli`) por defecto.
*   **`python/gaje/`**: Puente PyO3 y wrappers de inferencia nativos.

---

## 📦 7. Contenido Archivado (`legacy/` y `docs/archive/`)
*   **`legacy/archive/rust_bins/`**: Bins Rust exploratorios (trainers, breeders, MCTS) conservados como referencia.
*   **`legacy/archive/scripts/`** y **`legacy/archive/scratch/`**: Scripts de prueba transitorios.
*   **`docs/archive/research/`**: Notas de investigación exploratoria.
*   **`docs/archive/legacy_versions/`**, **`docs/archive/plans/`** y **`docs/archive/reports/`**: Documentación de etapas previas (incluye `PLAN_2BIT_ANCHORED_QUANTIZATION.md`, frente 2-bit congelado).

---
*Documento consolidado bajo el protocolo GAJE-Flow (Agosto 2026).*
