# 🗺️ Índice Maestro: GAJE Semantic Compression (v1.6.0-alpha)

Mapa central del repositorio consolidado. Define la estructura lógica actual, la documentación estratégica y la suite de validación.

---

## 📂 1. Documentación (`docs/`)

La documentación se organiza por función. La **investigación exploratoria** y las **versiones heredadas** se conservan íntegramente en `docs/archive/`.

### ⚖️ Gobernanza y Verdad Empírica (`docs/meta/`)
*   **[EMPIRICAL_TRUTH_STATE.md](docs/meta/EMPIRICAL_TRUTH_STATE.md)**: **DOCUMENTO CRÍTICO.** Estado real que contrasta código vs. resultados empíricos.
*   **[VALIDATION_PROTOCOLS.md](docs/meta/VALIDATION_PROTOCOLS.md)**: Protocolos de validación y gates de certificación.
*   **[CONSTITUTION_OF_BIRTH.md](docs/meta/CONSTITUTION_OF_BIRTH.md)**: Principios éticos y técnicos del protocolo.
*   **[RESPONSIBLE_POLICY_AND_GOVERNANCE.md](docs/meta/RESPONSIBLE_POLICY_AND_GOVERNANCE.md)**: Posicionamiento de IA Constitucional Local.
*   **[FINDINGS_AND_DIAGNOSTICS_2026.md](docs/meta/FINDINGS_AND_DIAGNOSTICS_2026.md)**: Registro de éxitos y fracasos del motor.
*   **[LIFECYCLE_FLOW.md](docs/meta/LIFECYCLE_FLOW.md)**: Ciclo de vida del organismo y gates de certificación.

### 📊 Resultados Verificados (`docs/reports/`)
*   **[SCIENTIFIC_BENCHMARK_v097.md](docs/reports/SCIENTIFIC_BENCHMARK_v097.md)**: Benchmark científico de rendimiento y paridad.
*   **[session_findings_v1.6.0_phase_3.1.md](docs/reports/session_findings_v1.6.0_phase_3.1.md)**: Hallazgos de la fase 3.1 (formato `.flat`).
*   **[session_findings_v1.6.0_phase_3.2.md](docs/reports/session_findings_v1.6.0_phase_3.2.md)**: Hallazgos de la fase 3.2.
*   **[factual_audit_phase_3.3.md](docs/reports/factual_audit_phase_3.3.md)**: Auditoría factual de la fase 3.3.
*   **[smollm2_fp32_parity.md](docs/reports/smollm2_fp32_parity.md)**: Paridad SmolLM2 FP32.
*   **[qwen2_distillation_report.md](docs/reports/qwen2_distillation_report.md)**: Reporte de destilación Qwen2.

### 🛠️ Guías Operativas (`docs/guides/`)
*   **[GAJE_CLI_GUIDE.md](docs/guides/GAJE_CLI_GUIDE.md)**: **MANUAL PRINCIPAL.** Comandos y parámetros del motor nativo Rust.
*   **[ARCHITECTURE.md](docs/guides/ARCHITECTURE.md)**: Arquitectura del motor.
*   **[OPERATIONAL_WORKFLOWS_V1.1.0.md](docs/guides/OPERATIONAL_WORKFLOWS_V1.1.0.md)**: Flujos de trabajo del protocolo GAJE-Flow.
*   **[USER_GUIDE.md](docs/guides/USER_GUIDE.md)**: Manual de usuario del ecosistema.
*   **[BREEDING_AND_BORN_GUIDE.md](docs/guides/BREEDING_AND_BORN_GUIDE.md)**: Guía de crianza y generación.
*   **[GAJE_PUBLISHING_GUIDE.md](docs/guides/GAJE_PUBLISHING_GUIDE.md)**: Guía de publicación.
*   **[SEMANTIC_ENRICHMENT_GUIDE.md](docs/guides/SEMANTIC_ENRICHMENT_GUIDE.md)**: Enriquecimiento semántico.

### 🗺️ Planes y Roadmap (`docs/plans/`)
*   **[MASTER_ROADMAP_2026.md](docs/plans/MASTER_ROADMAP_2026.md)**: Visión estratégica a largo plazo.
*   **[NEXT_STEPS_2026.md](docs/plans/NEXT_STEPS_2026.md)**: Próximos pasos operativos.
*   **[NATIVE_SEMANTIC_RAG_PLAN.md](docs/plans/NATIVE_SEMANTIC_RAG_PLAN.md)**: Plan de RAG semántico nativo.
*   **[PLAN_2BIT_ANCHORED_QUANTIZATION.md](docs/plans/PLAN_2BIT_ANCHORED_QUANTIZATION.md)**: Cuantización anclada 2-bit.
*   **[OPERATION_REBIRTH.md](docs/plans/OPERATION_REBIRTH.md)**: Metodología de transmutación.
*   **[DOCUMENTATION_CONSOLIDATION_PLAN.md](docs/plans/DOCUMENTATION_CONSOLIDATION_PLAN.md)**: Sincronización documental.

### 🏛️ Especificaciones Técnicas (`docs/sdd/` y `docs/bdd/`)
*   **[docs/sdd/ARCHITECTURE_CORE.md](docs/sdd/ARCHITECTURE_CORE.md)**: Arquitectura del núcleo.
*   **[docs/sdd/QAT_IMPLEMENTATION_DETAILS.md](docs/sdd/QAT_IMPLEMENTATION_DETAILS.md)**: Detalles de QAT.
*   **[docs/sdd/MIXED_BIT_ARCHITECTURE.md](docs/sdd/MIXED_BIT_ARCHITECTURE.md)**: Arquitectura de bits mixtos.
*   **[docs/bdd/BORN_GENOMIC_FLOW.md](docs/bdd/BORN_GENOMIC_FLOW.md)**: Flujo genómico.
*   **[docs/certifications/CERTIFICATION_REPORT_V1.5.md](docs/certifications/CERTIFICATION_REPORT_V1.5.md)**: Reporte de certificación.

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
*   **`docs/archive/legacy_versions/`** y **`docs/archive/plans/`**: Documentación de etapas previas.

---
*Documento consolidado bajo el protocolo GAJE-Flow (Agosto 2026).*
