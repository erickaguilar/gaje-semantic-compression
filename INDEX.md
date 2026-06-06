# 🗺️ Índice Maestro: DNA Semantic Compression (v1.0.0-alpha)

Este documento sirve como mapa central del repositorio tras la consolidación del hito **Silver Adult**. Define la estructura lógica, la ubicación de los planes estratégicos y la suite de validación.

---

## 📂 1. Documentación Estratégica (`docs/`)
La carpeta `docs` ha sido consolidada para reflejar la realidad técnica de 2026.

### ⚖️ Auditoría de Verdad y Estado Real
*   **[EMPIRICAL_TRUTH_STATE.md](docs/meta/EMPIRICAL_TRUTH_STATE.md)**: **DOCUMENTO CRÍTICO.** Mapa de la realidad que contrasta el código con resultados empíricos (PPL ~572).
*   **[TECHNICAL_BIRTH_AND_REALITY.md](docs/meta/TECHNICAL_BIRTH_AND_REALITY.md)**: Resumen del proceso de transmutación toroidal y el abismo semántico actual.
*   **[LIFECYCLE_FLOW.md](docs/meta/LIFECYCLE_FLOW.md)**: Diagrama de flujo del ciclo de vida y Gates de Certificación.

### 🛠️ Guías Operativas (`docs/guides/`)
*   **[GAJE_CLI_GUIDE.md](docs/guides/GAJE_CLI_GUIDE.md)**: **MANUAL PRINCIPAL.** Guía completa de comandos y parámetros para el motor nativo de Rust.
*   **[OPERATIONAL_WORKFLOWS_V1.1.0.md](docs/guides/OPERATIONAL_WORKFLOWS_V1.1.0.md)**: Flujos de trabajo del protocolo GAJE-Flow.
*   **[USER_GUIDE.md](docs/guides/USER_GUIDE.md)**: Manual de usuario para interacción con el ecosistema.

### 📋 Planes y Roadmap (`docs/plans/`)
*   **[NEXT_STEPS_2026.md](docs/plans/NEXT_STEPS_2026.md)**: **PLAN DE CHOQUE.** Hoja de ruta inmediata para la Operación Rescate (L1 y L2).
*   **[MASTER_ROADMAP_2026.md](docs/plans/MASTER_ROADMAP_2026.md)**: Visión estratégica a largo plazo.
*   **[DOCUMENTATION_CONSOLIDATION_PLAN.md](docs/plans/DOCUMENTATION_CONSOLIDATION_PLAN.md)**: Plan activo de sincronización documental.
*   **[ISLAND_MODEL_IMPLEMENTATION.md](docs/plans/ISLAND_MODEL_IMPLEMENTATION.md)**: (Fase 4 - Aspiracional) Paralelismo evolutivo en ARM.

---

## 🧪 2. Suite de Validación (`tests/`)
Estructura unificada compatible con `pytest` para asegurar la integridad del motor.

*   **`tests/unit/`**: Pruebas de bajo nivel.
    *   `test_kernels.py`: Consistencia de kernels de Rust vs NumPy (Mixed Precision).
    *   `test_rope_logic.py`: Validación de alineación RoPE (Split vs Interleaved).
*   **`tests/integration/`**: Flujos completos de sistema.
    *   `test_system_integration.py`: Integración de GajeIndex (ADC) e inferencia LLM.
*   **`tests/metrics/`**: Evaluación de calidad.
    *   `test_perplexity.py`: Medición de PPL sobre texto real y simulado.
    *   `test_precision_accuracy.py`: Validación de Recall@10 y predicción de tokens.

---

## 🧬 3. Gestión de Modelos (`models/`)
Jerarquía profesional de artefactos genómicos.

*   **`models/production/`**: Modelos soberanos y estables (ej. `silver_adult_steel.gaje`).
*   **`models/research/`**: Experimentos activos (Vortice, Island Model).
*   **`models/test_artifacts/`**: Modelos mínimos para ejecución de tests.
*   **`models/gguf/`**: Modelos maestros de referencia (SmolLM2, Qwen2).
*   **`models/archive/`**: Histórico de versiones experimentales previas.

---

## 📊 4. Activos de Datos (`data/`)
Organización jerárquica para entrenamiento y experimentación.

*   **`data/datasets/master/`**: Datasets consolidados para entrenamiento pesado (Silver Adult).
*   **`data/datasets/specialized/`**: Datasets de dominio específico (Rust, Lógica, Cultura AI).
*   **`data/datasets/training_splits/`**: Fragmentos pequeños para validación rápida.
*   **`data/training/curriculum/`**: Definición de fases de aprendizaje y nichos.
*   **`data/experiments/active_tests/`**: Resultados de pruebas de ingesta (DNI) y nichos.

---

## 🐍 5. Ecosistema Python (`python/gaje/`)
Interfaz de alto nivel y orquestación para investigación y entrenamiento.

*   **`python/gaje/core/`**: Motores de alto nivel (SessionMemory, DNIEngine).
*   **`python/gaje/nn/`**: Definición de modelos (`stabilized.py`) y lógica de entrenamiento híbrido (`trainer.py`).
*   **`python/gaje/processing/`**: Pipelines de procesamiento de señal y balanceo homeostático.
*   **`python/gaje/utils/`**: Utilidades de cuantización, métricas y gestión de versiones.
*   **`python/gaje/_impl/`**: Puente nativo (CFFI) para comunicación con el núcleo en Rust.

---

## 🏎️ 6. Análisis y Rendimiento (`benchmarks/`)
Evaluación técnica y métricas de eficiencia en el borde.

*   **`benchmarks/performance/`**: Suite unificada de latencia y throughput (`latencies_and_throughput.py`).
*   **`benchmarks/analysis/`**: Análisis profundo de entropía y señal semántica.
*   **`benchmarks/research/`**: Investigaciones de escalabilidad y diagnósticos de arquitectura.
*   **`benchmarks/logs/`**: Repositorio central de resultados históricos y de entrenamiento.

---

## 🛠️ 7. Scripts y Utilidades (`scripts/`)
Herramientas de soporte para mantenimiento, automatización e investigación.

*   **`scripts/setup/`**: Preparación del entorno y descarga de activos (modelos HF).
*   **`scripts/calibration/`**: Ajuste fino de parámetros homeostáticos y estabilidad.
*   **`scripts/data_processing/`**: Generación de datasets sintéticos y crawlers de datos.
*   **`scripts/maintenance/`**: Tareas de sistema, backups y consolidación de datasets.
*   **`scripts/benchmarks/`**: Scripts de evaluación específicos (ej. Needle in a Haystack).
*   **`scripts/research/`**: Utilidades para experimentos de arquitectura y topología.
*   **`scripts/archive/`**: Repositorio de scripts históricos de fases previas.

---

## 🏛️ 8. Archivo Histórico (`legacy/`)
Experimentos y versiones previas superadas por la arquitectura actual.

*   **`legacy/archive/`**: Contenedor de scripts, binarios de Rust y modelos Python de fases anteriores.
*   **`legacy/README.md`**: Explicación del contenido histórico y sus dependencias obsoletas.

---

## 🧪 9. Herramientas y Demos (`examples/`)
Vitrina de capacidades y aplicaciones del ecosistema GAJE.

*   **`examples/core_demos/`**: Demos funcionales de terminal.
    *   `chat_soberano.py`: Chat interactivo con memoria toroidal y soporte neuromórfico.
    *   `coherence_demo.py`: Evaluación rápida de sentido común y gramática.
    *   `smg1_protocol_demo.py`: Inferencia basada en eventos y capas neuromórficas.
    *   `born_genomic_demo.py`: Proceso de crianza y evolución de micro-genomas.
*   **`examples/ui/`**: Interfaces visuales y gráficas.
    *   `web_ui/`: Aplicación web nativa para interacción y monitorización.
    *   `visual_ui/`: Visualizaciones de topología y dinámicas neuronales.
*   **`examples/legacy_research/`**: Archivo de demos de etapas previas de investigación.

---
*Documentación generada automáticamente bajo el protocolo GAJE-Flow v1.5 (Junio 2026).*
