# 🧬 Protocolo GAJE: Adaptación Semántica y Compresión Genómica (v1.7.4-alpha)

[![Version](https://img.shields.io/badge/version-1.7.4--alpha_Helix_Ecosystem-purple)](docs/meta/EMPIRICAL_TRUTH_STATE.md)
[![Engine](https://img.shields.io/badge/Engine-Pure_Rust_PyO3_WASM-orange.svg)](src/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Format](https://img.shields.io/badge/Format-Zero--Copy_GAJE_mmap-brightgreen.svg)](docs/plans/UNIFIED_GAJE_ADAPTIVE_FORMAT_PLAN.md)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20Hub-yellow)](https://huggingface.co/eaguilar/gaje-models)
[![Language: English](https://img.shields.io/badge/Language-English-blue.svg)](README.en.md)
[![Language: 中文](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red.svg)](README.zh.md)

**GAJE (Genomic Adaptive Joint Embedding)** es un motor de inferencia nativa en Rust y compresión de alta densidad para Modelos de Lenguaje Masivos (LLMs). En producción comprime el cuerpo del transformer a **4-bits por peso (Q4_0, 16 centroides optimizados)** y mantiene los embeddings críticos (`token_embd` y `lm_head`) en **FP32**, dentro del formato plano unificado **`.gaje` v2** de acceso zero-copy por mapeo de memoria (mmap). Incluye el servidor HTTP de producción soberano **`gaje-server` (Zero-Python Runtime)** con streaming SSE token-a-token en tiempo real, hot-swap concurrente, capacidades adaptativas de mutación in-place, memoria persistente **Island Model `.gmem`**, cabeceras autodescriptivas dinámicas (**`ArchitectureDescriptor`**) y motor **WebAssembly In-Browser (Zero-Server)**.

> **2-bits (experimental):** la cuantización de **2-bits por peso (4 estados `00=A`, `01=C`, `11=G`, `10=T`)** se desarrolla en el módulo neuromórfico (`src/nn/spiking`) y quedó documentada como frente de investigación (inviable en hardware comercial por costo de cómputo). **La ruta de producción certificada es Q4_0 + FP32.**

---

## 📦 Catálogo de Modelos Certificados (Hugging Face Hub)

Los modelos oficiales listos para ejecución nativa en servidor o WebAssembly en navegador están disponibles en el [Repositorio Oficial de Hugging Face (`eaguilar/gaje-models`)](https://huggingface.co/eaguilar/gaje-models):

| Organismo / Modelo | Formato | Tamaño | Entorno Óptimo | Precisión y Capacidad |
| :--- | :---: | :---: | :---: | :--- |
| **`gaje_nano_1.5b.gaje`** | `.gaje` v2 | **1.23 GB** | WebAssembly (Móvil / Web) | Ultra-rápido, bajo consumo RAM, ideal para teléfonos. |
| **`gaje_prime_3b.gaje`** | `.gaje` v2 | **2.24 GB** | WASM Desktop / Cloud | Equilibrado, alta coherencia contextual y lógica general. |
| **`gaje_ultra_7b.gaje`** | `.gaje` v2 | **4.88 GB** | Servidor / Cloud Nativo | Razonamiento profundo, codificación y tareas complejas. |

---

## 🔬 Estado Empírico y Certificación del Motor (v1.7.0-alpha)

Siguiendo el **Mandato de Verdad Empírica** ([`docs/meta/EMPIRICAL_TRUTH_STATE.md`](docs/meta/EMPIRICAL_TRUTH_STATE.md)), el motor GAJE Helix cuenta con la siguiente certificación oficial:

### 🟢 Validación de Fase 0-3: Proyecto Completado

| Fase | Estado | Logro Principal |
|------|--------|-----------------|
| **Fase 0** | ✅ Aprobada | H1 (2.24× speedup SPSA vs mutación) y H3 (21.56× currículo híbrido) |
| **Fase 1** | ✅ Aprobada | Módulo Rust `train-zero-order`, ~21 tok/s, <50 MB memoria |
| **Fase 2** | ✅ Validada | Arquitectura escalado Qwen2.5, PPL ~1.60, 16× compresión |
| **Fase 3** | ✅ Validada | SPSA niche weights `.gmem`, needle_recall 1.0 mantenido |

### 📊 Métricas de Éxito Finales

| Métrica | Umbral | Resultado Actual | Estado |
|---------|--------|-----------------|--------|
| Speedup SPSA vs mutación (Fase 0) | ≥ 2× | **2.24×** | ✅ Cumplido |
| Speedup Currículo H3 | - | **21.56×** | ✅ Cumplido |
| Throughput vs ES refutado (Fase 1) | ≥ 5× | **~21 tok/s** (20×) | ✅ Cumplido |
| Memoria adicional (Fase 1) | < 50 MB | **0 MB** (zero-copy) | ✅ Cumplido |
| Estabilidad 10⁴ pasos (Fase 0) | Sí | **Sí** (pairs antitéticos) | ✅ Cumplido |
| PPL post-IQAT (Fase 2) | < 50 | **~1.60** | ✅ Cumplido |
| Needle recall Fase 3 | Mantener | **1.0** | ✅ Cumplido |

### 🏆 Certificación de Producción (Ryzen 7 5800H)

| Modelo | Formato | Throughput CPU | Consumo RAM | Speedup vs FP32 |
|--------|---------|----------------|-------------|-----------------|
| **Qwen2.5 1.5B Instruct** | `.gaje` Híbrido v2 | 11.31-12.13 tok/s | 2.6 GB Virtual | 8.2-8.8× |
| **Qwen2 0.5B Instruct** | `.gaje` Híbrido v2 | 19.20-23.00 tok/s | ~498 MiB | 13.9-16.7× |
| **SmolLM2 135M Instruct** | `.gaje` Zero-Copy | 28.28-32.10 tok/s | ~472 MB | 20.5-23.3× |

### 📈 Comparación vs PyTorch FP32

| Formato | Throughput | Consumo Memoria | Speedup |
|---------|------------|-----------------|---------|
| **HuggingFace PyTorch FP32** | 1.38 tok/s | 1,980 MB | 1× |
| **GAJE Engine nativo `.gaje`** | 19-32 tok/s | 448 MB RSS (**77% menos**) | **14-23×** |

---

### 🏆 1. Experimento de Control A/B (GAJE Q4_0 vs. HuggingFace PyTorch FP32)

Se ejecutó la prueba A/B ciega y de paridad en la misma máquina comparando el modelo original en FP32 (`Qwen/Qwen2-0.5B-Instruct`) en **PyTorch** contra el motor nativo **GAJE 4-bit `.gaje`** sobre un procesador **AMD Ryzen 7 5800H**:

| Entorno de Inferencia | Formato / Precisión | Respuesta Generada Exacta | Throughput E2E Real | Consumo de RAM |
| :--- | :---: | :--- | :---: | :---: |
| **HuggingFace PyTorch** | **FP32 Original (Alibaba)** | *"El planeta más grande del Sistema Solar es la Tierra, con una"* | **`1.38 tok/s`** | $1,980\text{ MB}$ |
| **GAJE Engine Nativo (`.gaje`)** | **4-bit Genómico Zero-Copy** | *"El planeta más grande del Sistema Solar es la Tierra."* | **`19.2 - 23.0 tok/s`** | **`448 MB` (RSS, ~77% vs FP32)** |

---

### ⚡ 2. Rendimiento Multimodelo Certificado en Producción (Ryzen 7 5800H)

| Modelo / Arquitectura | Formato Binario | Respuesta Factual Certificada | Throughput CPU | Tiempo de Carga Cold Start | Consumo de RAM Viva |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Qwen2.5 1.5B Instruct** | **`.gaje` (Híbrido v2)** | Español: *"La capital de Francia es París."* | **`11.31 - 12.13 tok/s`** | **`< 0.75 ms` (mmap)** | **`2.6 GB` (Virtual)** |
| **Qwen2 0.5B Instruct** | **`.gaje` (Híbrido v2)** | Chino: *"木星"* (Júpiter) / Español: *"París"* | **`19.20 - 23.00 tok/s`** | **`< 0.75 ms` (mmap)** | **`~498 MiB` (~74% vs FP32)** |
| **SmolLM2 135M Instruct** | **`.gaje` (Zero-Copy)** | Inglés: *"Berlin."* / *"100°C"* | **`28.28 - 32.10 tok/s`** | **`< 0.75 ms` (mmap)** | **`~472 MB` (cuerpo Q4_0 + embeddings FP32)** |

> [!IMPORTANT]
> **Formato .gaje v2 Híbrido**: Para preservar la fidelidad semántica y evitar el colapso del vocabulario multilingüe en idiomas CJK y europeos, el formato binario `.gaje` (con alias retrocompatible `.flat`) almacena las capas críticas de embeddings (`token_embd` y `lm_head`) en **FP32** (4 bytes/peso), mientras que el cuerpo del transformador (los bloques de atención y FFN) se comprime en **Q4_0** (4-bits) o **Q2_0** (2-bits).

---

### 🏝️ 3. Island Model (.gmem): Memoria Hipocampal RAG Submilisegundo

El sistema integra recuperación y persistencia de memoria contextual en tiempo real mediante índices binarios planos `.gmem` alineados a 64 bytes:

* **Latencia de Recuperación RAG**: **`< 0.5 ms`** por consulta multinicho mediante *Weighted Mean Pooling* sobre embeddings de entrada ($W_E$).
* **Gating de Brecha de Entropía ($\Delta_{top} \ge 0.12$)**: Pruning competitivo K-WTA y descarte matemático de consultas ambiguas o casos trampa.
* **Calibración Satélite por Whitening**: Centrado anisotrópico ($\boldsymbol{\mu} \in \mathbb{R}^D$) persistido en `data/calibration/` con fallback automático (Opción C).
* **Telemetría Canónica de 6 Estados**: Supervisión en tiempo real en SSE y HUD (`memory_injected`, `rejected_low_similarity`, `rejected_entropy_gap`, etc.).
* **Presupuesto de Contexto**: Inyección protegida dentro del bloque de sistema (`ChatML` / `Llama3`) preservando la fidelidad de turnos.

---

## 🛠️ Fundamentos Arquitectónicos de GAJE-Flow

### 1. Formato Binario Plano Zero-Copy Autodescriptivo (`.gaje` v2)
La cabecera binaria **`FlatHeaderV2`** contiene un descriptor dinámico de arquitectura (**`ArchitectureDescriptor`**). Al exportar un modelo con `gaje-cli export-flat` (o transmutadores), se extraen automáticamente las dimensiones, constantes de RoPE y el tipo de permutación de atención ($Q/K$), eliminando la intervención manual y blindando la carga contra bugs de alineación de atención.

### 2. Estabilización de Algoritmos QAT (Quantization-Aware Training)
GAJE incluye capacidades nativas de afinamiento y optimización local post-cuantización. Las actualizaciones del optimizador de centroides en Rust (`linear.rs`) se normalizan dividiendo el gradiente acumulado entre las activaciones reales de cada centroide (`centroid_counts`), erradicando pánicos de `NaN`/`Inf` y estabilizando la convergencia matemática del error de cuantización.

### 3. Muestreo Lagrangiano de Mínima Acción
La generación autoregresiva se modela como un sistema dinámico regido por el principio de mínima acción, evaluando la energía cinética $T$ (movilidad semántica) y el potencial $V$ (restricción gramatical):

$$\mathcal{L} = T - V$$

> El muestreo **Lagrangiano / Toroidal** (módulo `src/compute/lagrangian.rs`) es una heurística de generación del motor. Conviene distinguir la nomenclatura física de los resultados medibles: la fidelidad y el rendimiento se certifican en [`docs/reports/`](docs/reports/), no por el nombre del algoritmo.

---

## 📂 Organización del Repositorio (`v1.7.2-alpha`)

```text
gaje-semantic-compression/
├── src/                    # Núcleo Nativo en Rust (Kernels SIMD, LLM Engine, KV-Cache, Mmap Loader)
│   ├── bin/gaje-cli.rs     # CLI soberano único (sin scripts monolíticos descartables)
│   ├── cli/                # Capa de presentación y subcomandos de terminal (models, tools)
│   ├── compute/            # Motor numérico puro (sintergic, sampling, kernels SIMD, quantize)
│   ├── core/               # Estructuras base (gtok, tokenizadores, memoria de sesión de anillo, DNI)
│   ├── io/                 # E/S mmap zero-copy (adaptive, flat_reader, flat_writer, gguf/, gmem)
│   ├── nn/                 # Capas neuronales (WeightStorage, GenomicLLM, Spiking, Distiller)
│   └── server/             # Servidor HTTP embebido nativo con streaming SSE (tiny_http)
├── python/gaje/            # Puente PyO3 y Wrappers de Inferencia Nativas
├── examples/               # Demos de núcleo, Web UI, notebooks y utilidades Rust
│   └── ui/web_ui/          # Interfaz Visual Web UI (http://localhost:8080) y Servidor server.py
├── tests/                  # Suite de Pruebas (unit, integration, metrics, training, ui_e2e)
├── scripts/                # Herramientas de Mantenimiento y Transmutadores
├── benchmarks/             # Benchmarks de rendimiento (perplexity, decode, flat, RAG)
├── models/production/      # Modelos Cuantizados de Producción (.gaje plano v2)
└── docs/                   # Documentación Científica, Planes y Reportes de Certificación
    ├── reports/            # Resultados empíricos verificados (reportes de paridad y benchmarks)
    ├── guides/             # Manuales operativos (GAJE CLI, flujos de trabajo)
    ├── plans/              # Roadmaps y planes estratégicos
    ├── meta/               # Gobernanza y estado de verdad empírica
    └── archive/            # Investigación exploratoria y versiones heredadas
```

> **Nota de consolidación:** el contenido experimental (bins Rust exploratorios, notas de investigación y demos de etapas previas) se conserva íntegro en `legacy/` y `docs/archive/`. El árbol principal solo mantiene los componentes operativos y verificados.

---

## ⚡ Guía de Inicio Rápido — Binario Único Soberano (`gaje-cli`)

GAJE Helix se distribuye y ejecuta como un **ejecutable nativo autónomo en Rust** sin dependencias externas ni necesidad de Python en producción.

### 1. Compilación del Ejecutable Nativo
```bash
# Compilar binario de producción optimizado
cargo build --release --bin gaje-cli
```

### 2. Comandos Principales de `gaje-cli`

```bash
# Iniciar servidor HTTP nativo (Zero-Python) con streaming SSE en tiempo real y Web UI embebida
./target/release/gaje-cli serve --port 8080

# Sesión interactiva de Chat REPL en terminal
./target/release/gaje-cli chat --model models/production/gaje_pico_135m.gaje

# Descargar modelos directamente desde Hugging Face
./target/release/gaje-cli pull pico

# Catálogo e inspección estructural de modelos locales
./target/release/gaje-cli models list
./target/release/gaje-cli models inspect models/production/gaje_pico_135m.gaje

# Mutación genómica in-place sobre centroides (.gaje)
./target/release/gaje-cli mutate --model models/production/gaje_pico_135m.gaje --rate 0.05

# Inspección de linaje genético y generaciones adaptativas
./target/release/gaje-cli history --model models/production/gaje_pico_135m.gaje

# Exportar cualquier modelo (.gguf, .gaje) a formato plano zero-copy v2
./target/release/gaje-cli export-flat models/source/model.gguf -o models/production/model.gaje

# Benchmark de latencia (TTFT), Throughput (TPS) y Perplejidad (PPL)
./target/release/gaje-cli benchmark --model models/production/gaje_pico_135m.gaje --tokens 64

# Auditoría matemática de tensores (chequeo de 0 NaNs/Infs y entropía de centroides)
./target/release/gaje-cli audit models/production/gaje_pico_135m.gaje

# Diagnóstico de extensiones de hardware (AVX2, AVX512, NEON, FMA)
./target/release/gaje-cli doctor
```

### 3. Suite de Validación y Pruebas
```bash
# Ejecutar suite nativa completa en Rust
cargo test --lib
cargo test --test cli_standalone_test
```

---

## 🎯 Lo que GAJE no es

Para prevenir expectativas desalineadas antes de evaluar o desplegar el sistema:

* **No es un servidor multi-tenant empresarial:** Es una estación de inferencia y memoria soberana/edge diseñada para ejecución local, un solo usuario o hilo activo a la vez.
* **No es un runtime de sentence-embeddings de alta precisión contextual:** Su extractor ligero prioriza latencia submilisegundo ($< 0.5\text{ ms}$) y cero sobrecarga de memoria sobre inferencia contextual profunda.
* **No compite contra llama.cpp en throughput multi-hilo masivo en servidores:** Compite en huella mínima de memoria viva (`mmap` zero-copy), arranque en frío ultrarrápido y la integración de memoria asociativa persistente en un único binario autónomo.
* **No resuelve la paráfrasis semántica pura sin solapamiento léxico:** Depende de vocabulario compartido; resolver equivalencias abstractas sin términos comunes requeriría un modelo encoder satélite dedicado.
* **No ejecuta modelos > 3B en hardware de consumo sin degradación severa:** El catálogo certificado está enfocado estrictamente en la franja de 0.1B a 3B de parámetros.

---

## ⚠️ Limitaciones Conocidas y Alcance del Sistema

Este proyecto se rige por la **Verdad Empírica Certificada**. Las siguientes restricciones no son descuidos de implementación, sino decisiones de diseño deliberadas y propiedades matemáticas medidas formalmente:

### 1. Piso de Compresión Fiel: Q4_0
La cuantización por debajo de 4 bits (Q3_0, Q2_0) colapsa la coherencia semántica en contextos autorregresivos largos. Certificado empíricamente con similitud semántica $S_c < 0.40$ en capas intermedias y de salida (`tests/test_attention_ablation.rs`, `tests/test_ffn_bitdepth_isolation.rs`). La amplificación angular acumulada en la función Softmax vuelve matemáticamente inviable la compresión a 2-bits en transformers autoregresivos estándar. **La ruta de producción certificada es Q4_0 + embeddings FP32.**

### 2. Retrieval Semántico y Paráfrasis
El extractor hipocampal (`embed_text`) ejecuta *Weighted Mean Pooling* sobre los embeddings de entrada ($W_E \in \mathbb{R}^{V \times D}$) sin pasar por los bloques del transformer, logrando una latencia $< 0.5\text{ ms}$ con $0\text{ MB}$ de RAM adicional.
* **Compromiso / Decisión:** Depende de solapamiento léxico o léxico-semántico parcial. No recupera paráfrasis puras sin vocabulario común (ejemplos documentados en calibración: $P_6$, $P_{11}$, $P_{13}$ en `tests/test_threshold_calibration.rs`).
* **Alternativa:** Integrar un encoder denso dedicado (ej. MiniLM, BGE, E5-small), lo cual queda fuera del alcance del binario único ligero actual.

### 3. Concurrencia Serializada
El modelo activo en memoria está protegido por un bloqueo de lectura/escritura (`active_model.write()`). Las peticiones entrantes de chat y streaming SSE se serializan una tras otra.
* **Compromiso / Decisión:** Decisión de diseño orientada a hardware mononodo y edge modesto (Android/Termux, laptops de consumo) para garantizar cero carreras de memoria y estabilidad térmica.
* **Alternativa:** La concurrencia multi-inquilino paralela requeriría paginación de KV-Cache (*PagedAttention* / slots independientes), planificada para futuras evoluciones de servidor de alta concurrencia.

### 4. Vector de Centrado Satélite $\boldsymbol{\mu}$ y Degradación Controlada
Los modelos de alta dimensionalidad (SmolLM2 576d, Qwen2.5 896d) sufren colapso de cono anisotrópico. El sistema utiliza vectores de centrado satélite persistidos en `data/calibration/<modelo>.mu.bin`.
* **Degradación Controlada (Opción C Fallback):** Si el archivo `.mu.bin` no existe o está dañado, el motor conmuta automáticamente al umbral no blanqueado $\tau^* = 0.50$ y emite `whitening_missing: true` en la telemetría SSE.
* **Efecto medible:** Mayor tasa de rechazo o falsos positivos en el retrieval, pero el sistema mantiene el 100% de operatividad y estabilidad sin interrumpir la inferencia.

### 5. Alcance y Estado de los Componentes
* ✅ **Certificado para Producción:** Inferencia Q4_0 pura, cabeceras `FlatHeaderV2`, servidor SSE soberano en Rust (Zero-Python), memoria RAG Island Model con gating de entropía ($\Delta_{top} \ge 0.12$).
* 📁 **Cerrado con Rigor:** Cuantizaciones extremas Q2_0 y Q3_0 en pesos de atención/FFN (descartadas por inviabilidad matemática certificada).
* 🔬 **I+D Separada:** Afinamiento adaptativo de centroides (`fit_lm_head` funcional sobre corpus limpio; optimización de cuerpo completo numéricamente inestable en contextos largos).

### 6. Catálogo y Hardware Objetivo
El rendimiento y la estabilidad están validados formalmente para organismos de 0.1B a 3B en CPUs de consumo estándar (AMD Ryzen 7 5800H, Intel Core i7-8550U, ARM Cortex en Android). Modelos mayores a 3B no están certificados para ejecución local en estas arquitecturas.

---

## ⚖️ Licencia y Gobernanza
Licenciado bajo la **GNU Affero General Public License v3.0 (AGPL-3.0)**. Ver [LICENSE](LICENSE) para más información.

---
*Protocolo GAJE-Flow v1.7.4-alpha (Silver Adult) — Hacia la Soberanía de la Inferencia de Ultra-Alta Densidad.*
