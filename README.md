# 🧬 Protocolo GAJE: Adaptación Semántica y Compresión Genómica (v1.6.0-alpha)

[![Version](https://img.shields.io/badge/version-1.6.0--alpha_Silver_Adult-purple)](docs/meta/EMPIRICAL_TRUTH_STATE.md)
[![Engine](https://img.shields.io/badge/Engine-Pure_Rust_PyO3-orange.svg)](src/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Format](https://img.shields.io/badge/Format-Zero--Copy_Flat_mmap-brightgreen.svg)](docs/reports/session_findings_v1.6.0_phase_3.1.md)

**GAJE (Genomic Adaptive Joint Embedding)** es un protocolo de inferencia nativa en Rust y compresión de ultra-alta densidad para Modelos de Lenguaje Masivos (LLMs). El protocolo empaqueta los tensores neuronales en un alfabeto genómico digital de **4-bits por peso (16 centroides optimizados)** y **2-bits por peso (4 estados: `00=A`, `01=C`, `11=G`, `10=T`)**, integrando memoria persistente zero-copy (**Island Model `.gmem`**), cabeceras autodescriptivas dinámicas (**`ArchitectureDescriptor`**) y carga instantánea por mapeo de memoria en disco (**`.gaje.flat` v2**).

---

## 🔬 Estado Empírico y Certificación del Motor (v1.6.0-alpha)

Siguiendo el **Mandato de Verdad Empírica** ([`docs/meta/EMPIRICAL_TRUTH_STATE.md`](docs/meta/EMPIRICAL_TRUTH_STATE.md)), el motor GAJE Helix cuenta con la siguiente certificación oficial:

### 🏆 1. Experimento de Control A/B (GAJE Q4_0 vs. HuggingFace PyTorch FP32)

Se ejecutó la prueba A/B ciega y de paridad en la misma máquina comparando el modelo original en FP32 (`Qwen/Qwen2-0.5B-Instruct`) en **PyTorch** contra el motor nativo **GAJE 4-bit `.gaje.flat`** sobre un procesador **AMD Ryzen 7 5800H**:

| Entorno de Inferencia | Formato / Precisión | Respuesta Generada Exacta | Throughput E2E Real | Consumo de RAM |
| :--- | :---: | :--- | :---: | :---: |
| **HuggingFace PyTorch** | **FP32 Original (Alibaba)** | *"El planeta más grande del Sistema Solar es la Tierra, con una"* | **`1.38 tok/s`** | $1,980\text{ MB}$ |
| **GAJE Engine Nativo (`.flat`)** | **4-bit Genómico Zero-Copy** | *"El planeta más grande del Sistema Solar es la Tierra."* | **`19.2 - 23.0 tok/s`** | **`448 MB` (`87.5%` de ahorro)** |

---

### ⚡ 2. Rendimiento Multimodelo Certificado en Producción (Ryzen 7 5800H)

| Modelo / Arquitectura | Formato Binario | Respuesta Factual Certificada | Throughput CPU | Tiempo de Carga Cold Start | Consumo de RAM Viva |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Qwen2.5 1.5B Instruct** | **`.gaje.flat` (Híbrido v2)** | Español: *"La capital de Francia es París."* | **`11.31 - 12.13 tok/s`** | **`< 0.75 ms` (mmap)** | **`2.6 GB` (Virtual)** |
| **Qwen2 0.5B Instruct** | **`.gaje.flat` (Híbrido v2)** | Chino: *"木星"* (Júpiter) / Español: *"París"* | **`19.20 - 23.00 tok/s`** | **`< 0.75 ms` (mmap)** | **`448 MB` (`87.5%` de ahorro)** |
| **SmolLM2 135M Instruct** | **`.gaje.flat` (Zero-Copy)** | Inglés: *"Berlin."* / *"100°C"* | **`28.28 - 32.10 tok/s`** | **`< 0.75 ms` (mmap)** | **`140 MB` (`93.0%` de ahorro)** |

> [!IMPORTANT]
> **Formato .flat v2 Híbrido**: Para preservar la fidelidad semántica y evitar el colapso del vocabulario multilingüe en idiomas CJK y europeos, el formato `.flat` de GAJE almacena las capas críticas de embeddings (`token_embd` y `lm_head`) en **FP32** (4 bytes/peso), mientras que el cuerpo del transformador (los bloques de atención y FFN) se comprime en **Q4_0** (4-bits).

---

### 🏝️ 3. Island Model (.gmem): Persistencia Submilisegundo

El sistema integra persistencia de contexto contextual mediante índices binarios planos `.gmem` alineados a 64 bytes:

* **Latencia de Recuperación RAG**: **`0.75 ms`** ($750\text{ µs}$) por consulta multinicho.
* **Arranque en Frío (Cold Start `.gmem`)**: **`0.12 ms`** ($120\text{ µs}$) desde archivo en disco.
* **Presupuesto de Contexto**: Inyección automática de $128\text{ tokens}$ de alta resonancia ($\text{CosSim} = 0.9998$).

---

## 🛠️ Fundamentos Arquitectónicos de GAJE-Flow

### 1. Formato Binario Plano Zero-Copy Autodescriptivo (`.gaje.flat` v2)
La cabecera binaria **`FlatHeaderV2`** contiene un descriptor dinámico de arquitectura (**`ArchitectureDescriptor`**). Al exportar un modelo con `export_gaje_flat.py`, se extraen automáticamente las dimensiones, constantes de RoPE y el tipo de permutación de atención ($Q/K$), eliminando la intervención manual y blindando la carga contra bugs de alineación de atención.

### 2. Estabilización de Algoritmos QAT (Quantization-Aware Training)
GAJE incluye capacidades nativas de afinamiento y optimización local post-cuantización. Las actualizaciones del optimizador de centroides en Rust (`linear.rs`) se normalizan dividiendo el gradiente acumulado entre las activaciones reales de cada centroide (`centroid_counts`), erradicando pánicos de `NaN`/`Inf` y estabilizando la convergencia matemática del error de cuantización.

### 3. Muestreo Lagrangiano de Mínima Acción
La generación autoregresiva se modela como un sistema dinámico regido por el principio de mínima acción, evaluando la energía cinética $T$ (movilidad semántica) y el potencial $V$ (restricción gramatical):

$$\mathcal{L} = T - V$$

---

## 📂 Organización del Repositorio (`v1.6.0-alpha`)

```text
gaje-semantic-compression/
├── src/                    # Núcleo Nativo en Rust (Kernels SIMD AVX2/FMA, LLM Engine, KV-Cache, Mmap Loader)
│   └── bin/gaje-cli.rs     # CLI principal del motor nativo
├── python/gaje/            # Puente PyO3 y Wrappers de Inferencia Nativas
├── examples/               # Demos de núcleo, Web UI, notebooks y utilidades Rust
│   └── ui/web_ui/          # Interfaz Visual Web UI (http://localhost:8080) y Servidor server.py
├── tests/                  # Suite de Pruebas (unit, integration, metrics, training, ui_e2e)
├── scripts/                # Herramientas de Mantenimiento y Exportadores Flat (.gaje.flat)
├── models/production/      # Modelos Cuantizados de Producción (Qwen2 0.5B, SmolLM2 135M)
└── docs/                   # Documentación Científica, Planes y Reportes de Certificación
    ├── reports/            # Resultados empíricos verificados (reportes de paridad y benchmarks)
    ├── guides/             # Manuales operativos (GAJE CLI, flujos de trabajo)
    ├── plans/              # Roadmaps y planes estratégicos
    ├── meta/               # Gobernanza y estado de verdad empírica
    └── archive/            # Investigación exploratoria y versiones heredadas
```

> **Nota de consolidación:** el contenido experimental (bins Rust exploratorios, notas de investigación y demos de etapas previas) se conserva íntegro en `legacy/` y `docs/archive/`. El árbol principal solo mantiene los componentes operativos y verificados.

---

## ⚡ Guía de Inicio Rápido y Despliegue Web UI

### 1. Instalación y Compilación Nativa (PyO3)
```bash
# Crear entorno virtual optimizado
uv venv && source .venv/bin/activate

# Compilar motor nativo Rust con maturin optimizado para CPU host
maturin develop --release --features python
```

### 2. Ejecutar la Web UI Interactiva
```bash
python examples/ui/web_ui/server.py
```
Abre en tu navegador `http://localhost:8080` y selecciona dinámicamente tu modelo `.flat` optimizado.

### 3. Ejecutar Suite de Validación Nativa
```bash
# Ejecutar la suite de pruebas de Python (unitaria, integración y métricas)
pytest tests/unit tests/integration tests/metrics
```

---

## ⚖️ Licencia y Gobernanza
Licenciado bajo la **GNU Affero General Public License v3.0 (AGPL-3.0)**. Ver [LICENSE](LICENSE) para más información.

---
*Protocolo GAJE-Flow v1.6.0-alpha (Silver Adult) — Hacia la Soberanía de la Inferencia de Ultra-Alta Densidad.*
