# 🧬 GAJE Nomenclatura Oficial del Ecosistema

> **Versión del Estándar:** GAJE Helix Runtime v1.7.0-alpha (Silver Adult)  
> **Directorio de Modelos Certificados de Producción:** `models/production/*.flat`  
> **Directorio de Modelos Experimentales / Nacidos:** `models/born/*.gaje` (Investigación)  
> **Motor de Inferencia:** Zero-Copy Flat Memory Map (`mmap`) con aceleración nativa SIMD (AVX2/FMA/NEON) en Rust 2021.  
> **Estandarización Dual-Layer:** Ver [NOMENCLATURE_AND_STANDARDIZATION_MAPPING.md](../research/NOMENCLATURE_AND_STANDARDIZATION_MAPPING.md).

---

## 🏛️ 1. Jerarquía y Filosofía de Extensiones

| Extensión | Categoría | Definición y Origen | Ubicación | Estado |
| :---: | :--- | :--- | :--- | :---: |
| **`*.flat`** | ⚡ **Estándar de Producción Certificado** | Modelos con cabecera `FlatHeaderV2` y pesos híbridos (`Q4_0` en cuerpo + `FP32` en `token_embd` / `lm_head` o `Q8_0`). Inferencia ultrarrápida `mmap` zero-copy. | `models/production/` | 🟢 **Producción** |
| **`*.gaje`** | 🧬 **Organismos Nacidos / Experimentales** | Modelos nativos exploratorios generados en el marco bio-inspirado (ADN artificial, matrices cuaternarias 2-bit, DNI y balance epigenético). | `models/born/` | 🔬 **Investigación** |

---

## ⚡ 2. Modelos Certificados en Producción (`models/production/*.flat`)

Disponibles en local y sincronizados con el [Hub Oficial de Hugging Face (`eaguilar/gaje-models`)](https://huggingface.co/eaguilar/gaje-models):

| Alias / Archivo | Arquitectura Base | Peso HD | RAM Residente (mmap) | Cuantización | Rol y Especialidad |
| :--- | :--- | :---: | :---: | :---: | :--- |
| 👑 **`gaje_prime_3b.flat`** (`qwen2_5_3b.flat`) | **Qwen2.5-3B-Instruct** | **2.24 GB** | **~1.7 GB - 2.5 GB** | `Q4_0` + `Q8_0` | **Insignia General Multilingüe**: Modelo principal. Fluidez en 29+ idiomas, código y síntesis lógica. |
| 🚀 **`gaje_ultra_7b.flat`** | **Qwen2.5-7B-Instruct** | **4.88 GB** | **~4.2 GB - 5.5 GB** | `Q4_0` + `FP32` | **Servidor / Cloud Nativo**: Razonamiento profundo, matemáticas y programación avanzada. |
| 🧠 **`gaje_nano_1.5b.flat`** (`deepseek_r1_1_5b.flat`) | **DeepSeek-R1-Distill-1.5B** | **1.23 GB** | **~475 MB - 1.4 GB** | `Q4_0` + `Q8_0` | **Razonamiento Máximo (CoT)**: Monólogo interno (`<think>`), deducción y resolución lógica. |
| ⚡ **`qwen2_0_5b.flat`** | **Qwen2-0.5B-Instruct** | **499 MB** | **~350 MB - 500 MB** | `Q4_0` + `Q8_0` | **Micro-Modelo Ultrarrápido**: Respuestas de baja latencia para entornos de recursos limitados. |
| 🧬 **`gaje_pico_135m.flat`** (`smollm2_135m.flat`) | **SmolLM2-135M-Instruct** | **474 MB** | **~180 MB - 300 MB** | `Q4_0` + `FP32` | **Nano-Agente Edge**: Dispositivos móviles, IoT y validación de compresión zero-copy. |

---

## 🧬 3. Banco de Investigación y Organismos Nacidos (`models/born/*.gaje`)

> [!NOTE]
> Estos modelos forman parte de la línea de investigación teórica cuaternaria/discreta. Su rendimiento y perplejidad generativa están documentados en `docs/reports/` y `docs/research/`. La ruta operativa oficial para despliegues es el formato `.flat` de la sección anterior.

| Archivo | Estado Biológico | Tamaño | Rol y Descripción |
| :--- | :--- | :---: | :--- |
| 👑 **`max.gaje`** | **Organismo Conforme 2-Bit (Born Native)** | **11.39 MB** | **Insignia de Nacimiento Nativo**: Primer organismo nacido desde cero bajo la constelación cuaternaria $Q2\_0\_CONFORMAL$ (8 bloques, 256 dim, $A,C,G,T$). |
| 🧬 **`feto_genomico_v1.gaje`** | **Feto / Embrión en Desarrollo** | **2.08 GB** | **Organismo Genómico en Desarrollo**: Modelo con arquitectura nativa `gaje_native` (2 bloques, 768 dim). Banco de pruebas de mutación y destilación. |
