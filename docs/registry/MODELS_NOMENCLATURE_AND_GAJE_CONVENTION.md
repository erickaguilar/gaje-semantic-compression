# 🧬 GAJE Nomenclatura Oficial del Ecosistema

> **Versión del Estándar:** GAJE Genomic Runtime v0.9.8  
> **Directorio de Modelos Nacidos por GAJE:** `models/born/*.gaje`  
> **Directorio de Modelos Transmutados:** `models/production/*.flat`  
> **Motor de Inferencia:** Zero-Copy Flat Memory Map (`mmap`) con aceleración nativa SIMD AVX2/FMA en Rust 2021.

---

## 🏛️ 1. Jerarquía y Filosofía de Extensiones

| Extensión | Categoría | Definición y Origen | Ubicación |
| :---: | :--- | :--- | :--- |
| **`*.gaje`** | 🧬 **Organismos Nacidos por GAJE** | Modelos **nacidos, entrenados o destilados desde cero** en el protocolo bio-inspirado de GAJE (ADN artificial, matrices de 2/4-bit, DNI y balance epigenético). | `models/born/` |
| **`*.flat`** | ⚡ **Modelos Externos Transmutados** | Modelos consolidados y pre-entrenados (Qwen, DeepSeek, SmolLM) cuantizados y adaptados al formato binario plano para inferencia ultrarrápida `mmap` zero-copy. | `models/production/` |

---

## 🧬 2. Organismos Nacidos por GAJE (`models/born/*.gaje`)

| Archivo | Estado Biológico | Tamaño | Rol y Descripción |
| :--- | :--- | :---: | :--- |
| 👑 **`max.gaje`** | **Organismo Conforme 2-Bit (Born Native)** | **11.39 MB** | **Insignia de Nacimiento Nativo**: Primer organismo nacido desde cero bajo la constelación cuaternaria $Q2\_0\_CONFORMAL$ (8 bloques, 256 dim, $A,C,G,T$). Cabe en caché L3 y opera a $38+$ tok/s sin FP32. |
| 🧬 **`feto_genomico_v1.gaje`** | **Feto / Embrión en Desarrollo** | **2.08 GB** | **Organismo Genómico en Desarrollo**: Modelo nacido con arquitectura nativa `gaje_native` (2 bloques, 768 dim) en etapa embrionaria. Sirve como banco de pruebas para algoritmos de mutación y destilación. |

---

## ⚡ 3. Modelos Transmutados en Producción (`models/production/*.flat`)

| Archivo en Producción | Arquitectura Base | Peso HD | RAM Residente (mmap) | Cuantización | Rol y Especialidad |
| :--- | :--- | :---: | :---: | :---: | :--- |
| 👑 **`qwen2_5_3b.flat`** | **Qwen2.5-3B-Instruct** | **2.24 GB** | **~1.7 GB - 2.5 GB** | `Q4_0` + `Q8_0` | **Insignia General Multilingüe**: Modelo principal predeterminado. Excelente fluidez en 29+ idiomas, código y síntesis. |
| 🧠 **`deepseek_r1_1_5b.flat`** | **DeepSeek-R1-Distill-1.5B** | **1.23 GB** | **~475 MB - 1.4 GB** | `Q4_0` + `Q8_0` | **Razonamiento Máximo (CoT)**: Monólogo interno (`<think>`), deducción paso a paso y resolución lógica. |
| ⚡ **`qwen2_0_5b.flat`** | **Qwen2-0.5B-Instruct** | **499 MB** | **~350 MB - 500 MB** | `Q4_0` + `Q8_0` | **Micro-Modelo Ultrarrápido**: Respuestas de baja latencia para entornos de recursos limitados. |
| 🧬 **`smollm2_135m.flat`** | **SmolLM2-135M-Instruct** | **474 MB** | **~180 MB - 300 MB** | `Q4_0` + `FP32` | **Nano-Agente Edge**: Dispositivos embebidos, IoT y pruebas de compresión genómica. |
