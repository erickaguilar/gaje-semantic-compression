# 🧬 GAJE Nomenclatura Oficial del Ecosistema

> **Versión del Estándar:** GAJE Genomic Runtime v0.9.8  
> **Directorio de Modelos Nacidos por GAJE:** `models/born/*.gaje`  
> **Directorio de Modelos Transmutados:** `models/production/*.flat`  
> **Motor de Inferencia:** Zero-Copy Flat Memory Map (`mmap`) con aceleración nativa SIMD AVX2/FMA en Rust 2021.

---

## 🏛️ 1. Jerarquía y Filosofía de Extensiones

El ecosistema GAJE establece una distinción clara y estricta entre dos categorías de modelos:

| Extensión | Categoría | Definición y Origen | Ubicación |
| :---: | :--- | :--- | :--- |
| **`*.gaje`** | 🧬 **Organismos Nacidos por GAJE** | Modelos **entrenados, evolucionados o destilados genómicamente desde cero** mediante los algoritmos bio-inspirados de GAJE (matrices de ADN, tripletes de 2/4-bit, DNI y balance epigenético). | `models/born/` |
| **`*.flat`** | ⚡ **Modelos Externos Transmutados** | Modelos abiertos consolidados (Qwen, DeepSeek, SmolLM) cuantizados en `Q4_0`/`Q8_0` y adaptados al formato binario plano para inferencia ultrarrápida `mmap` zero-copy. | `models/production/` |

---

## 🧬 2. Organismos Nacidos por GAJE (`models/born/*.gaje`)

| Archivo | Arquitectura | Tamaño | SHA-256 Checksum | Rol y Especialidad |
| :--- | :--- | :---: | :--- | :--- |
| 🧬 **`gemma4_student.gaje`** | **`gaje_native` (Born-Genomic)** | **2.08 GB** | `ba4b0f767776fc3a17b621a16cbd5aae93bdb6fcde2948ddea820f7221ec4161` | **Estudiante Destilado Gemma 4**: Organismo genómico nativo nacido en GAJE con conocimiento asimilado del maestro multimodal `google/gemma-4-E2B-it`. |

---

## ⚡ 3. Modelos Transmutados en Producción (`models/production/*.flat`)

| Archivo en Producción | Arquitectura Base | Peso HD | RAM Residente (mmap) | Cuantización | Rol y Especialidad |
| :--- | :--- | :---: | :---: | :---: | :--- |
| 👑 **`qwen2_5_3b.flat`** | **Qwen2.5-3B-Instruct** | **2.24 GB** | **~1.7 GB - 2.5 GB** | `Q4_0` + `Q8_0` | **Insignia General Multilingüe**: Modelo principal predeterminado. Excelente fluidez en 29+ idiomas, código y síntesis. |
| 🧠 **`deepseek_r1_1_5b.flat`** | **DeepSeek-R1-Distill-1.5B** | **1.23 GB** | **~475 MB - 1.4 GB** | `Q4_0` + `Q8_0` | **Razonamiento Máximo (CoT)**: Monólogo interno (`<think>`), deducción paso a paso y resolución lógica. |
| ⚡ **`qwen2_0_5b.flat`** | **Qwen2-0.5B-Instruct** | **499 MB** | **~350 MB - 500 MB** | `Q4_0` + `Q8_0` | **Micro-Modelo Ultrarrápido**: Respuestas de baja latencia para entornos de recursos limitados. |
| 🧬 **`smollm2_135m.flat`** | **SmolLM2-135M-Instruct** | **474 MB** | **~180 MB - 300 MB** | `Q4_0` + `FP32` | **Nano-Agente Edge**: Dispositivos embebidos, IoT y pruebas de compresión genómica. |

---

## 🛠️ 4. Pipeline Completo de Destilación (Fase 1 ➔ Fase 3)

```bash
# Fase 1: Generación del corpus de conocimiento maestro
python3 scripts/generate_distill_corpus_gemma4.py

# Fase 2: Entrenamiento y nacimiento del estudiante genómico
python3 scripts/train_genomic_distill.py

# Fase 3: Activación en servidor web visual
python3 examples/ui/web_ui/server.py
```
