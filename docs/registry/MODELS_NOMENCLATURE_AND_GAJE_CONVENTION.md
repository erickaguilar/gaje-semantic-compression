# 🧬 GAJE Nomenclatura Oficial de Modelos y Tabla de Equivalencias

> **Versión del Estándar:** GAJE Genomic Runtime v0.9.8  
> **Ubicación en Producción:** `models/production/*.gaje`  
> **Formato de Carga:** Zero-Copy Flat Memory Map (`mmap`) con aceleración nativa SIMD AVX2/FMA en Rust 2021.

---

## 📋 1. Tabla Maestra de Nomenclatura y Equivalencias

| Nombre Canónico GAJE | Nombre Técnico Original | Arquitectura Base | Peso HD | RAM Residente | Tipo Cuantización | Rol y Especialidad |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| 🧠 **`maximo.gaje`** | `deepseek_r1_1_5b_q4_0_q8_0_embd.gaje.flat` | **DeepSeek-R1-Distill-Qwen-1.5B** | **1.23 GB** | **~475 MB - 1.4 GB** | `Q4_0` (Cuerpo) + `Q8_0` (Embd/Head) | **Razonamiento Máximo (CoT)**: Lógica formal, matemáticas, código, deducción y pensamiento profundo. |
| 🌐 **`pro.gaje`** | `qwen2_5_3b_q4_0_q8_0_embd.gaje.flat` | **Qwen2.5-3B-Instruct** | **2.24 GB** | **~1.7 GB - 2.5 GB** | `Q4_0` (Cuerpo) + `Q8_0` (Embd/Head) | **Capacidad Pro Multilingüe**: Lenguaje general, redacción avanzada, síntesis multilingüe (ES, EN, ZH, RU, DE, FR). |
| ⚡ **`turbo.gaje`** | `qwen2_0_5b_q4_0_q8_0_embd.gaje.flat` | **Qwen2-0.5B-Instruct** | **499 MB** | **~350 MB - 500 MB** | `Q4_0` (Cuerpo) + `Q8_0` (Embd/Head) | **Micro-Modelo Ultrarrápido**: Respuestas instantáneas de baja latencia en entornos de memoria limitada. |
| 🧬 **`nano.gaje`** | `smollm2_4bit_clean.gaje.flat` | **SmolLM2-135M-Instruct** | **474 MB** | **~180 MB - 300 MB** | `Q4_0` (Cuerpo) + `FP32` (Embeddings) | **Nano-Agente Edge**: Dispositivos embebidos, IoT, pruebas de compresión genómica extrema y micro-controladores. |

---

## 🔬 2. Ficha Técnica Detallada de Modelos

### 🧠 A. `maximo.gaje`
* **Identificador Canónico:** `maximo.gaje`
* **Modelo Origen:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
* **Plantilla de Chat:** ChatML con Trigger de Pensamiento Profundo:
  ```text
  <|im_start|>system\nEres un asistente experto y preciso que responde en español.<|im_end|>\n<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n
  ```
* **Dimensiones de Embedding:** `1536`
* **Capas Transformer:** `28`
* **Cabezas de Atención (Q / KV):** `12` / `2` (GQA - Grouped Query Attention)
* **Rotary Embedding (RoPE Base):** `1,000,000.0`
* **Algoritmo de Inferencia:** Cuantización híbrida 4-bit AVX2 con kernel `q4_0_dot` y de-cuantización `Q8_0` en tiempo real.

---

### 🌐 B. `pro.gaje`
* **Identificador Canónico:** `pro.gaje`
* **Modelo Origen:** `Qwen/Qwen2.5-3B-Instruct`
* **Plantilla de Chat:** ChatML estándar:
  ```text
  <|im_start|>system\nYou are a helpful and precise assistant.<|im_end|>\n<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n
  ```
* **Dimensiones de Embedding:** `2048`
* **Capas Transformer:** `36`
* **Cabezas de Atención (Q / KV):** `16` / `2` (GQA)
* **Vocabulario:** `151,936` tokens

---

### ⚡ C. `turbo.gaje`
* **Identificador Canónico:** `turbo.gaje`
* **Modelo Origen:** `Qwen/Qwen2-0.5B-Instruct`
* **Plantilla de Chat:** ChatML estándar.
* **Dimensiones de Embedding:** `896`
* **Capas Transformer:** `24`
* **Cabezas de Atención (Q / KV):** `14` / `2` (GQA)

---

### 🧬 D. `nano.gaje`
* **Identificador Canónico:** `nano.gaje`
* **Modelo Origen:** `HuggingFaceTB/SmolLM2-135M-Instruct`
* **Plantilla de Chat:** ChatML estándar.
* **Dimensiones de Embedding:** `576`
* **Capas Transformer:** `30`
* **Cabezas de Atención (Q / KV):** `9` / `3` (GQA)

---

## 🛠️ 3. Recetas de Generación de 1 Sola Línea

Para regenerar cualquiera de estos organismos directamente desde sus fuentes GGUF hacia el formato binario `.gaje`:

```bash
# 1. Regenerar maximo.gaje (DeepSeek-R1 1.5B)
python3 scripts/export_gaje_flat.py --input data/models/deepseek-r1-distill-qwen-1.5b-q8_0.gguf --output models/production/maximo.gaje --quant-embed

# 2. Regenerar pro.gaje (Qwen 2.5 3B)
python3 scripts/export_gaje_flat.py --input data/models/qwen2.5-3b-instruct-q8_0.gguf --output models/production/pro.gaje --quant-embed

# 3. Regenerar turbo.gaje (Qwen 2 0.5B)
python3 scripts/export_gaje_flat.py --input data/models/qwen2-0_5b-instruct-fp16.gguf --output models/production/turbo.gaje --quant-embed

# 4. Regenerar nano.gaje (SmolLM2 135M)
python3 scripts/export_gaje_flat.py --input data/models/smollm2-135m-instruct-fp16.gguf --output models/production/nano.gaje
```

---

## 🔒 4. Verificación de Integridad Criptográfica (SHA-256)

```text
7c2c19e836109df44bcf84ecad9cba157c166d1fcaaeefc464ef6db0026e633d  models/production/maximo.gaje
e20ec4bf79c6d4ba40e0bc8ae92ff9fb172c72b2dd2bbcefa042533b3a39e31d  models/production/pro.gaje
507f35213606f7df2b6b553c1537233f81e370a4a838520ec719ce3f9b231ff6  models/production/turbo.gaje
fca97beeaeb3bfa8ba2061b47fb5d58d929ca32fbcf2b55f17d36371fc5bb290  models/production/nano.gaje
```
