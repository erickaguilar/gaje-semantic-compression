# 📜 Certificado de Existencia y Recetario de Reproducción de Modelos GAJE

**Fecha de Registro:** 2026-08-22  
**Versión del Ecosistema:** GAJE Native Runtime v0.9.8  
**Repositorio:** `gaje-semantic-compression` (Branch: `develop`)

---

## 🏛️ 1. Registro Criptográfico de Existencia (SHA-256 Hashes)

Este registro certifica la creación, verificación y hash criptográfico exacto de todos los modelos del ecosistema GAJE:

### A. Organismos Nacidos por GAJE (`models/born/*.gaje`)
| Modelo / Archivo | Tamaño | SHA-256 Checksum | Estado |
| :--- | :---: | :--- | :---: |
| `gemma4_student.gaje` | 2.08 GB | `ba4b0f767776fc3a17b621a16cbd5aae93bdb6fcde2948ddea820f7221ec4161` | 🟢 **BORN (Destilado de Gemma 4 E2B)** |

---

### B. Modelos Transmutados de Producción (`models/production/*.flat`)
| Modelo / Archivo | Tamaño | SHA-256 Checksum | Estado |
| :--- | :---: | :--- | :---: |
| `qwen2_5_3b.flat` | 2.24 GB | `e20ec4bf79c6d4ba40e0bc8ae92ff9fb172c72b2dd2bbcefa042533b3a39e31d` | 🟢 **PRODUCCIÓN (Insignia General Multilingüe)** |
| `deepseek_r1_1_5b.flat` | 1.23 GB | `97bb9dadcd27273c30c39b7ad7685343c291a312143f77c73267c6fb3f117693` | 🟢 **PRODUCCIÓN (Líder Razonamiento CoT)** |
| `qwen2_0_5b.flat` | 499 MB | `507f35213606f7df2b6b553c1537233f81e370a4a838520ec719ce3f9b231ff6` | 🟢 **PRODUCCIÓN (Micro-Rápido 0.5B)** |
| `smollm2_135m.flat` | 474 MB | `fca97beeaeb3bfa8ba2061b47fb5d58d929ca32fbcf2b55f17d36371fc5bb290` | 🟢 **PRODUCCIÓN (Nano Edge 135M)** |

---

## 🍳 2. Recetario de Reproducción (Cómo Recrear Cualquier Modelo)

### 🥇 Receta Destilación: Organismo Nacido Gemma 4 (`gemma4_student.gaje`)
```bash
# 1. Generar el corpus de conocimiento maestro
python3 scripts/generate_distill_corpus_gemma4.py

# 2. Entrenar y dar a luz al estudiante genómico
python3 scripts/train_genomic_distill.py
```

---

### 🥈 Receta Transmutación: Qwen2.5-3B-Instruct (`qwen2_5_3b.flat`)
```bash
# 1. Descargar GGUF oficial
wget -c https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf \
  -O models/source/qwen2.5-3b-instruct-q4_k_m.gguf

# 2. Transmutar a formato binario plano GAJE con embeddings Q8_0
python3 scripts/export_gaje_flat.py \
  --input models/source/qwen2.5-3b-instruct-q4_k_m.gguf \
  --output models/production/qwen2_5_3b.flat \
  --quant-embed

# 3. Limpiar origen
rm -f models/source/qwen2.5-3b-instruct-q4_k_m.gguf
```
