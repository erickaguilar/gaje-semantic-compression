# 📜 Certificado de Existencia y Recetario de Reproducción de Modelos GAJE

**Fecha de Registro:** 2026-08-21  
**Versión del Ecosistema:** GAJE Native Runtime v0.9.8  
**Repositorio:** `gaje-semantic-compression` (Branch: `develop`)

---

## 🏛️ 1. Registro Criptográfico de Existencia (SHA-256 Hashes)

Este registro certifica la creación, verificación y hash criptográfico exacto de todos los modelos genómicos binarios planos (`.gaje`) producidos en producción:

| Modelo / Archivo | Tamaño | SHA-256 Checksum | Estado |
| :--- | :---: | :--- | :---: |
| `maximo.gaje` (DeepSeek-R1 1.5B) | 2.45 GB | `f2e6e066ecc3c3da39137f515607b68a9e38ba9e2019fd3e8ee30787fb583206` | 🟢 **PRODUCCIÓN (Líder Razonamiento CoT)** |
| `pro.gaje` (Qwen 2.5 3B) | 2.24 GB | `e20ec4bf79c6d4ba40e0bc8ae92ff9fb172c72b2dd2bbcefa042533b3a39e31d` | 🟢 **PRODUCCIÓN (General Multilingüe 3B)** |
| `turbo.gaje` (Qwen 2 0.5B) | 499 MB | `507f35213606f7df2b6b553c1537233f81e370a4a838520ec719ce3f9b231ff6` | 🟢 **PRODUCCIÓN (Micro-Rápido 0.5B)** |
| `nano.gaje` (SmolLM2 135M) | 474 MB | `fca97beeaeb3bfa8ba2061b47fb5d58d929ca32fbcf2b55f17d36371fc5bb290` | 🟢 **PRODUCCIÓN (Nano Edge 135M)** |

---

## 🍳 2. Recetario de Reproducción (Cómo Recrear Cualquier Modelo)

Cualquier modelo puede ser regenerado en minutos descargando el GGUF fuente desde HuggingFace y ejecutando el exportador nativo universal `scripts/export_gaje_flat.py`.

### 🥇 Receta 1: DeepSeek-R1-Distill-1.5B (Razonamiento CoT Máximo)
```bash
# 1. Descargar GGUF oficial Q8_0
wget -c "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf" \
  -O models/source/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf

# 2. Transmutar a formato binario plano GAJE con embeddings Q8_0
python3 scripts/export_gaje_flat.py \
  --input models/source/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf \
  --output models/production/maximo.gaje \
  --quant-embed

# 3. Limpiar origen
rm -f models/source/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf
```

---

### 🥈 Receta 2: Qwen2.5-3B-Instruct (Capacidad General)
```bash
# 1. Descargar GGUF oficial
wget -c https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf \
  -O models/source/qwen2.5-3b-instruct-q4_k_m.gguf

# 2. Transmutar a formato binario plano GAJE con embeddings Q8_0
python3 scripts/export_gaje_flat.py \
  --input models/source/qwen2.5-3b-instruct-q4_k_m.gguf \
  --output models/production/pro.gaje \
  --quant-embed

# 3. Limpiar origen
rm -f models/source/qwen2.5-3b-instruct-q4_k_m.gguf
```

---

### 🥉 Receta 3: Qwen2-0.5B-Instruct (Micro Modelo Ultrarrápido)
```bash
# 1. Descargar GGUF
wget -c https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-fp16.gguf \
  -O models/source/qwen2-0_5b-instruct-fp16.gguf

# 2. Transmutar a GAJE
python3 scripts/export_gaje_flat.py \
  --input models/source/qwen2-0_5b-instruct-fp16.gguf \
  --output models/production/turbo.gaje \
  --quant-embed

# 3. Limpiar origen
rm -f models/source/qwen2-0_5b-instruct-fp16.gguf
```

---

### 🧬 Receta 4: SmolLM2-135M-Instruct (Nano-Agente Edge)
```bash
# 1. Descargar GGUF
wget -c https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct-GGUF/resolve/main/smollm2-135m-instruct-fp16.gguf \
  -O models/source/smollm2-135m-instruct-fp16.gguf

# 2. Transmutar a GAJE
python3 scripts/export_gaje_flat.py \
  --input models/source/smollm2-135m-instruct-fp16.gguf \
  --output models/production/nano.gaje

# 3. Limpiar origen
rm -f models/source/smollm2-135m-instruct-fp16.gguf
```
