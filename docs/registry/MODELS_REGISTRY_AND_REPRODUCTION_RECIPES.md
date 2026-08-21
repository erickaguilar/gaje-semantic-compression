# 📜 Certificado de Existencia y Recetario de Reproducción de Modelos GAJE

**Fecha de Registro:** 2026-08-21  
**Versión del Ecosistema:** GAJE Native Runtime v0.9.8  
**Repositorio:** `gaje-semantic-compression` (Branch: `develop`)

---

## 🏛️ 1. Registro Criptográfico de Existencia (SHA-256 Hashes)

Este registro certifica la creación, verificación y hash criptográfico exacto de todos los modelos genómicos binarios planos (`.gaje.flat`) producidos durante las Fases 1 a 4 del proyecto:

| Modelo / Archivo | Tamaño | SHA-256 Checksum | Estado |
| :--- | :---: | :--- | :---: |
| `deepseek_r1_1_5b_q4_0_q8_0_embd.gaje.flat` | 1.26 GB | `d533351237a8d4aa6a8e19a8d7034b7cd162d4c8c3ebee9ba543fa4a05c875b6` | 🟢 **PRODUCCIÓN (Principal)** |
| `qwen2_5_3b_q4_0_q8_0_embd.gaje.flat` | 2.30 GB | `e922eb338317a4b183e401bdeb39629bdc9669f6238deb8f9dd160f5c61ac056` | 🟢 **PRODUCCIÓN (Max Capacidad)** |
| `qwen2_0_5b_q4_0_q8_0_embd.gaje.flat` | 499 MB | `a59f3860b65ea1e9e61b16e955229c99aebdc6d0b30db41aec4c9847421d7f37` | 🟢 **PRODUCCIÓN (Micro)** |
| `smollm2_4bit_clean.gaje.flat` | 474 MB | `17f3a74a45f888488e78efa119c8eba657d731ce13abf0cbcbfc79fbb14c8c01` | 🟢 **PRODUCCIÓN (Nano Edge)** |
| `qwen2_5_1_5b_q4_0_q8_0_embd.gaje.flat` | 1.30 GB | `d88688ca9c159a99f50baa8ea5fdf2011df544f4038e350555a7c15ec89fefe9` | 🟡 *Archivado (Reemplazado por DeepSeek-R1)* |
| `qwen2_5_1_5b_q4_0.gaje.flat` | 2.60 GB | `ab3e70e4ced269bfaf426419eab46f24e14c6bd18211626c77816aa23f2d63bd` | 🟡 *Archivado (Embeddings FP32 redundante)* |
| `qwen2_0_5b_q2_0.gaje.flat` | 1.20 GB | `05c967f305fd0507adb11eebdce5af0f049aaa2da01114888ac15d38c77a1646` | 🔴 *Experimento Q2_0 (H2 Rechazada)* |
| `qwen2_0_5b_q2_0_q8_0_embd.gaje.flat` | 414 MB | `3e5bf11d94dd4b22f26dda614dd442e9dc985467b8738998b6afe80e388abfa5` | 🔴 *Experimento Q2_0 (H2 Rechazada)* |
| `qwen2_0_5b_qualityA.gaje.flat` | 494 MB | `7c216ab8bf2e86790541b8530c2e56c1296800da26e2e1242f6d6865ce2ce5c3` | 🔵 *Checkpoint de Calibración* |
| `qwen2_0_5b_qualityB.gaje.flat` | 494 MB | `923c1498752def8b971582d3974545e9ae685a47192cdc325442ecbe59ef20cc` | 🔵 *Checkpoint de Calibración* |
| `smollm2_4bit.gaje.flat` | 472 MB | `fb5c6fc401078467f092263ecd647cb09a9b94758b3d036b6d538f363fc540bd` | 🔵 *Checkpoint Base* |
| `smollm2_4bit_quality_big.gaje.flat` | 474 MB | `9e1f3c8cb5e435495e8746d8d03dee08009b7e3d6410563d3fdedbba50474e10` | 🔵 *Checkpoint DNI Batch* |
| `smollm2_4bit_quality.gaje.flat` | 474 MB | `0bb3407d21ba105b50905d1052dc7e158b55544a7bcde531dce4a8f2536a4ae2` | 🔵 *Checkpoint Calibración* |
| `smollm2_4bit_quality_kl.gaje.flat` | 474 MB | `31dd0da1c3fb308c3bbb745c52784c0d41459b6e0fbd765e7cedbef66d80233b` | 🔵 *Checkpoint KL-Divergence* |
| `smollm2_4bit_trained.gaje.flat` | 1.10 GB | `0148bf686a40861b69dfea7ca7804cab36e96494d7b89aeed11dc5c28bda642c` | 🔵 *Checkpoint DNI 3000 Pasos* |

---

## 🍳 2. Recetario de Reproducción (Cómo Recrear Cualquier Modelo)

Cualquier modelo puede ser regenerado en minutos descargando el GGUF fuente desde HuggingFace y ejecutando el exportador nativo universal `scripts/export_gaje_flat.py`.

### 🥇 Receta 1: DeepSeek-R1-Distill-1.5B (Razonamiento CoT)
```bash
# 1. Descargar GGUF oficial
wget -c https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf \
  -O models/source/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf

# 2. Transmutar a formato plano GAJE con embeddings Q8_0
python3 scripts/export_gaje_flat.py \
  --input models/source/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf \
  --output models/production/deepseek_r1_1_5b_q4_0_q8_0_embd.gaje.flat \
  --quant-embed

# 3. Limpiar origen
rm models/source/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf
```

---

### 🥈 Receta 2: Qwen2.5-3B-Instruct (Capacidad Máxima)
```bash
# 1. Descargar GGUF oficial
wget -c https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf \
  -O models/source/qwen2.5-3b-instruct-q4_k_m.gguf

# 2. Transmutar a formato plano GAJE
python3 scripts/export_gaje_flat.py \
  --input models/source/qwen2.5-3b-instruct-q4_k_m.gguf \
  --output models/production/qwen2_5_3b_q4_0_q8_0_embd.gaje.flat \
  --quant-embed

# 3. Limpiar origen
rm models/source/qwen2.5-3b-instruct-q4_k_m.gguf
```

---

### 🥉 Receta 3: Qwen2-0.5B-Instruct (Micro-Modelo Rápido)
```bash
# 1. Descargar GGUF oficial
wget -c https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_0.gguf \
  -O models/source/qwen2-0_5b-instruct-q4_0.gguf

# 2. Transmutar a formato plano GAJE
python3 scripts/export_gaje_flat.py \
  --input models/source/qwen2-0_5b-instruct-q4_0.gguf \
  --output models/production/qwen2_0_5b_q4_0_q8_0_embd.gaje.flat \
  --quant-embed

# 3. Limpiar origen
rm models/source/qwen2-0_5b-instruct-q4_0.gguf
```

---

### 🔬 Receta 4: SmolLM2-135M-Instruct (Nano-Agente Edge)
```bash
# 1. Descargar GGUF oficial
wget -c https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct-GGUF/resolve/main/smollm2-135m-instruct-q4_k_m.gguf \
  -O models/source/smollm2-135m-instruct-q4_k_m.gguf

# 2. Transmutar a formato plano GAJE
python3 scripts/export_gaje_flat.py \
  --input models/source/smollm2-135m-instruct-q4_k_m.gguf \
  --output models/production/smollm2_4bit_clean.gaje.flat

# 3. Limpiar origen
rm models/source/smollm2-135m-instruct-q4_k_m.gguf
```

---

## 🎯 3. Criterio de Selección del Cuarteto de Producción

Los 4 modelos que se mantienen en el disco local cubren el 100% de los casos de uso:

1. **`deepseek_r1_1_5b_q4_0_q8_0_embd.gaje.flat` (1.26 GB):** Resuelve razonamiento abstracto, CoT, álgebra y lógica profunda.
2. **`qwen2_5_3b_q4_0_q8_0_embd.gaje.flat` (2.30 GB):** Máximo vocabulario y conocimiento enciclopédico en español.
3. **`qwen2_0_5b_q4_0_q8_0_embd.gaje.flat` (499 MB):** Ideal para pruebas de streaming y respuestas rápidas.
4. **`smollm2_4bit_clean.gaje.flat` (474 MB):** Base para entrenamiento de micro-adaptadores DNI y destilación local.
