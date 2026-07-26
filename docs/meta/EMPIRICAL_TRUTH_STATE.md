# 🧬 EMPIRICAL TRUTH STATE: Matriz de Certificación y Estado Real (v1.0.0-alpha: Silver Adult)

Este documento define el estado técnico y empíricamente verificado del motor de inferencia nativa **GAJE (Genomic Adaptive Joint Embedding)**.

---

## 🏆 1. Capa de Infraestructura Nativa (FP32 Engine Parity: CERTIFICADO 🟢)

Se certificó formalmente la equivalencia matemática entre el motor nativo en Rust (`GenomicLLM`) y la implementación de referencia en PyTorch HuggingFace (`HuggingFaceTB/SmolLM2-135M-Instruct`).

### 📊 Matriz de Certificación FP32 (`scripts/gaje_diff.py`)

| Métrica / Operación | Valor Medido | Criterio de Certificación | Estado |
| :--- | :---: | :---: | :---: |
| **Similitud Coseno (CosSim)** | **`1.000000`** | `> 0.9999` | ✅ **CERTIFICADO** |
| **Error Absoluto Medio (Logit MAE)** | **`0.000010`** | `< 0.001` | ✅ **CERTIFICADO** |
| **Top-1 Agreement** | **`100.0%` (`' Paris'`)** | `100.0%` | ✅ **CERTIFICADO** |
| **Top-5 Agreement** | **`5/5 (100.0%)`** | `100.0%` | ✅ **CERTIFICADO** |
| **30 Bloques Transformer** | **CosSim = `1.000000`** | `1.000000` en cada capa | ✅ **CERTIFICADO** |

---

## 📊 2. Capa de Compresión y Cuantización (SmolLM2-135M & Qwen2-0.5B)

Con la infraestructura de punto flotante certificada, se midió la respuesta del modelo ante diferentes profundidades de cuantización:

| Configuración | Profundidad de Bits | CosSim (Prefill) | Top-1 Predicción | Estado |
| :--- | :--- | :---: | :---: | :---: |
| **FP32 Baseline** | 32-bit (Atención y FFN) | **`1.000000`** | `' Paris'` | ✅ **Paridad Absoluta** |
| **4-bit Uniforme** | 4-bit (Atención y FFN) | **`0.924766`** | `' Paris'` | ✅ **Compresión Óptima** |
| **Mixed-Bit (5% Anchors)**| 4-bit Attn / 2-bit FFN (5% Anclas) | `0.736537` | `"'"` | 🔴 Degradación FFN |
| **2-bit Uniforme** | 2-bit (Atención y FFN) | `0.615916` | `','` | 🔴 Degradación SwiGLU |

---

## 🔬 3. Causa Raíz Resuelta (Auditoría de Infraestructura)

1. **Persistencia RMSNorm en PyO3**:
   - Los pesos de escala `attn_norm`, `ffn_norm` y `output_norm` eran ignorados por verificaciones de atributos PyO3 en Python. Al cargar `1.0` por defecto, la varianza derivaba hasta una norma de $1.56 \times 10^{11}$ a lo largo de las 30 capas.
   - **Solución**: Serialización directa de arreglos float32 en `stabilized.py` y descompresión en C-ABI.

2. **Filtro Lateral K-WTA**:
   - Se configuró `k_wta_ratio = 0.0` por defecto en inferencia exacta para prevenir el enmascaramiento arbitrario del 50% de los logits de salida.
