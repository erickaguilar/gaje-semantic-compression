# 🔬 Reporte de Validación de Paridad FP32: SmolLM2-135M

## 📋 Resumen Ejecutivo

Este documento detalla el procedimiento de audición capa por capa realizado sobre el motor nativo en Rust de **GAJE** para certificar la paridad exacta frente a la implementación de referencia de PyTorch HuggingFace.

---

## 🛠️ Modificaciones Arquitectónicas Clave

### 1. Persistencia de Pesos RMSNorm en PyO3
- **Archivo Afectado**: [`python/gaje/nn/stabilized.py`](file:///home/erickaguilar/Documentos/gaje-semantic-compression/python/gaje/nn/stabilized.py)
- **Problema**: `attn_norm`, `ffn_norm` y `output_norm` fallaban la verificación de atributos de Python en envolventes PyO3. Al no serializarse en `.gaje`, la reconstrucción asignaba arreglos unitarios (`1.0`), provocando una deriva multiplicativa de varianza a través de 30 capas hasta una norma final de $1.56 \times 10^{11}$.
- **Solución**: Se almacenaron las listas `attn_norm` y `ffn_norm` directamente en el wrapper de Python `GenomicTransformerBlock` y se serializaron como tensores de punto flotante en `save()`.

### 2. Concatenación de Cabezas de Atención Multi-Cabeza (GQA)
- **Archivo Afectado**: [`src/nn/attention.rs`](file:///home/erickaguilar/Documentos/gaje-semantic-compression/src/nn/attention.rs)
- **Problema**: El aplanamiento de cabezas mediante Rayon `flat_map` mantenía zancadas por cabeza `[N_heads, S_seq, D_head]` antes de la multiplicación por $W_o$.
- **Solución**: Inserción explícita de cada vector de cabeza en rangos contiguos por token `attn_out[h * head_dim .. (h+1) * head_dim]`.

---

## 📊 Resultados de la Evaluación (`scripts/gaje_diff.py`)

```text
======================================================
GAJE Validation Report (SmolLM2 FP32)
======================================================
✓ Engine Forward Parity: ✅ PERFECTA
  - Cosine Similarity:   1.000000
  - Logit MAE:           0.000010
  - Top-1 Agreement:     ✅ SÍ (HF=' Paris', GAJE=' Paris')
  - Top-5 Agreement:     5/5 (100.0%)
```
