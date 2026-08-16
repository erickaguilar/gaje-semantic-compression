# 🧬 IQAT Q4_0 del Cuerpo — Diseño

**Versión:** 0.1 (borrador, Agosto 2026)
**Estatus:** Diseño para discusión (Fase A del plan `TRAINING_PIPELINE_PLAN.md`)
**Objetivo:** Hacer entrenable el cuerpo Q4_0/Q8_0 (no solo el `lm_head`) de forma cuantización-aware, preservando el formato `.gaje.flat`.

---

## 🧭 1. Contexto y hallazgos del codebase

- **El cuerpo NO se entrena hoy.** `backward_core` (`src/nn/linear/backward.rs:10`) devolvía `vec![0.0; in_features]`: no había reverse-mode (backprop) a través del transformer. `train_on_sequence_core` y `train_step` solo refinan `lm_head`.
- **CORRECCIÓN CRÍTICA (verificado en modelo real): el cuerpo del `.flat` se guarda como `Genomic4Bit` (basado en centroides)**, NO como `GenomicQ4_0`. El update de `refine_with_grads_core` para `Genomic4Bit` muta `self.centroids` (no los bytes del database). Esa ruta **ya existía y funcionaba**; el problema era que **nunca se invocaba** en el pipeline (solo se refinaba `lm_head`). `bit_depth()=4` no distingue `Genomic4Bit` de `GenomicQ4_0` (ambos devuelven 4).
- **`IQATEngine` existente (`src/nn/iqat.rs`) era no-op**: llamaba `refine_with_grads_core(gate_gen/up_gen, ...)` con **wiring roto** (input `s_act_in` en vez de `x_ffn_n`, y `w_down` recibía `vec![0.0; gate.len()]`). `ffn_down`, `w_o` y atención nunca se tocaban.
- **`GenomicF32` ya entrena** (fix `76a2066`): la cabeza funciona. El bloqueador real es el cuerpo.
- **Bloques por capa** (`src/nn/block/mod.rs`): `attn.rmsnorm`, `q_gen`, `k_gen`, `v_gen`, `w_o` (atención) + `ffn_norm`, `gate_gen`, `up_gen`, `w_down` (FFN SwiGLU). Opcionales `fused_qkv`, `fused_gate_up`.
- **Q4_0Block** (`src/io/header/blocks.rs`): 32 pesos → `{scale f16, min f16, qs[16]}`. Dequant: `W = q·scale + min`, con `q ∈ [0,15]`. Layout row-major: `db[i*n_blocks + b]`. (Relevante para otros formatos/GGUF, no para el cuerpo `.flat` actual.)
- **Mutabilidad**: `database_mut()` hace `panic!("Q4_0 is read-only")`. Para entrenar hay que mutar vía `Arc::make_mut` dentro de `refine_with_grads_core` (patrón ya usado para `GenomicF32`).

## 🎯 2. Objetivo

Entrenar el cuerpo con dos mecánicas complementarias y escalonadas:

1. **QAT de escala/min (calibración in-flight)**: actualiza `scale`/`min` de bloques Q4_0 (y `scale` de Q8_0) manteniendo `q` fijo. Preserva la cuantización, mínimamente invasivo, seguro.
2. **Reverse-mode a través del cuerpo**: propaga gradientes desde la loss final (CE o drift) hasta los linears interiores, para poder aplicar el QAT sobre capas medias.

La etapa 1 (escala/min) es la más barata y de mayor valor inmediato; la etapa 2 (backward) es lo que la habilita de verdad end-to-end.

## 🧮 3. Diseño de la QAT de escala/min para Q4_0

Fórmulas (misma lógica LSQ/STE a nivel de bloque, manteniendo `q` fijo):

**Q4_0:** `W[i,j] = q[i,b,k]·scale[i,b] + min[i,b]`, con `b = j/32`, `k = j%32`.

Dado `grad_W[i,j] = grad_output[i]·input[j]` (gradiente del dot-product), acumulamos por bloque:

```
grad_scale[i,b] += Σ_{k∈[0,32)} grad_W[i, b·32+k] · q[i,b,k]
grad_min[i,b]   += Σ_{k∈[0,32)} grad_W[i, b·32+k]
```

Actualización:

```
scale[i,b] -= lr·grad_scale[i,b]
min[i,b]   -= lr·grad_min[i,b]
```

La escala se mantiene en f16 al escribirse; se acumula en f32 y se convierte. `q` no cambia → la cuantización se conserva (solo se recalibra la grilla).

**Q8_0:** `W = q8·scale`, solo escala:

```
grad_scale[i,b] += Σ_k grad_W[i, b·32+k] · q8[i,b,k]
scale[i,b] -= lr·grad_scale[i,b]
```

**STE:** como `q` es fijo, no hace falta STE sobre `q` en esta etapa. La etapa futura (re-cuantización de `q`) sí lo necesitaría.

## 🧮 4. Diseño del backward a través del cuerpo (la pieza que falta)

Dos vías, en orden de costo:

### Vía A — Drift local por sub-capa (rápida, primer hito)
Extender el `IQATEngine` existente para que capture activaciones del **maestro en cada frontera de sub-capa** (tras rmsnorm, tras atención, tras gate/up, tras w_down) y minimice la deriva por sub-capa. Esto da `grads` por sub-capa **sin** reverse-mode completo, y corrige el no-op actual añadiendo la rama Q4_0 en `refine_with_grads_core`.

**Ventaja:** se implementa y valida en ~1 iteración. **Limitación:** no es end-to-end (no propaga gradientes entre capas), pero desbloquea la QAT de escala/min de todo el cuerpo.

### Vía B — Reverse-mode completo (correcta, objetivo final)
Implementar backward (reverse-mode) guardando activaciones por token y propagando `dL/dx` desde la loss:

```
dL/dx  ← lm_head backward  (dW_lm_head = dL/dlogits ⊗ h_norm,  dL/dh = W_lm_head^T·dL/dlogits)
  → output_norm (RMSNorm backward)
  → por cada bloque invertido:
      residual: dL/dx += dL/dx_res; dL/dx_res = dL/dx_out
      FFN (w_down, gate, up): backprop a través de act_fn (SwiGLU)
      ffn_norm (RMSNorm backward)
      atención (w_o, softmax + máscara causal, v/k/q) + cache
      attn rmsnorm
```

Cada `GenomicLinear` expone:
- `forward` (ya existe)
- `backward(input, dL/dout) -> dL/dinput` (nuevo) que además acumula `grad_scale/min` en la rama Q4_0/Q8_0 (o en un buffer de grads).

Requiere añadir backward para: `RMSNorm`, `dot_product` (trivial: `dW = grad⊗input`, `dinput = W^T·grad`), `softmax`, `SwiGLU`, `add residuals`. Es el grueso del trabajo.

## 🏗️ 5. Arquitectura de cambios

| Módulo | Cambio |
|--------|--------|
| `src/nn/linear/backward.rs` | Añadir rama `GenomicQ4_0` y `GenomicQ8_0` en `refine_with_grads_core` (escala/min, ver §3). |
| `src/nn/linear/backward.rs` | Implementar `backward_core(input, dL/dout) -> dL/dinput` (hoy devuelve ceros). |
| `src/nn/linear/database.rs` | (opcional) helper de acceso mutable por bloque. |
| `src/nn/block/*` | Backward por sub-capa (Vía A) o reverse-mode de bloque (Vía B): rmsnorm, atención, FFN, residuals. |
| `src/nn/llm/forward.rs` | `train_on_sequence_core`: propagar grads a bloques, no solo a `lm_head`. |
| `src/nn/iqat.rs` | Corregir `refine_block_drift` (grads correctos por sub-capa) y cubrir `w_down`, `w_o`, atención. |
| `src/nn/trainer.rs` | Exponer IQAT de cuerpo vía Python para el pipeline. |

## 🚀 6. Rollout por etapas (de-riesgo)

**Etapa 0 — QAT escala/min aislada (sin backward):** ✅ HECHO (`9383442`)
- Ramas `GenomicQ4_0`/`GenomicQ8_0` en `refine_with_grads_core` + `Q4_0Block::q_value`.
- Tests: `test_q4_0_scale_min_update`, `test_q8_0_scale_update` (scale/min cambian, output cambia).
- Nota: el cuerpo `.flat` usa `Genomic4Bit` (centroides), así que esta etapa aplica a formatos Q4_0/Q8_0 (GGUF), no al cuerpo actual.

**Etapa 1 — Backprop dentro del bloque + wiring del IQAT:** ✅ HECHO (en curso de commit)
- `backward_core` implementada (transpuesta `d_input = W^T·d_output` para F32/Q4_0/Q8_0). Test `test_backward_transpose_q4_0`.
- Corregido `refine.rs`: `w_down` recibe `ffn_out` real (no ceros); gradientes de gate/up correctos (SwiGLU).
- `IQATEngine::refine_block_drift` ahora llama al backprop del bloque corregido.
- **Test de integración en modelo real**: el cuerpo `Genomic4Bit` muta sus centroides tras refine.
- **Validación end-to-end pendiente**: requiere maestro/estudiante de la **misma familia** (mismo hidden dim y vocab) para el drift IQAT, p. ej. Qwen2.5-1.5B → Qwen2.5-0.6B, o full reverse-mode (Vía B).

**Etapa 2 — Reverse-mode completo (Vía B):**
- Backward end-to-end desde CE.
- Criterio: IQAT + SFT de cuerpo con gradientes verdaderos; pérdida held-out consistente con la señal.

**Etapa 3 — (futuro) Re-cuantización de `q` con STE** para ganancia de precisión adicional.

## ⚠️ 7. Riesgos

- **Estabilidad numérica** de la QAT de escala/min (escalas f16 pequeñas; clipping para evitar overflow/undeflow).
- **Memoria**: guardar activaciones para reverse-mode en mmap; mitigar con grad-checkpointing por bloque o backward online por token (ya se hace online en `train_step`).
- **Regresión de inferencia**: cambios de escala/min deben ser conservadores (lr pequeño en cuerpo, `block_lr_scale` ya existe).
- **No corromper formato**: los Q4_0 se mutan en memoria vía `Arc::make_mut`; el guardado debe reflejar los nuevos scale/min.

## ✅ 8. Definition of Done

- `refine_with_grads_core` actualiza scale/min de `GenomicQ4_0`/`GenomicQ8_0` (test unitario `before != after`).
- El cuerpo se entrena de verdad (no `_ => {}` no-op): loss del cuerpo baja.
- La prueba held-out mejora (prompts NO vistos dejan de ser gibberish), validando que la destilación ya no es solo memorización de lm_head.
- El pipeline reproducible del plan Fase A→C corre con cuerpo entrenable.