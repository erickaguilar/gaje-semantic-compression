# 🧬 IQAT Q4_0 del Cuerpo — Diseño

**Versión:** 0.1 (borrador, Agosto 2026)
**Estatus:** Diseño para discusión (Fase A del plan `TRAINING_PIPELINE_PLAN.md`)
**Objetivo:** Hacer entrenable el cuerpo Q4_0/Q8_0 (no solo el `lm_head`) de forma cuantización-aware, preservando el formato `.gaje.flat`.

---

## 🧭 1. Contexto y hallazgos del codebase

- **El cuerpo NO se entrena hoy.** `backward_core` (`src/nn/linear/backward.rs:10`) devuelve `vec![0.0; in_features]`: no hay reverse-mode (backprop) a través del transformer. `train_on_sequence_core` y `train_step` solo refinan `lm_head`.
- **`IQATEngine` existente (`src/nn/iqat.rs`) es también no-op para Q4_0**: usa *activation drift* por bloque y llama `refine_with_grads_core(gate_gen/up_gen, ...)`, pero esos linears son `GenomicQ4_0` que caen en `_ => {}` de `refine_with_grads_core`. `ffn_down`, `w_o` y atención nunca se tocan.
- **`GenomicF32` ya entrena** (fix `76a2066`): la cabeza funciona. El bloqueador real es el cuerpo.
- **Bloques por capa** (`src/nn/block/mod.rs`): `attn.rmsnorm`, `q_gen`, `k_gen`, `v_gen`, `w_o` (atención) + `ffn_norm`, `gate_gen`, `up_gen`, `w_down` (FFN SwiGLU). Opcionales `fused_qkv`, `fused_gate_up`.
- **Q4_0Block** (`src/io/header/blocks.rs`): 32 pesos → `{scale f16, min f16, qs[16]}`. Dequant: `W = q·scale + min`, con `q ∈ [0,15]`. Layout row-major: `db[i*n_blocks + b]`.
- **Q8_0Block**: `{scale f16, qs[i8;32]}`. Dequant: `W = q8·scale`.
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

**Etapa 0 — QAT escala/min aislada (sin backward):**
- Añadir ramas `GenomicQ4_0`/`GenomicQ8_0` en `refine_with_grads_core`.
- Test unitario: `test_q4_0_scale_min_update` — dado input+grads, verificar `scale`/`min` cambian y el output cambia.
- Criterio: `max|scale_before - scale_after| > 0`.
- **Entregable:** mecánica QAT validada, autónoma del backward. *Esto NO requiere backprop y es testeable ya.*

**Etapa 1 — Drift local por sub-capa (Vía A):**
- Corregir `IQATEngine` para pasar grads reales por sub-capa y cubrir todos los linears del bloque.
- Validar con la prueba held-out: el modelo debe empezar a **generalizar** prompts nuevos (el objetivo que lm_head-only no logró).
- Criterio: PPL held-out baja y salidas de prompts NO vistos dejan de ser gibberish.

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