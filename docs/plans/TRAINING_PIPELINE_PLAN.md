# 🧬 PLAN: Pipeline de Entrenamiento Completo GAJE

**Versión:** 1.0 (Agosto 2026)
**Estatus:** Plan de trabajo (roadmap de ingeniería)
**Objetivo:** Un pipeline end-to-end, reproducible en cualquier corpus/modelo: destilación + QAT/IQAT del cuerpo + evaluación. No solo lm_head.

---

## 🧭 1. Estado Actual (punto de partida)

- **Inferencia**: motor Rust (mmap `.gaje.flat`, reader/writer GGUF) listo y funcional.
- **lm_head FP32 SFT**: ya entrena (fix en `refine_with_grads_core`, commit `76a2066`). Loss baja (1.94 → 0.59 en 6 épocas).
- **Cuerpo Q4_0/Q8_0**: es **read-only** para entrenar. `database_mut()` hace `panic!("Q4_0 is read-only")`. Solo las variantes 2/4-bit actualizan centroides.
- **Destilación**: MVP 1-a-1 a nivel de texto existe (`scripts/distill_smollm2_1teacher.py`), pero sobreajusta a los prompts (solo cabeza).

## 🚨 2. Cuello de botella / Ruta Crítica

**Fase A — IQAT/QAT real del cuerpo (Q4_0/Q8_0)** es lo que falta y lo más difícil. Sin ella no hay "pipeline completo": hoy solo se entrena la cabeza, por eso el modelo sobreajusta y repite el prompt.

## 🗺️ 3. Fases y Estimaciones (1 dev, full-time)

| Fase | Trabajo | Estimación | ¿Obligatoria? |
|------|---------|-----------|---------------|
| **A** | QAT/IQAT del cuerpo Q4_0/Q8_0: gradiente a través de dequantización, update de escalas/centroides, estabilidad numérica, memoria para gradientes del cuerpo. | **3-6 semanas** | Sí (crítica) |
| **B** | Data/entrenamiento configurable: cargar cualquier corpus (no `CONTROL_PROMPTS` hardcodeado), config/seed/CLI reproducible. | 1-2 semanas | Sí |
| **C** | Evaluación: PPL en held-out real, muestras de generación, métrica de calidad. | 1-2 semanas | Sí |
| **D** | Calidad de destilación: mejorar maestro/respuestas, quizá multi-teacher, evitar overfitting a prompts. | 2-4 semanas | No (mejora) |
| **Integración** | Hardening + tests del pipeline completo. | 1-2 semanas | Sí |

**Total estimado: ~8-15 semanas** de trabajo efectivo. El rango es amplio porque la Fase A puede alargarse si la estabilidad numérica da guerra. El calendario real depende de full-time vs. parcial.

## 🎯 4. Orden recomendado

1. **Fase A** → QAT del cuerpo (bloqueante).
2. **Fase B** → reproducibilidad con datos arbitrarios.
3. **Fase C** → evaluación real (held-out).
4. **Fase D** → calidad (opcional).

## 🧪 5. Criterios de "funcional" (Definition of Done)

- El pipeline entrena (no solo lm_head) el cuerpo Q4_0/Q8_0 sin `panic!` y sin no-ops silenciosos.
- Reproducible: mismo corpus + config + seed ⇒ mismos resultados.
- Evaluación sobre datos no vistos (PPL held-out) y muestras de generación coherentes, no solo los 5 prompts de control.
- Los pesos del cuerpo realmente cambian tras el entrenamiento (test unitario `before != after`).
