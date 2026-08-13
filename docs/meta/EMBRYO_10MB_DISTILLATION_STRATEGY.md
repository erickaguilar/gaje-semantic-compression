# 🧬 EMBRYO-10MB: Preservación de Ranking y Destilación desde el Motor Certificado

> **GAJE Helix · v1.6.0-alpha · Hallazgo empírico y propuesta estratégica**
> **Fecha:** 2026-08 · **Clasificación:** Meta / Estrategia de investigación

---

## 1. Objetivo

Documentar un hallazgo empírico del motor GAJE (`Q4_0` + `FP32`) y re-basar la línea de investigación del **embrión de ~10 MB** sobre una evidencia medible, en lugar de la narrativa legacy de cuantización 2-bit (congelada por inviabilidad en hardware comercial).

La pregunta de fondo: *si el modelo elige el token correcto en su propio sistema (su distribución interna es autoconsistente), ¿podemos usar eso para desarrollar un embrión de 10 MB?*

---

## 2. Hallazgo empírico: conserva decisiones, no valores

Medición de **perplejidad diferencial (PPL)** del modelo Qwen2-0.5B en formato `.gaje.flat` (`Q4_0` cuerpo + `FP32` embeddings) contra el mismo checkpoint en **FP16** (HuggingFace), sobre corpus en español.

### 2.1 Instrumento reproducible

```
scripts/benchmarks/ppl_parity_fp16.py
```

- Maestro de referencia: `Qwen/Qwen2-0.5B-Instruct` (FP16, torch).
- Organismo: `models/production/qwen2_0_5b_q4_0_q8_0_embd.gaje.flat`.
- Metodología: log-verosimilitud desplazada (shifted log-likelihood), softmax sobre vocabulario completo (151646 tokens), corpus ES held-out filtrado de cabeceras/líneas raras.

### 2.2 Resultados

| Medición | Muestras | `max_len` | PPL GAJE | PPL FP16 | Ratio | Correlación |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Exploración rápida | 12 | 64 | 46.97 | 37.89 | 1.24 | 0.933 |
| **Corpus limpio (confirmación)** | **120** | **128** | **102.73** | **58.49** | **1.76** | **0.875** |

### 2.3 Lectura honesta

- **La paridad absoluta de PPL NO se sostiene.** Con texto largo y corpus limpio, el cuerpo 4-bit cuesta ~**76% más de perplejidad** que FP16 (el error de cuantización se acumula en secuencias largas). La primera cifra (1.24) era optimista por muestras cortas y poco representativas.
- **La correlación 0.87–0.93 es robusta.** El motor 4-bit conserva el **orden de probabilidades** de FP16 aunque no los valores absolutos. Es decir: **mantiene la *decisión* (qué token es más probable), no la *valoración* (cuánto).**
- **Vocabulario intacto:** intersección completa de 151646/151646 tokens entre GAJE y HF → no hay colapso del vocabulario, confirmando el diseño FP32 en `token_embd`/`lm_head`.

**Conclusión del hallazgo:** GAJE Q4_0+FP32 produce un modelo cuya *función de decisión* (argmax sobre tokens) está fuertemente alineada con FP16, incluso cuando su distribución absoluta diverge. Esto explica empíricamente la calidad de razonamiento con verificación observada en los logs del 3B: no es paridad numérica, es **paridad de ranking**.

---

## 3. Por qué esto resuelve el embrión de 10 MB

La distinción "conserva decisiones, no valores" es la clave que faltaba para una ruta moderna y comprobable.

### 3.1 El motor certificado sirve como maestro de destilación

La **destilación de conocimiento** (Hinton et al.) funciona con *soft labels* — las probabilidades **relativas**, no las absolutas. Como GAJE conserva el ranking (r ≈ 0.87–0.93), sus logits son una señal de maestro **confiable**: el "dark knowledge" (qué token es más probable que cuál) está intacto, pese a que su PPL absoluta sea alta.

### 3.2 Distilación GAJE → embrión

Un embrión de ~10 MB (5–20M parámetros) no puede memorizar el conocimiento general de un 0.5B/3B, pero sí puede aprender a **reproducir sus decisiones** (el argmax) sobre un dominio acotado. Eso es exactamente lo que la preservación de ranking garantiza: el maestro decide bien, el alumno imita esa decisión.

### 3.3 Auto-consistencia como señal

"Elige el token correcto en su propio sistema" = calibrar por **auto-consistencia** (muestreo múltiple / cadena de razonamiento estable). Es el mismo mecanismo de "verificación" que ya exhibe el 3B. Puede usarse como *señal de recompensa* (RL / self-training) para el embrión, sin depender de etiquetas externas.

---

## 4. Re-base frente a la documentación legacy

| Dimensión | Enfoque legacy (2-bit genómico) | Enfoque fundamentado (propuesto) |
| :--- | :--- | :--- |
| Base | Narrativa especulativa ACGT, inviable en hardware comercial (congelada) | Evidencia medible: correlación de ranking 0.87–0.93 |
| Técnica | Cuantización extrema a 2 bits de 10M params | **Destilación de conocimiento desde el motor certificado Q4_0+FP32** |
| Señal de validación | No definida | Auto-consistencia del embrión (agreement de argmax) |
| Ruta de producción | Bloqueada | Comprable y testeable |

**La propuesta no descarta el valor conceptual de la visión legacy, pero la re-base técnica parte de un dato ya medido.**

---

## 5. Metodología propuesta (experimento)

### 5.1 Fase A — Cuantificar la fiabilidad del maestro (argmax agreement)
Medir qué porcentaje del tiempo el argmax de GAJE coincide con el de FP16 sobre un corpus variado.

- **Éxito:** agreement > 90% en dominio objetivo.
- **Instrumento:** extensión de `ppl_parity_fp16.py` (o script dedicado `argmax_agreement.py`).

### 5.2 Fase B — Destilación con soft labels y temperatura
Entrenar un embrión pequeño para reproducir las distribuciones (con temperatura) del maestro GAJE.

- Logits del maestro pasados por `softmax(·/T)` con `T > 1` para suavizar y transferir dark knowledge.
- **Éxito:** el embrión conserva la calidad de decisión del maestro en un dominio acotado, con footprint ≪ del maestro.

### 5.3 Fase C — Auto-consistencia como reward
Optimizar el embrión para que su cadena de razonamiento sea estable bajo perturbación (muestreo múltiple), usando la verificación interna como recompensa.

---

## 6. Riesgos y advertencias

- **Límite de conocimiento:** 10 MB no compite con un 0.5B en conocimiento general; apuntar a un **dominio vertical**, no a un razonador general.
- **Calibración del maestro:** la PPL absoluta del maestro es alta; destilar con **temperatura + soft targets**, no con probabilidades crudas.
- **Heredar limitaciones:** el embrión hereda las limitaciones de razonamiento del maestro (modestas ya en 0.5B).

---

## 7. Próximos pasos

1. Correr **Fase A** (argmax agreement) para fijar el umbral de fiabilidad del maestro.
2. Montar el pipeline de destilación (soft labels + temperatura) sobre un dominio acotado.
3. Integrar el hallazgo en `docs/reports/` y en el futuro `BENCHMARKS.md` oficial (con 3ª columna: un 4-bit total tipo llama.cpp, para demostrar que el delta FP32-embeddings es real).
4. Actualizar `INDEX.md` referenciando este documento.

---

## 8. Referencias

- Instrumento: `scripts/benchmarks/ppl_parity_fp16.py`
- Modelo probado: `models/production/qwen2_0_5b_q4_0_q8_0_embd.gaje.flat`
- Contexto vision legacy: `docs/meta/vision/` (SILVER_ADULT_ORGANISM_THEORY, TECHNICAL_BIRTH_AND_REALITY, VISION_AND_STRATEGY_2026)
- Reportes de paridad previos: `docs/reports/smollm2_fp32_parity.md`, `docs/reports/qwen2_distillation_report.md`