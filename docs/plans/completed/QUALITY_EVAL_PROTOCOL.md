# Protocolo de Evaluación de Calidad — Fine-tuning del Cuerpo Q4_0

> Fecha: 2026-08-16 · Rama: `develop`
> Objetivo: definir **qué medimos y cómo** ANTES de la corrida con corpus grande, para
> no repetir el patrón "corrida larga → resultado ambiguo → definir métricas después".
> Complementa a `docs/plans/QUALITY_EXPORT_PLAN.md`.

---

## 1. Por qué existe este protocolo

El milestone de Vía B certificó el **mecanismo** (backward + generalización held-out CE
2.559→2.427). Pero la CE agregada favorable **no implica calidad de generación
perceptible** (lección aprendida con Qwen2.5-1.5B). Se validó la métrica de loss, no el
producto. Este protocolo hace explícito **qué constituye "calidad" de forma medible**
antes de gastar tiempo de cómputo en el corpus grande.

## 2. Dos afirmaciones separadas (no confundir)

| Afirmación | Métrica | Estado |
|:---|:---|:---|
| **A. El cuerpo puede aprender y generalizar** | CE/PPL held-out | Validada (2.427) |
| **B. El modelo entrenado genera texto útil** | Calidad de generación | No confirmada |

Este protocolo mide **B** y, de paso, **preservación de conocimiento** (que el
entrenamiento no destruya la capacidad preexistente).

## 3. Métricas

### 3.1 Preservación de la capacidad (regresión de conocimiento)

Comparar `modelo base` vs `modelo entrenado` sobre prompts **que NO están en el corpus
de entrenamiento**:

- **CE / PPL held-out** sobre el corpus de entrenamiento (medida de adaptación).
- **KL divergence** `KL(P_base ‖ P_trained)` sobre un conjunto OOD: cuánto se aleja el
  entrenado del base en distribución. **KL baja → preserva; KL alta → destruye
  conocimiento**, incluso si la CE held-out baja.
- **Facts check**: respuestas a hechos factuales (capitales, fechas, aritmética,
  traducciones, pregunta desconocida). Correcta / incorrecta / evasiva.

### 3.2 Calidad de generación

- **Coherencia** (escala 0-3): el texto es gramatical y no se atasca en un lazo.
- **Repetitividad** (métrica): proporción de n-gramas repetidos por encima de un umbral.
- **A/B ciego** contra referencia: ver §4.

## 4. Protocolo A/B ciego (extendido del usado con Qwen2.5)

### 4.1 Modelos a comparar

- `base` = `smollm2_4bit.gaje.flat` (sin entrenar).
- `entrenado` = export de calidad (candidato).
- `ref` = referencia de calidad alta para anclar la escala (HuggingFace, p.ej.
  `HuggingFaceTB/SmolLM2-135M-Instruct`), usada **solo como ancla de la rúbrica**.

### 4.2 Prompts de control (fijos, 3 categorías × 5 prompts)

| Categoría | Ejemplos |
|:---|:---|
| Facto en el corpus | contenido literal del corpus (debe reproducirse) |
| Facto OOD | capitales, aritmética, traducción, fecha (debe preservarse) |
| Abierta | "escribe un poema sobre el mar", "explica la fotosíntesis" |

Regla: los prompts OOD y abiertos **nunca** entran en el corpus de entrenamiento.

### 4.3 Muestreo

- Fijar `temperature` (0.2 y 0.8), `top_p=0.9`, `rep_penalty=1.1`, `max_new_tokens=64`.
- 3 semillas por (prompt, temp) → 3 respuestas por modelo.
- Los evaluadores ven las respuestas **sin saber qué modelo las generó**.

### 4.4 Rúbrica por prompt

1. **Correcta/Coherente** (factuales y OOD).
2. **Repetitividad** detectada (lazo "…es un idioma claro…" = fallo).
3. **Relevancia** al prompt (no gibberish genérico).

Escala Likert 1-5 por (prompt, modelo); se reporta media y mediana por categoría.

## 5. Criterios de aceptación para el export de calidad

| Criterio | Umbral |
|:---|:---|
| CE held-out | < 2.559 (baseline); ideal < 2.427 |
| KL(base‖trained) OOD | sin explosión (si CE baja y KL sube mucho → solo adaptación, no preservación) |
| Facts check OOD | al menos 2/3 de los factuales OOD siguen correctos |
| Coherencia | media ≥ 2 en escala 0-3 |
| Repetitividad | sin lazo detectable en ≥ 4/5 prompts abiertos |

## 6. Implementación mínima

1. `examples/eval_quality.rs` (Rust): carga base + entrenado, computa CE/PPL held-out y
   KL(P_base‖P_trained) sobre prompts OOD. Reutiliza `forward_with_hidden_core` y la
   distribución softmax de logits.
2. `scripts/eval_ab_blind.py`: genera las 3 respuestas por (prompt, temp, semilla) con
   ambos modelos (vía Web UI `server.py` o `GenomicLLM.generate`), baraja y escribe el
   fichero de evaluación ciego.
3. Resultados se registran en `docs/meta/EMPIRICAL_TRUTH_STATE.md`.

## 7. Secuencia de uso

1. **Primero** este protocolo (ya definido aquí).
2. **Después** el export de calidad (corpus grande + lm_head congelado + lr bajo).
3. Correr `eval_quality` + `eval_ab_blind` → llenar la rúbrica.
4. Solo si B se confirma, considerar el producto de especialización.