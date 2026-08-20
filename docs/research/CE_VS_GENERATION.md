# CE ≠ Calidad de Generación: Un Hallazgo Empírico

> **Estado**: Cerrado (negativo documentado). Fuente: `docs/meta/EMPIRICAL_TRUTH_STATE.md`,
> Fase 4b. Fecha: 2026-08.

## Contexto

Durante el desarrollo del pipeline de entrenamiento IQAT del cuerpo cuantizado
(Genomic4Bit Q4_0) de GAJE, la cross-entropía media (CE) sobre el corpus de
entrenamiento se usó como señal de progreso. Tras varios experimentos se observó
una contradicción: el modelo con la **mayor baja de CE** producía la **peor
generación**.

## El Hallazgo

Matriz medida con la misma tokenización (`eval_ce_core`) y con el harness
generativo objetivo (`eval_generation.py`: distinct-1/2, repetición de bigramas,
% de respuestas degeneradas):

| Corpus | Base CE | Train CE | Δ CE | Generación (harness) |
|:---|:---:|:---:|:---:|:---|
| distill (1520 tok, lm_head frozen) | 1.54 | 1.46 | −0.08 | **mejor**: d1/d2 máx, 0% degeneradas |
| corpus_unified (2.9k tok .txt) | 2.99 | 3.11 | +0.12 | degenerada |
| dataset_1000 (30k tok .txt) | 4.88 | 3.83 | **−1.05** | degenerada |
| clean (22k tok, per-secuencia) | 2.83 | 2.77 | −0.06 | degenerada |

La mayor caída de CE (−1.05) correspondió a la generación peor ("Por,", "Boriga").
La adaptación menor (−0.08, destilación corta con lm_head congelado) fue la única
que mejoró la generación.

## Causa raíz identificada

El objetivo de entrenamiento original era el CE medio sobre un **stream
concatenado** de parejas prompt+respuesta sin separadores. Ese objetivo premia
continuar el ruido local de concatenación, no la capacidad generativa. La CE mide
la probabilidad del siguiente token dado el contexto; un modelo puede lograr CE
baja **memorizando el corpus** o **colapsando a un atractor de alta probabilidad**
sin ganar diversidad, coherencia de largo plazo ni factualidad.

## Implicaciones (principios rectores)

1. **La CE es métrica AUXILIAR; la capacidad generativa es la métrica de ÉXITO.**
   Optimizar el CE de un stream ruidoso puede empeorar el producto real.
2. Toda decisión de optimización futura (cuantización, fine-tuning, IQAT,
   ambigüedad de CosSim) debe evaluarse con el harness generativo, no solo con CE.
3. **La evaluación debe ser generativa y fija** (prompts idénticos, temp 0.4 +
   greedy, métricas agregadas), no con ejemplos sueltos (corrigió dos juicios
   previos basados en cherry-picking).

## Lecciones para el futuro

- Entrenar **per-secuencia** (cache reseteado por pareja) en vez de stream
  concatenado: baja la CE base del corpus limpio de 4.53 a 2.83 y es ~150× más
  rápido (22k tokens en ~2 min).
- **Fine-tune grande del cuerpo cuantizado degrada la generación** en todos los
  experimentos probados (corpus, escala, lr 5e-5–2e-4, 8–24 bloques); la
  regularización KL (β=0.1 y 1.0) no lo evita.
- El único régimen que mejoró la generación fue la **destilación corta con lm_head
  congelado**, y aun así solo en el eje de **fluidez**: ese mismo modelo no
  generaliza a preguntas fuera de su corpus ("capital de México" → "capital de
  España"). Diversidad y ausencia de degeneración **no** implican factualidad en
  preguntas nuevas.

## Cierre

El mecanismo de backprop del cuerpo cuantizado está **validado numéricamente**
(gradient check f64, rel_err ≈ 0.06) pero **contraindicado** para la calidad
generativa en las condiciones probadas. La vía de producto es: **preservar el
conocimiento del maestro** (inferencia con 1.5B/3B) y usar destilación controlada
sobre corpus limpios, evaluando siempre con `eval_generation.py`.