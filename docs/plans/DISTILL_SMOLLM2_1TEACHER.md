# Destilación 1-a-1 con estudiante SmolLM2 — MVP de prueba rápida

> Estado: **diseño + primer script funcional** (`scripts/distill_smollm2_1teacher.py`).
> Objetivo: validar el mecanismo de destilación GAJE con un **solo maestro** y un
> **estudiante SmolLM2** en un ciclo corto y medible, antes de escalar a multi-maestro.

## 1. Por qué SmolLM2 como estudiante

- Es el modelo más pequeño del ecosistema GAJE (`smollm2_4bit.gaje.flat`, ~494 MB).
- Carga `mmap` zero-copy casi instantánea → cada iteración del test es barata.
- Suficiente para comprobar que el estudiante **converge a algo coherente** (la
  disciplina que recomendamos antes de intentar multi-maestro).

## 2. Estrategia: Vía A (destilación de texto, offline)

La Vía A evita los tres problemas de las vías de logits en este hardware:

| Problema | Cómo lo evita la Vía A |
|---|---|
| Mapeo de vocabulario maestro→estudiante (Qwen 150k vs SmolLM2 49k) | Los textos se re-tokenizan en el tokenizer del estudiante; no hay `vocab_mapping`. |
| Cargar 2+ modelos grandes a la vez (swapping) | 1 modelo por fase: generar → guardar `jsonl` → liberar → entrenar. |
| Entrenar el cuerpo Q4_0 (frágil) | SFT solo sobre el `lm_head`, que en el formato híbrido es **FP32**. |

### Pipeline

```
[Fase 1] Maestro (.flat) ──genera──▶ data/distill/train_*.jsonl (prompt→answer)
[Fase 2] Estudiante SmolLM2 (.flat) ──SFT lm_head (CE)──▶ dataset = re-tokenizar(prompt+answer)
[Fase 3] Evaluación in-place: control prompts antes/después
```

## 3. Código involucrado

- **`src/nn/trainer.rs`**: nueva `NativeGenomicTrainer.fit_lm_head` (y `GenomicTrainerCore::fit_lm_head`).
  Entrena solo el `lm_head` (fase 1) sobre `Vec<Vec<usize>>`, devolviendo el loss medio.
  Expuesto vía PyO3 (`fit_lm_head(model, dataset, lr)`).
- **`scripts/distill_smollm2_1teacher.py`**: orquesta las 3 fases usando la API pública
  `GenomicLLM.load_genomic` / `.generate` / `.tokenizer` y el trainer nativo.
- **`data/distill/`**: caché offline de los pares generados (gitignored si hace falta).

## 4. Por qué `lm_head` primero

- En el formato híbrido GAJE, `lm_head`/`token_embd` están en **FP32**: entrenarlos no
  sufre el ruido del cuerpo Q4_0.
- Es el cambio más pequeño que demuestra la mecánica (maestro → gradientes → estudiante).
- El cuerpo Q4_0 es la siguiente fase (IQAT profundo), y debe validarse **por separado**
  por su fragilidad documentada.

## 5. Métrica de éxito (mínima)

- El loss de `fit_lm_head` baja de forma estable entre épocas.
- La evaluación en prompts de control muestra una respuesta **coherente** (no necesariamente
  perfecta) tras el SFT, sin NaN/Inf (los guards de `train_step` ya los filtran).

## 6. Siguiente paso (solo si el MVP converge)

- Escalar a multi-maestro con **logits cacheados offline** (cargar cada maestro, emitir
  probs a disco, liberar) para no repetir el swapping.
- Si se quiere entrenar el cuerpo: hacerlo en un script separado (IQAT fase 2+), nunca
  en el test rápido.

## 7. Limitaciones honestas

- No aprende conocimiento factual nuevo; solo re-calibra el `lm_head` hacia el estilo del
  maestro. Para hechos se necesita RAG (`.gmem`) o entrenar el cuerpo.
- El `lm_head` entrenado no se re-exporta a `.flat` todavía (el guardado es `.gaje` db);
  el export a `.flat` es trabajo de seguimiento.
