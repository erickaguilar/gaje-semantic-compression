# Plan: Export de Calidad — Fine-tuning del cuerpo para generación real

> Rama objetivo: `develop` · Autor: sesión 2026-08-16
> Contexto: el export de prueba (`smollm2_4bit_trained.gaje.flat`) ya carga y genera en
> el Web UI, pero la calidad es pobre. Este plan lo convierte en un export de calidad
> evaluable, atacando las 3 palancas identificadas en `BODY_TRAINING_VIA_B_FINDINGS.md`.

---

## 1. Objetivo

Producir `smollm2_4bit_quality.gaje.flat` entrenando **solo el cuerpo** sobre un corpus
grande, con **lr bajo** y **lr por capas**, de modo que:
1. La **CE held-out** baje respecto al baseline (2.559) y al export de prueba.
2. La **generación** deje de ser repetitiva y muestre conocimiento del corpus.
3. Se mantenga estable (forward finito, sin NaN, sin degradar el vocabulario).

## 2. Palancas de calidad (de `BODY_TRAINING_VIA_B_FINDINGS.md`)

| Palanca | Impacto | Estado actual | Acción |
| :--- | :--- | :--- | :--- |
| Corpus grande | **Dominante** | 1520 tokens (minúsculo) | Usar corpus de ~0.5-2 MB |
| No entrenar `lm_head` | Alto | Se entrena (corrompe vocabulario) | Añadir flag `train_lm_head=false` |
| lr bajo / pocas epochs | Medio | lr=2e-4, 1 epoch | lr≈5e-5, 1-3 epochs, decay layer-wise |
| Lr por capas | Medio | Disponible | `train_sequence_cached_layerwise_core` |

## 3. Corpus

### 3.1 Candidatos (datos existentes en `data/datasets/`)

| Fichero | Tamaño | Nota |
| :--- | :--- | :--- |
| `consolidated_silver_dataset.txt` | ~2.5 MB (414k palabras) | **Principal candidato** (calidad destilada) |
| `dataset_1000.txt` | ~87 KB (14k palabras) | Plan B si el silver es ruidoso |
| `dataset_es.txt` / `coherence_es.txt` / `dataset_es_ext.txt` | — | Opciones en español |
| `data/distill/train_smollm2_1t.jsonl` | 1520 tokens | Referencia (no basta solo) |

### 3.2 Estrategia
- **Primaria**: `consolidated_silver_dataset.txt` (máxima diversidad). Codificar entero con
  `GajeTokenizer::from_file("models/core/tokenizer.json")`.
- **Backup**: si el corpus mezcla idiomas/dominios con tokenización pobre, usar
  `dataset_1000.txt` (conocido) como control.
- **Split 80/20** para medir held-out CE; el 20% final se reserva para validación.

## 4. Cambios de código necesarios

> Estado: el writer mmap `GAJE` (`save_genomic_flat`) ya está implementado en
> `src/io/flat_writer.rs` y `export_trained.rs` lo usa como salida final. Round-trip
> verificado (Δlogits=0.0). El flag `train_lm_head` ya está implementado (default 0).

### 4.1 Entrenar solo el cuerpo (sin `lm_head`) — IMPLEMENTADO

`train_sequence_cached_layerwise_core(..., train_lm_head: bool)` salta la llamada a
`refine_with_grads_core` del `lm_head` (el backward del cuerpo sigue usando `d_logits`).
CLI: `examples/export_trained.rs` acepta `train_lm_head` (default 0).

**Resultado del primer export de calidad** (`smollm2_4bit_quality.gaje.flat`, 8 bloques,
lr=5e-5, decay=0.8, 1 epoch, lm_head congelado, 1520 tokens):

| Antes (lm_head entrenado) | Después (lm_head congelado) |
|:---|:---|
| "El capital de Francia es un idioma claro..." (sin sentido) | coherente, vocabulario intacto |

**Conclusión**: congelar el `lm_head` **arregla la corrupción de vocabulario** (la causa
dominante del gibberish). Persisten la repetición en prompts abiertos y los errores
factuales, que requieren la palanca de **corpus** (corrida de 30k+ tokens, la dominante).
Hoy `train_sequence_cached_layerwise_core` también llama
`self.lm_head.refine_with_grads_core(...)`. Para no corromper la proyección de vocabulario:
- Añadir parámetro `train_lm_head: bool` (default `true` para no romper tests existentes).
- Si `false`, **saltar** la llamada a `refine_with_grads_core` del `lm_head` (el backward
  del cuerpo sí depende de `d_logits`, que se sigue computando igual).

### 4.2 `examples/export_trained.rs`
- Aceptar corpus en `.txt` (ya soportado) y pasar `train_lm_head=false`, `lr`, `decay`, `epochs`.
- Añadir flag CLI `--skip-lm-head` (o `--train-lm-head`).
- Imprimir **CE held-out** al final además de train CE.
- Guardar como `models/production/smollm2_4bit_quality.gaje.flat`.

## 5. Hiperparámetros (basados en los barridos)

| Parámetro | Valor inicial | Racional |
| :--- | :--- | :--- |
| `n_blk` | 16 | layer-wise escala bien hasta 24 |
| `lr` | 5e-5 | bajo: proteger conocimiento base |
| `decay` | 0.85 | layer-wise (tardíos más lr) |
| `gclip` | 1.0 | sin NaN en todos los barridos |
| `epochs` | 1 → 3 | medir si la 2ª/3ª época aún mejora held-out |
| `train_lm_head` | false | evitar corrupción de vocabulario |

### Plan de barrido (pruebas `#[ignore]`, lentas)
1. **Corpus**: silver (2.5MB) vs `dataset_1000` (87KB) — misma config.
2. **lr**: 5e-5 vs 1e-4 vs 2e-4 (sobre el corpus ganador).
3. **epochs**: 1 vs 3 (sobre la mejor config), vigilando sobreajuste (held-out).

## 6. Validación y criterios de aceptación

### 6.1 Cuantitativa
- **CE held-out** final < 2.427 (mejorado sobre el punto dulce de 8 bloques, si el corpus
  es más expresivo) y al menos **< 2.559** (baseline).
- Sin NaN; `forward` finito tras export y recarga.

### 6.2 Cualitativa (Web UI)
- Prompts de control: hechos factuales, un párrafo del corpus, una frase de conocimiento
  general. Criterios: **no repetitiva** (sin lazo "El capital de Francia es..."), responde
  con contenido del corpus, coherencia de 2-4 frases.
- Comparar lado a lado contra `smollm2_4bit.gaje.flat` (base) y el export de prueba.

## 7. Pasos de ejecución

1. Implementar `train_lm_head` flag en `train_sequence_cached_layerwise_core` (+test unitario).
2. Ampliar `examples/export_trained.rs` (flag + CE held-out + ruta de salida).
3. Barrido de corpus y lr (tests `#[ignore]`) → elegir config ganadora.
4. Generar el export final `smollm2_4bit_quality.gaje.flat`.
5. Cargar en Web UI, correr prompts de control y evaluar cualitativamente.
6. Actualizar `BODY_TRAINING_VIA_B_FINDINGS.md` con los resultados y `EMPIRICAL_TRUTH_STATE.md`.
7. Commits incrementales + push a GitHub y GitLab (`--no-verify` por el hook de formato).

## 8. Riesgos y mitigaciones

| Riesgo | Mitigación |
| :--- | :--- |
| Corpus silver ruidoso o multi-idioma | Backup `dataset_1000.txt`; inspección de muestras |
| Sobreajuste en epochs altas | Vigilar held-out por epoch; quedarse en la mejor |
| Degradación de vocabulario por lm_head | `train_lm_head=false` |
| Coste de tiempo (2.5MB ≈ muchas horas en 1 core) | Empezar con un subconjunto (p.ej. 20k tokens) para iterar, luego full |
| `save_genomic_model` escribe redb (no mmap GAJE) | **Resuelto**: export final usa `save_genomic_flat` (mmap `GAJE`, round-trip exacto, test `#[ignore]`); redb se conserva para checkpoints |

## 9. Entregables

- Código: flag `train_lm_head` + export ampliado + tests de barrido (`#[ignore]`).
- Modelo: `models/production/smollm2_4bit_quality.gaje.flat` (gitignored).
- Evidencia: tabla de CE held-out por config + muestras de generación.
- Docs: este plan actualizado con resultados + notas en `EMPIRICAL_TRUTH_STATE.md`.

## 10. Decisión abierta
- ¿Entrenar el cuerpo completo (30 bloques) con lr muy bajo y decay agresivo, o quedarse en
  los 16-24 bloques validados? La prueba `full-body` (30 bloques, lr=1e-6) ya fue estable
  (held-out 2.459), pero **más bloques ≠ mejor** (8→2.427, 16→2.432, 24→2.443).

## 11. Resultado experimental (corrida 30k en `dataset_1000`)

**Config**: `export_trained`, 8 bloques, lr=5e-5, decay 0.8, 1 epoch, lm_head congelado,
30020 tokens. Tiempo real: **20602s ≈ 5.7 h** (no semanas — viable; el crecimiento de
contexto pesa más que la extrapolación lineal de la sonda de 4k tokens).

**Desenlace — negativo informativo**:
- train CE = **3.8262** (> baseline held-out 2.559) → el cuerpo **no redujo la pérdida**
  sobre `dataset_1000`; apenas se adaptó.
- **Generación degenerada** ("Por,,,,,", "Boriga, Boriga") — **peor** que el modelo de
  corpus pequeño de destilación (que produce "El mar es el mundo...", coherente).

**Lección (más importante que "más datos = mejor")**:
> **El corpus dominante no es el tamaño, es la pertinencia.** `dataset_1000` es
> **out-of-distribution** y ruidoso para SmolLM2 (base de inglés/instrucciones). Entrenar
> el cuerpo con lr conservador sobre un corpus OOD grande **degrada la generación** sin
> bajar el CE, en vez de mejorarla.

**Implicación para el plan**: retirar `dataset_1000` como candidato principal. El camino
ganador es un corpus **in-domain/curated** para SmolLM2 (instrucciones o el corpus de
destilación ampliado), no un `.txt` arbitrario, y con lr/bloques que permitan al cuerpo
adaptarse de verdad (subir lr o bloques respecto a 5e-5/8).

**Modelo utilizable**: el mejor sigue siendo `smollm2_4bit_quality.gaje.flat` (corpus de
destilación 1520 tokens, dominio-matched, lm_head congelado) — coherente aunque repetitivo.

## 12. Diagnóstico C: CE base vs entrenado (misma tokenización, `eval_ce_core`)

Se añadió `eval_ce_core` (forward-only, misma ruta que entrenar) y el modo `--eval-only` en
`export_trained.rs` para medir el CE del modelo **BASE** sin tocar pesos.

| Corpus | BASE CE (PPL) | TRAINED CE | Δ | Generación |
|:---|:---:|:---:|:---:|:---|
| `distill` (jsonl 1.5k) | 1.5386 (4.7) | 1.4622 | **−0.08** | coherente ✓ |
| `corpus_unified` (txt 2.9k) | 2.9919 (19.9) | 3.1146 | **+0.12** | degenerada |
| `dataset_1000` (txt 30k) | 4.8825 (132.0) | 3.83 | **−1.05** | degenerada |

**Conclusiones**:
- **Hipótesis de tokenización REFUTADA**: el CE base de `corpus_unified` es 2.99, no ~4.0.
  No es el formato; es el contenido.
- **El cuerpo SÍ se adapta**: baja CE en 2 de 3 corpora (dataset_1000 **−1.05**, la mayor).
  La afirmación previa "apenas se adapta" era incorrecta.
- **Hallazgo decisivo — el CE NO correlaciona con la calidad de generación**: dataset_1000
  baja CE −1.05 pero degenera; distill baja −0.08 y es coherente.
- **Causa raíz**: el objetivo de entrenamiento/eval (media de CE sobre un stream
  concatenado) está **desalineado** con la meta de generación. `dataset_1000` es patológico
  (PPL base 132); el cuerpo sobreajusta sus patrones locales ("Por,", "Boriga"), baja el CE
  medio pero destruye la generación. `distill` funciona porque es limpio y delimitado.
- **Implicación**: la Opción A (arreglar tokenización de `.txt`) queda descartada. El camino
  correcto es un **corpus limpio y delimitado** (tipo distill) escalado, no reformatear los
  `.txt` crudos.
  (held-out 2.5525); el lr por capas podría permitir full-body con mejor lr efectivo. Se
  evaluará si el tiempo de cómputo lo permite.