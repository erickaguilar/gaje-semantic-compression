# Hallazgos: Entrenamiento del Cuerpo Genomic4Bit (Vía B)

> Fecha: 2026-08-16 · Rama: `develop` · Ámbito: validar y optimizar el entrenamiento
> del cuerpo del estudiante `smollm2_4bit.gaje.flat` por backprop end-to-end con
> cross-entropía, y llevar el resultado al Web UI.

---

## 1. Resumen ejecutivo

- El **gradiente del cuerpo `Genomic4Bit` quedó validado** contra numérico (f64) con
  error relativo ≈ **0.06** en capas tempranas (antes "mentía", con discrepancias de
  500x y signos opuestos).
- El entrenamiento del cuerpo **demostró generalización real** (held-out CE baja),
  punto dulce: **8 bloques, lr≈2e-4, gclip=1.0 → held-out 2.427** (baseline 2.559).
- El **lr por capas (layer-wise decay)** permite escalar el entrenamiento a **16-24
  bloques** de forma estable, con held-out casi idéntico al de 8, sin degradar ni NaN.
- El **export a `.flat` y su carga en el Web UI** quedaron operativos (se descubrió y
  corrigió un bug de formato: `save_genomic_model` escribe DB `redb`, no el mmap `GAJE`).

---

## 2. Corrección del gradiente del cuerpo `Genomic4Bit`

Dos bugs reales, ya corregidos:

1. **Nibble invertido en el transpose** (`backward_core` de `Genomic4Bit`): el kernel
   descomprime cada byte como par→nibble alto, impar→nibble bajo. El transpose usaba la
   convención inversa, produciendo gradientes erróneos en capas tempranas. Corregido
   para alinearse con el forward y con `read`.

2. **STE de `refine_with_grads_core`** (capa Genomic4Bit): dividía por `centroid_counts`.
   El gradiente verdadero es la **suma** de `g·x` sobre el centroide. Se quitó la
   división `/count` (solo en la rama `Genomic4Bit`; `Genomic2Bit` quedó sin cambios).

### Sobre los falsos "500x" y "signo opuesto"

No eran del backward, sino del **harness de pruebas**:
- `k_cache`/`v_cache` crecen con cada `forward_core_cached`/`forward_core`. Hay que
  llamar `clear_cache_core()` **entre fases** (análisis y numérico) para que ambos
  puntos de partida sean idénticos.
- Diferencias finitas en **f32** sobre la loss del modelo completo son ruido
  (ΔL≈1e-6). Usar **f64** con una loss sintética de gradiente fuerte.

### Tests de gradiente añadidos (ruta rápida, ~8.7s)

| Test | Qué valida | Resultado |
| :--- | :--- | :--- |
| `test_gradient_check_block_robust` | `backward_core_cached` bloque 0 vs numérico f64 | worst rel_err ≈ 0.06 |
| `test_refine_indexing_matches_forward` | STE vs suma manual del forward | rel 0.0033 |
| `test_transpose_isolated_nibble` | transpose por fila vs forward | rel 0.02 |

Se eliminó el ruidoso `test_gradient_check_numeric` y sus helpers.

---

## 3. Validación de generalización (held-out)

Configuración de prueba: estudiante `models/production/smollm2_4bit.gaje.flat`,
tokenizer `models/core/tokenizer.json` (vocab 49152), corpus
`data/distill/train_smollm2_1t.jsonl` (1520 tokens, split 80/20). **Baseline held-out
CE = 2.5590.**

| Configuración | train CE | held-out CE | Lectura |
| :--- | :--- | :--- | :--- |
| Escalera (4 bloques, lr=1e-5) | — | **2.482** | generaliza |
| Full-body (30 bloques, lr=1e-6) | — | **2.5525** | estable, generaliza, sin NaN |

Ambas finitas; el bloque 0 y el último mutan; sin NaN. Tests `#[ignore]`:
`test_body_training_heldout_generalization`, `test_body_training_fullbody_heldout`.

---

## 4. Barrido de bloques (escalera)

`n_blk` 4 → 8 → 12 → 16 con lr decreciente (train_len=180). **Mejor: 8 bloques,
lr=1e-5 → held-out 2.5299.** Test `#[ignore]`: `test_body_ladder_sweep`.

---

## 5. Barrido de lr (a 8 bloques)

| lr | held-out | Δ |
| :--- | :--- | :--- |
| 1e-5 | 2.5299 | −0.029 |
| 5e-5 | 2.5033 | −0.056 |
| 1e-4 | 2.4522 | −0.107 |
| **2e-4** | **2.4270** | **−0.132** |
| 5e-4 | 2.7484 | +0.189 (degrada) |

**Punto dulce: 8 bloques, lr≈2e-4, gclip=1.0 → held-out 2.427.** Sin NaN en todo el
rango; la frontera de estabilidad está justo antes de 5e-4, donde la CE sube
(sobreajuste/explosión de gradiente). Los límites conservadores iniciales (lr=1e-5)
eran demasiado bajos. Tests `#[ignore]`: `test_body_lr_sweep_blk8`,
`test_body_lr_high_boundary`.

---

## 6. Escalar bloques con lr por capas (layer-wise decay)

Nuevo método `train_sequence_cached_layerwise_core(tokens, lr, n_train_blocks, gclip,
lr_decay)`, con `lr_b = lr·decay^(n-1-b)` (mayor en bloques tardíos, menor en
tempranos). `decay=1.0` ≡ lr uniforme.

Resultado (lr=2e-4, baseline 2.559):

| n_blk | decay | held-out | Δ |
| :--- | :--- | :--- | :--- |
| 8 | 1.0 (uniforme) | 2.4270 | −0.132 |
| 8 | 0.7 | 2.4366 | −0.122 |
| 16 | 0.8 | 2.4325 | −0.127 |
| 24 | 0.85 | 2.4432 | −0.116 |

**Lectura**: el lr por capas permite **escalar a 16-24 bloques** (más capacidad del
cuerpo entrenada) con held-out casi idéntico al punto dulce de 8 y **sin degradar ni
NaN**. Con `decay` se entrena más del cuerpo por el mismo coste en calidad. Test
`#[ignore]`: `test_body_layerwise_scale`.

---

## 7. Export a `.flat` y carga en el Web UI

### 7.1 El pipeline (operativo)

`examples/export_trained.rs`: carga el modelo, entrena el cuerpo
(`train_sequence_cached_layerwise_core`), y guarda con `save_genomic_model`
reutilizando el `ModelConfig` del `.flat` original.

```
cargo run --release --example export_trained -- \
  models/production/smollm2_4bit.gaje.flat \
  models/production/smollm2_4bit_trained.gaje.flat \
  data/distill/train_smollm2_1t.jsonl [n_blk] [lr] [decay] [epochs]
```

Export de prueba: 16 bloques, lr=2e-4, decay=0.8, 1 epoch (1520 tokens) → train CE
**1.4756**, recargado y forward finito. Tiempo ≈ 9.7 min.

### 7.2 Bug descubierto: `save_genomic_model` escribe DB `redb`, no mmap `GAJE`

**Síntoma**: el Web UI fallaba con `OSError: Not a directory (os error 20)` al cargar
el `.flat` exportado.

**Causa raíz (doble)**:
1. `save_genomic_model` (Rust) usa `GajeDatabaseWriter` → produce un **DB redb**
   (magic `redb`), NO el formato mmap flat (magic `GAJE`). El escritor del formato
   mmap real es `scripts/export_gaje_flat.py`.
2. En `python/gaje/nn/stabilized.py`, el fallback de `load_genomic` anexaba
   `model.gaje` a cualquier path que no terminara en `.gaje`, asumiendo que era un
   directorio → el fichero redb se convertía en `archivo/model.gaje` → "Not a
   directory".

**Fix** (`stabilized.py`): solo anexar `model.gaje` si el path es un **directorio**;
si es un **fichero** (DB redb), abrirlo directo. Verificado:
`GenomicLLM.load_genomic(smollm2_4bit_trained.gaje.flat)` carga 30 bloques / n_embd
576 / tokenizer embebido.

### 7.3 Comportamiento observado en el Web UI

- El modelo entrenado aparece en el selector como `SMOLLM2 4BIT TRAINED` y se carga
  en RAM. Genera respuesta (el pipeline de export→carga→inferencia funciona).
- **La calidad de generación es baja** (respuestas repetitivas tipo "El capital de
  Francia es un idioma claro..."). Causa esperada: fine-tuning sobre un corpus mínimo
  (1520 tokens, 1 epoch) que sobreescribe el conocimiento base.

**Palancas para mejorar la calidad de generación** (en orden de impacto):
1. Corpus mucho mayor (`dataset_1000.txt`, `tiny_shakespeare.txt`).
2. No entrenar el `lm_head` (solo el cuerpo) para no corromper la proyección.
3. Menos epochs / lr menor (p.ej. 5e-5) para no sobreescribir la base.

---

## 8. Estado de la suite

- **Ruta rápida**: 38 passed / 0 failed / 6 ignored.
- **Tests lentos de entrenamiento** (todos `#[ignore]`): escalera, full-body,
  ladder_sweep, lr_sweep_blk8, lr_high_boundary, layerwise_scale.

---

## 9. Commits relacionados (`develop`)

| Commit | Contenido |
| :--- | :--- |
| `db13d1c` / `ed4788f` | Body backprop + IQAT wiring (Stage 1 / 1.5) |
| `f38093d` | Fix gradiente Genomic4Bit + gradient check |
| `766861c` | web_ui → submodule independiente |
| `68ef92e` | Validación held-out + barridos de escalera/lr |
| `096a96b` | lr por capas (layer-wise decay) |
| `16a0efd` | Export a `.flat` (`examples/export_trained.rs`) |
| `71c9af3` | Fix carga de `.flat` redb en el Web UI |

---

## 10. Archivos clave

- `src/nn/linear/backward.rs`: transpose Genomic4Bit (nibble) + STE (suma).
- `src/nn/block/cache.rs`, `src/nn/attention.rs`: ForwardCache + backward cacheado.
- `src/nn/llm/forward.rs`: `train_sequence_cached_core`, `train_sequence_cached_layerwise_core`.
- `src/nn/llm/integration_tests.rs`: tests de gradiente + entrenamiento (`#[ignore]`).
- `src/io/flat_writer.rs` (escribe DB redb), `src/io/flat_reader.rs` (lee mmap `GAJE` y DB).
- `python/gaje/nn/stabilized.py`: `load_genomic` (fix de carga redb).
- `examples/export_trained.rs`: export entrenado a `.flat`.
- `docs/meta/EMPIRICAL_TRUTH_STATE.md`: sección 9 con tablas de barridos.

---

## 11. Trabajo pendiente / siguiente

- Export de calidad real (corpus grande + solo cuerpo + lr bajo) para evaluar
  generación.
- **Resuelto**: el export a formato mmap `GAJE` ahora se hace con
  `save_genomic_flat` (Rust), que escribe magic `GAJE` zero-copy (mismo formato que
  `scripts/export_gaje_flat.py` y que carga el Web UI por mmap). Round-trip exacto
  (Δlogits=0.0), test `#[ignore]` `test_flat_mmap_roundtrip`. `save_genomic_model`
  (redb) se conserva para checkpoints intermedios; el export final usa mmap.
- Guardar/versionar el modelo entrenado fuera de git (los `.flat` están gitignored).