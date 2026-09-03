# 🧬 Hallazgos del Fallo: Born max.gaje Q2_0 dim256×8 — Validación Completa 5 Pasos

> **Audit refs:** `GAJE-20260830-015520` (78 tok gibberish), `GAJE-20260830-025600` (78 tok), `GAJE-20260830-100817`/`131310`/`131519` (81-245 tok gibberish multilingüe)  
> **Artefacto:** `models/born/max.gaje` Llama `dim256` `n_blocks8` `n_head4` `vocab49152` `Q2_0` `32×12B` `99.57 MB` 75t `471 MB` cache `25491 págs` `src/nn/repl.rs:32` `v1.7.0-alpha`  
> **Fecha:** 2026-08-30, **Archivado:** `experiments/archived/q2_0_born/` `PPL45`

> [!WARNING]
> **EXPERIMENTO CONCLUIDO — HIPÓTESIS REFUTADA (RESULTADO NEGATIVO)**  
> El modelo nacido puramente en 2-bit ($Q2\_0$) demostró viabilidad numérica pero colapso semántico severo ($PPL \approx 45$, lenguaje incoherente).  
> **Estándar Certificado para Producción:** Formato `.gaje.flat` v2 con cuerpo en **$Q4\_0$** y embeddings en **$FP32$**.

---

## 1. Resumen Ejecutivo

Born Q2_0 es **viable numéricamente** (`Loss 6.66→3.80 ↓42%`, `0 NaN/Inf`, `150-162 tok/s`, `GTOK 49152` OK) pero **no semánticamente** (`PPL≈45 >>10`, `0/3` prompts coherentes, gibberish `"un mont comoio catals..."`). Confirma predicción `Q2_0_2BIT_SPATIAL_EXPERIMENT` (0/20 coherentes) y descarte embrión desde cero por capacidad insuficiente.

---

## 2. Artefacto y Entrenamiento STE

* **Inspect** `src/io/models_cmd.rs`: `Q2_0 v2312 99.57 MB 75 tens` vs `gaje_pico 576d 273t 472 MB` — formato válido
* **20 épocas** `train-born` `unified_distilled_corpus 350 seq 43096 tok` `lr0.004 decay0.95` `8494.99s` `6.6628→3.8364 ↓42.42%` `88→116 tok/s` `src/bin/gaje-cli.rs:845` `PPL45`
* **+2 épocas** `3.8213→3.8076 ↓0.36%` `765s` meseta, **+5 épocas DNI** `pro3b` `50×5 lr0.002 temp1.0 batch64` `6.88→6.46 ↓11.8%` `100 MB` — maestro `pico135m` coherente, alumno gibberish persiste
* **1 época pro3b** `50 texts` `7.33→6.88` `60s` + `5 épocas` `7.33→6.46` `~1h` `162 tok/s` — no baja PPL

---

## 3. Validación 5 Pasos

| Paso | Comando | Resultado | Veredicto |
| :--- | :--- | :--- | :--- |
| 1 `inspect` | `models inspect max.gaje` | `Q2_0 99.57 MB 75t Llama 256` | OK formato |
| 2 `audit` | `audit max.gaje` `src/io/cli_tools.rs:479` | `0 NaN/0 Inf 100% limpio` `entropía alta homogénea` `anclas 0.0%` `veredicto producción` | OK numérico |
| 3 `chat Hola` | `chat --prompt Hola` `src/bin/gaje-cli.rs:609` | `177 tok/s` gibberish `src/nn/repl.rs:32` | Gibberish |
| 4 `benchmark 32` | `benchmark --tokens 32` `src/bin/gaje-cli.rs:527` | `150.29 tok/s 115 MB 0% degeneración` gibberish | Throughput OK, semántica no |
| 5 `pyo3_shim` | `GTOK 49152 merges48900 encode Hola=[42654,5825] decode OK` `src/core/gtok.rs:58` | Vocab sano | Descartado tokenizer |

**Bitácora** 3 turnos 245 tok `13:13` `¿Quién eres?`→gibberish, `capital Germany`→gibberish (no Berlin), `largest planet`→gibberish (no Jupiter) — 0/3 multilingüe

---

## 4. Hallazgos Clave

* **Convergencia ≠ coherencia:** Loss ↓42% necesaria pero no suficiente; PPL45 indica mínimo local plano de subwords (`"España" "capital" "sistema"` fragmentos sin sintaxis)
* **Capacidad:** `256×8` `Q2_0 12B/bloque` `lm_head 256×49152` (>50% params) no separa 49k conceptos con 2b/peso (~0.75b efectivo)
* **Q2_0 vs Q4_0:** `Q4_0 16 centroides` 20/20 coherentes vs `Q2_0 0/20` — destrucción de matices no lineal
* **Tokenizer sano:** `GTOK` unificado `src/nn/repl.rs:32` elimina variable BPE; gibberish persiste → no es BPE
* **Throughput vs calidad:** `150 tok/s` ruido ≠ valor; audit web `0.00` y `471 MB/WASM` son plantilla fija, no header real

---

## 5. Pipeline GPU — Infraestructura Validada

`src/compute/gpu/pipeline.rs` `RADV RENOIR Vulkan 16.32 GB/s` `16c AVX2/FMA` `ste_q2_backward.wgsl:48` `batched_gemv_q2.wgsl:53 workgroup(32,8,1)` `kl_divergence.wgsl:15` `GpuOnlineDistiller batch64` `CalibratedScheduler` `src/compute/gpu/scheduler.rs` `EWMA 0.3` `DistillationGraph N→M` `src/nn/distiller/graph.rs` `cargo test 60/60` `doctor` óptimo — CPU `1137%→639%` GPU `20%→31%` balance 60/40

---

## 6. Lecciones y Ruta

1. **Cerrar Born Q2_0** como `research` (este doc + `experiments/archived/q2_0_born/README.md`)
2. **Fix auditoría** leer `QuantFormat/dims` del header, no plantilla
3. **Pivot Q4_0** `gaje_nano_0_5b`/`pico` `Q4_0+FP32` `dim512/12L` `alpha0.7 lr0.001` 3×150 `pro3b` para PPL<10 `docs/plans/Q4_0_NANO_PICO_PRODUCTION_PLAN.md`
4. **Matriz control** `dim256×8 Q4_0 vs Q2_0` greedy 20 prompts `<500s` para aislar capacidad vs cuantización

**Veredicto:** Born `max.gaje` es éxito de infra (STE, WGSL, mmap, 150 tok/s) y fracaso semántico esperado — Q2_0 + dim256 no emerge lenguaje desde cero con 43k tok.
