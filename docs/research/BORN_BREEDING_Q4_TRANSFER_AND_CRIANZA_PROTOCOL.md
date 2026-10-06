# 🧬 Born y Crianza: Transferencia Q4_0 → Nacimiento Q2_0 y Protocolo de Evaluación

> **Tipo:** Hallazgo de investigación + protocolo operativo
> **Fecha:** 2026-10-06
> **Estado:** Propuesta documentada, no ejecutada
> **Alcance:** `src/nn/llm/birth.rs`, `gaje-cli birth/mutate/history`, `models/born/*`, pipeline de destilación y harness de evaluación

Este documento consolida tres conclusiones discutidas en sesión:

1. El problema de `born` no es incapacidad del sustrato 2-bit, sino **falta de tiempo de crianza, ausencia de consejo multi-maestro y evaluación incorrecta**.
2. Lo investigado y certificado en **Q4_0** es directamente trasplantable a nacimiento y crianza: mejoras en `lm_head`, arquitectura asimétrica y tokens de ingesta.
3. Se fija el protocolo mínimo para no repetir los caminos muertos ya certificados.

Referencias canónicas: `docs/research/BORN_2BIT_TRAINING_AND_COHERENCE_FINDINGS.md`, `docs/research/CE_VS_GENERATION.md`, `docs/plans/GOLDEN_RECIPE_2BIT_HIGH_POWER_DISTILLATION.md`, `docs/plans/DISTILLATION_DEEPSEEK_GEMMA_STRATEGY.md`, `docs/meta/EMPIRICAL_TRUTH_STATE.md` (Fases 4b, 16, 17, 20, 21), `docs/guides/BREEDING_AND_BORN_GUIDE.md`, `docs/reports/MAX_LASER_GENESIS_REPORT.md`.

---

## 1. Diagnóstico: por qué `born` parece fallar

### 1.1 Falta de crianza, no de nacimiento

* `max.gaje` (`D256/L8/H4/V4000`, 11.39 MB, génesis 76.69 ms, mmap 20 ms) reduce loss STE `7.9090 → 0.0641` (-99.19% en 2.25 s CPU, 0 NaNs) pero en `serve` balbucea léxico real sin gramática. Veredicto registrado: **fase embrionaria/infancia genómica** — canales alineados, gramática no ingerida.
* Causa: solo vio ~1520 tokens toy, 15 épocas. La hoja ontogénica exige Fase 1 léxico ~1.5M + Fase 2 sintaxis ~4.0M + Fase 3 diálogo ~7.5M tokens. Nunca se le dio ese tiempo.
* El `breeder` evolutivo se congeló tras el fallo en 135M+ (`421 s/gen, fitness 0.016, OOM-killed`). Esa refutación es válida para evolución directa en 135M+ en CPU, **no** para micro-embriones 1-5M ni para destilación en GPU. Congelar toda la línea fue una sobregeneralización.

### 1.2 Falta de consejo multi-maestro

* El único régimen que mejoró generación fue `distill 1520 tok, lm_head congelado` (d1/d2 máx, 0% degeneradas). Todo lo demás con un solo maestro o texto plano `.txt` en stream concatenado degeneró aunque la CE bajara (caso `dataset_1000: 4.88 → 3.83` = peor generación: "Por,", "Boriga").
* La receta ya escrita y no ejecutada (`GOLDEN_RECIPE`): `deepseek-r1` para `<think>` + `qwen2.5-3b` para sintaxis ES, `L = 0.3·CE + 0.7·KL(τ=2.0)`, 2000 pares CoT, 50 épocas GPU ~2 h.

### 1.3 Evaluación incorrecta

* CE sobre **stream concatenado** premia continuar ruido de concatenación, no hablar. CosSim sufre anisotropía (falsos positivos >0.85 en pares léxicamente cercanos pero disjuntos).
* Hallazgo central Fase 4b: **CE es métrica auxiliar; capacidad generativa es métrica de éxito**. Mayor caída de CE (-1.05) = peor generación.
* Norma vigente: `eval_ce_core` (misma tokenización) + `eval_generation.py` fijo (`temp 0.4 + greedy`, `distinct-1/2`, repetición de bigramas, % degeneradas). Sin cherry-picking. Separar ejes **fluidez/no-degeneración** vs **factualidad/generalización**.

---

## 2. Transferencia Q4_0 → Born: qué copiar literalmente

### 2.1 `lm_head` y embeddings: desacoplar y congelar

Lecciones Q4 híbrido v2 (cuerpo Q4_0 + `token_embd+lm_head` FP32 → CosSim ~0.90 vs maestro; puro Q4 colapsa CJK):

* Nacer ya híbrido: cuerpo Q2_0 + `embeddings/lm_head` FP32 (o Q8_0). Es lo que hace `create_born_organism` hoy — mantenerlo. Q2_0 puro implica `ρ = V/D > 100` y colisión masiva de vocabulario.
* Congelar `lm_head` en Fases 1-2 ontogénicas. Descongelar solo al final con `lr ≤ 2e-4` y early-stop guiado por harness.
* No intentar rescatar ranking roto con sampler: con `lm_head` Q2 distorsionado el muestreo da 0/15 recuperaciones. El typo (`Ganymiel`, `VeuS`) es geométrico, no de muestreo.
* Bugs ya corregidos a no reintroducir: `transpose backward` con nibble invertido (par→alto es canónico) y STE que debe **sumar** (`lr·Σ`), no promediar por `centroid_counts`.

### 2.2 Arquitectura: asimétrica y dinámica

* Atención intocable en bajo bit: Q2/Q3 en `Wq,Wk,Wv,Wo` → `Sc ≈ 0.23-0.38` letal por amplificación exponencial en `softmax(QKᵀ/√d)` acumulada en L capas. FFN tolera 3-bit (`Sc 0.9437` cuerpo, `0.9667` salida). Diseño born recomendado: **atención Q4_0 + FFN Q2_0** (~2.65 bits efectivos), no todo Q2_0.
* Reutilizar sin cambios: `FlatHeaderV2 + ArchitectureDescriptor` (`qk_permute` dinámico Qwen-split vs Llama-interleaved, `rope_base/eps` en cabecera), `ForwardCache` sin re-forward, backward correcto de RMSNorm + atención con RoPE inverso, `train_sequence_cached_layerwise_core` con `decay 0.8-0.85` y `gclip 1.0` para escalar 16-24 bloques sin NaN.
* Geometría de referencia `max_laser`: `D384/L12/H6` (64 dim/cabeza óptimo RoPE), `V4096, ρ = 10.6 ≤ 16`, `FFN 2.66×D`. `SwiGLU` puro es destructivo en 2-bit pequeño; evaluar `GELU/GeGLU` acotado estilo Gemma (`+1` en RMSNorm, `tanh` soft-capping si hay explosión de logits).
* Cuantizador Q4 nativo como plantilla: bloque 32 (`scale + min`), colocación simétrica de nibbles alineada con `Q4_0Block::dequantize_weight`, corrección Gray vs binario natural (`[c0,c1,c3,c2]` llevó CosSim 0.76 → 0.94 por capa).

### 2.3 Ingesta y tokens: per-secuencia + ChatML + GTOK

* Vocabulario humano calibrado GTOK 4K incrustado (`ρ ≤ 16`). Presión mayor colapsa ortogonalidad.
* Envoltura ChatML obligatoria (`<|im_start|>system … [Recuerdos] …<|im_end|>` + turnos). Texto plano provoca alfabetos aleatorios en 135M. Plantilla y `stop_tokens` por introspección de cabecera/GTOK, cero heurísticas por nombre de archivo.
* Entrenamiento **per-secuencia** con cache reseteado por pareja (CE base 4.53 → 2.83, ~150× más rápido), nunca stream concatenado. Corpus limpio delimitado desde maestro 3B (`generate_distill_corpus.py`). Para micro-experto: mix 40% código / 40% doc técnica / 20% diálogo general.
* Sampler calibrado Q4 como punto de partida: `T = 0.4`, `repetition_penalty = 1.1-1.15`, dedup con `HashSet` (una penalización por ID único) + exclusión de EOS/controles. Sin esto se miden loops, no inteligencia.
* Memoria congénita `.gmem` desde el nacimiento: `embed_text` (weighted mean pooling sobre `W_E`, `<0.5 ms`, 0 MB), whitening `μ` en `data/calibration/*.mu.bin` con fallback Opción C (`τ* = 0.50` + `whitening_missing:true`), gating `Δ_top ≥ 0.12` + K-WTA, telemetría de 6 estados, inyección solo en bloque system.

---

## 3. Protocolo mínimo de crianza (DoD)

1. **Nacer híbrido desacoplado:** `gaje-cli birth --dim 384 --layers 12 --heads 6 --ffn-dim 1024 --vocab-size 4096 --tokenizer data/gtok_human_4k.bin --with-memory -o models/born/<nombre>.gaje` → `audit` (0 NaN/Inf) obligatorio.
2. **Destilar con consejo, no con texto ciego:** 2000 pares CoT curados (`deepseek-r1` <think> + `qwen2.5-3b` fluidez), `α = 0.3, τ = 2.0`, cuerpo asimétrico (atención protegida), `lm_head` congelado, `lr` por capas, `n_blk = 8` punto dulce inicial.
3. **Evaluar generativo fijo:** `eval_ce_core` + `eval_generation.py` (`temp 0.4` y greedy, d1/d2, rep, %deg), más matriz bilingüe sostenida y calibración de prompts multilingües. CE solo como auxiliar. Prohibido declarar éxito por compilar o por bajar CE.
4. **Criterios de certificación:** `PPL held-out < 12`, correlación de ranking `> 0.88`, `0%` degeneradas en 64 toks, throughput ARM 35-45 tok/s, `audit` limpio, recall `.gmem` ≥ 19/20 con E2E declarado honestamente (techo del modelo base separado del recall de memoria).

Lo que queda explícitamente **no** recomendado hasta nueva hipótesis: fine-tune grande del cuerpo cuantizado sobre `.txt` concatenado, KL-logits como regularizador genérico (refutado β=0.1/1.0), CoT libre en ≤1B en producción, y Q2_0/Q3_0 homogéneo en atención.
