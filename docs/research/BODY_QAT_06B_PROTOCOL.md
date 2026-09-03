# Protocolo 0.6B: Técnica A vs B (¿QAT del cuerpo?) — Resultado NEGATIVO

> **Estado**: Cerrado (negativo documentado). Fecha: 2026-08.
> Contexto: validación de la regla general "la especialización se hace por
> contexto/lm_head, nunca por QAT del cuerpo" en un modelo sin los problemas de
> gibberish de SmolLM2-135M.

> [!WARNING]
> **HALLAZGO NEGATIVO CERTIFICADO — REGLA OPERATIVA DE CUERPO CONGELADO**  
> Aplicar QAT al cuerpo ya cuantizado ($Q4\_0$) destruye la coherencia de generación (100%/95% degeneradas vs 0% en base).  
> **Regla de Producción:** El cuerpo del transformer cuantizado se mantiene estrictamente congelado (*Frozen Body*); la adaptación se efectúa únicamente por contexto o sobre la cabeza del modelo (`lm_head`).

---

## Pregunta

El punto dulce de SmolLM2 fue la **destilación corta con lm_head congelado,
8 bloques, lr≈2e-4**. ¿Es una regla generalizable? Protocolo sugerido:

- **Técnica A**: 8 bloques profundos, lm_head congelado, lr 2e-4.
- **Técnica B**: cuerpo completo (24 bloques), lm_head congelado, lr 2e-4.
- Estudiante: **Qwen2-0.5B** (`qwen2_0_5b_q4_0_q8_0_embd.gaje.flat`).
- Maestro del corpus: **Qwen2.5-1.5B** (`generate_distill_corpus.py`).
- Evaluación: **20 prompts held-out** (fuera del banco de entrenamiento),
  `eval_generation.py --prompts-file data/distill/heldout_06b.json`.

## Corpus

- `data/distill/train_06b_15.jsonl` — 16 pares (selección del corpus de 55
  pares generado con el maestro 1.5B), 2274 tokens por-secuencia.
- Base CE del 0.6B sobre ese corpus: **9.12** (PPL 9146; corpus duro para 0.6B).

## Resultados

| Modelo | Train CE | Held-out d1 | Held-out d2 | rep | deg% |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Base** (sin entrenar) | 9.12 | **0.764** | **0.909** | **0.091** | **0%** |
| **A** (8 bloques, lm_head frozen) | 16.85 | 0.167 | 0.200 | 0.800 | **100%** |
| **B** (24 bloques, lm_head frozen) | 10.28 | 0.171 | 0.207 | 0.793 | **95%** |

Ambas técnicas **destruyen la generación** del 0.6B: colapso a "en en en en"
(len=6). El modelo base genera coherentemente en los mismos 20 prompts held-out.

### Interpretación

- El punto dulce de SmolLM2 **NO transfiere** a Qwen2-0.5B: incluso 8 bloques con
  lm_head congelado colapsan (100% degeneradas).
- A (8 bloques) subió el train CE de 9.12 a 16.85 (perturbación concentrada en
  las capas altas); B (24 bloques) subió menos (10.28) porque el decay por capas
  reparte lr desde 7.6e-8 hasta 2e-4 por 24 capas. Pese a ello, B también colapsa.
- **Conclusión reforzada**: el QAT del cuerpo cuantizado (Q4_0) degrada la
  generación de forma general, no es un artefacto de SmolLM2. La especialización
  debe hacerse por **contexto externo** (RAG/templates) y **lm_head/adaptadores
  ligeros**, no por gradientes sobre el cuerpo Q4_0.

## Infraestructura habilitada por este protocolo

Para poder correrlo fue necesario corregir/optimizar el backend de entrenamiento
(commits `571e8a2`, `d4b7276`):

1. **Path cacheado con `fused_qkv`/`fused_gate_up`**: el entrenamiento del 0.5B
   estaba roto (ignoraba los lineales fusionados de GQA, panic RoPE len=0).
2. **`backward_core` y forwards paralelos por rangos**: el lm_head (vocab 151936)
   convertía forward+backward en ~0.2 tok/s; ahora ~28 tok/s en forward.
3. **`refine_with_grads_core` Q4_0 paralelo** (dos fases, numéricamente idéntico).
4. **`flat_writer` serializa lineales fusionados** (`attn_qkv`/`ffn_gate_up`);
   antes la recarga de un modelo Qwen2 entrenado perdía los pesos.
5. **`eval_generation.py --prompts-file`** para evaluar con sets held-out.

## Lección

> En GAJE, el cuerpo cuantizado se **congela**: la adaptación se hace por contexto
> (RAG/.gmem, templates) y destilación sobre corpus limpios evaluando SIEMPRE con
> el harness generativo, nunca por CE ni por QAT del cuerpo Q4_0.