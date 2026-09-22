# 📚 Plan Escalado de Catálogo — Nuevos Modelos `.gaje` (Decisión + Gaps)

**Estado:** `draft` · **Versión:** `v1.7.4-alpha`
**No duplica:** este plan decide *qué* exportar y *qué falta por parchear*; la *cómo* transmutar/destilar vive en los documentos referenciados.

## 0. Referencias existentes (no duplicar)

| Tema | Documento canónico |
|---|---|
| Registro SHA-256 + recetas (qwen2_5_3b, deepseek_r1_1_5b, qwen2_0_5b, smollm2_135m, gemma4_student BORN de Gemma 4 E2B) | `docs/registry/MODELS_REGISTRY_AND_REPRODUCTION_RECIPES.md` |
| Rutas destilación CoT vs transmutación directa; GeGLU/RMSNorm+1/soft-cap Gemma; DeepSeek-Qwen compatibilidad inmediata | `docs/plans/DISTILLATION_DEEPSEEK_GEMMA_STRATEGY.md` |
| Roadmap MLA/MoE DeepSeek nativo, validación Gemma (RoPE interleaved, GeGLU, paridad HF) — EN PROGRESO | `docs/plans/completed/DEEPSEEK_GEMMA_SUPPORT_PLAN.md` |
| Plantilla `gemma` y resolución soberana sin heurísticas | `docs/guides/CHAT_TEMPLATES_AND_TOKEN_DICTIONARY_GUIDE.md` |
| Benchmarks oficiales (DeepSeek 1.5B: 7.0 tok/s, 8.0x) | `docs/reports/BENCHMARK_OFFICIAL_v1_6.md` |

## 1. Ya certificado (no re-certificar salvo cambio de formato)

`qwen2_5_3b` (2.24GB, multilingüe insignia) · `deepseek_r1_1_5b` (1.23GB, CoT) · `qwen2_0_5b` (499MB, micro-rápido) · `smollm2_135m` (474MB, nano edge) · `gemma4_student.gaje` (2.08GB, BORN destilado de Gemma 4 E2B — destilación, no transmutación directa).

## 2. Candidatos nuevos — matriz de decisión

| Candidato | Tamaño est. Q4_0 | Aporte vs 1.5B actual | Estado exportador | Veredicto |
|---|---|---|---|---|
| **SmolLM2-360M** | ~250MB | Rellena hueco 135M→500M en teléfonos | ✅ Mecánico (familia `SmolLM`, `apply_smollm_rope_patch`, `src/io/gguf/loader/metadata.rs:159`) | **1º — 1 tarde** |
| **Qwen3-0.6B** | ~400MB | Heredero directo del 0.5B | ❌ **Gap: cae en `Unknown`** — `"qwen3".contains("qwen2")==false` (`src/io/arch.rs:94`); asignaría `rope_base 10000+silu+standard` en vez de `1000000+swiglu+chatml` | **2º — tras parche 3 líneas** (rama `qwen3` en `arch.rs` + `flat.rs:137`) |
| **DeepSeek-R1-Distill-1.5B** (transmutación directa) | ~1GB | Razonamiento `<think>`, ya registrado como `deepseek_r1_1_5b.flat` | ✅ Mecánico (mapeado a Qwen2_5, `arch.rs:113`; recetas en docs referenciados) | **3º — solo si se quiere variante GGUF nueva; si ya está en registry, no duplicar archivo** |
| **Gemma 4 E2B** (transmutación directa) | ~1.5GB | Multilingüe + soporte Google; lo que el 1.5B menos tiene | 🟡 Parcial — rama `Gemma` existe (`interleaved+geglu`, `arch.rs:131`) pero pendiente validar tokenizador SentencePiece, GeGLU real, RMSNorm+1, soft-cap y sliding-window (ver `DEEPSEEK_GEMMA_SUPPORT_PLAN.md` Fase 4) | **4º — último, tras suite de paridad HF** |

> Nota: `gemma4_student.gaje` ya cubre Gemma 4 E2B por *destilación*; la transmutación directa es redundante salvo que se busque fidelidad total al maestro.

## 3. Checklist de certificación por modelo nuevo (obligatorio, no solo `export-flat`)

1. `gaje-cli export-flat <fuente.gguf> -o models/production/<nombre>.gaje` + `models inspect` (dims, `ArchitectureDescriptor`, RoPE, permutación Q/K).
2. Paridad token-a-token vs referencia (tokenizador + plantilla canónica, sin heurísticas por nombre).
3. PPL post-cuantización (gate Fase 2: `<50`; referencia ~1.60 en Qwen2.5) + factual multilingüe (París/Júpiter/Berlín/100°C).
4. Calibración memoria: τ por `dim` + `data/calibration/<modelo>.mu.bin` o fallback Opción C documentado.
5. Throughput + RAM + cold-start en tabla (Ryzen 5800H + ARM si aplica).
6. Entrada en `MODELS_REGISTRY_AND_REPRODUCTION_RECIPES.md` (SHA-256 + receta) + `BENCHMARK_OFFICIAL` + `EMPIRICAL_TRUTH_STATE.md`.

```gherkin
Feature: Nuevo modelo en catálogo
  Scenario: Export Qwen3 sin regresión
    Given GGUF Qwen3-0.6B oficial
    When export-flat + inspect + suite paridad/PPL/factual/τ
    Then descriptor Qwen3 correcto, PPL<50, factual publicado y registry actualizado
```

*Pregunta guía: no "¿puedo exportarlo?" sino "¿qué aporta que el 1.5B no tenga?". Orden: SmolLM2-360M → Qwen3-0.6B (tras parche) → DeepSeek solo si falta variante → Gemma 4 E2B directo último.*
