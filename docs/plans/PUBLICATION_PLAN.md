# 📝 Plan de Publicación — Tres Hallazgos Verificables (Tech Report + Post)

**Estado:** `draft` · **Versión:** `v1.7.4-alpha`
**Evidencia canónica:** `docs/meta/EMPIRICAL_TRUTH_STATE.md` (líneas citadas abajo) · No requiere publicar el motor, solo metodología + números + artefacto reproducible.

## 0. Relación con docs existentes (no duplicar)

- Guía de publicación de *modelos* (destilación/despliegue GGUF): `docs/guides/GAJE_PUBLISHING_GUIDE.md` — no es publicación académica.
- Benchmarks oficiales: `docs/reports/BENCHMARK_OFFICIAL_v1_6.md`, `docs/reports/BENCHMARKS.md`.
- Estrategia DeepSeek/Gemma, registry SHA-256, catálogo, MCP, Memory Station: planes propios en `docs/plans/` — este plan solo los cita como trabajo futuro, no los repite.

## 1. Los tres hallazgos (con evidencia)

### H1. Zero-copy mmap real en Rust para inferencia LLM
- **Resultado:** `VmData` 4.16GB → 20.5MB (-99.5%), carga 18.5s → 140ms (132×) — `EMPIRICAL_TRUTH_STATE.md:614,616,726`; réplica Android `VmData=48.3MB/187ms` (`:680`), preservación zero-copy (`:717`).
- **Método a publicar:** lectura `/proc/<pid>/status` (`VmData`/`VmRSS`) + timestamp de `load_model_and_tokenizer` vía `mmap` (`memmap2`), antes (carga heap `Vec<f32>`) vs después (vista `mmap` + `FlatHeaderV2`). Comandos `gaje-cli benchmark/inspect`.
- **Alcance honesto:** 1 máquina Ryzen 7 5800H + Android/Termux; modelos 0.1–3B `.gaje` v2. No se afirma para >3B ni multi-hilo masivo.

### H2. La paradoja del codebook: "Q4_0" con más bits efectivos que Q8_0
- **Resultado:** layout histórico Q4_0 con 16 centroides f32 por bloque = 80 bytes/bloque ≈ **20 bits/peso** vs `Q8_0Block` (1 escala f16 + 32×i8 = 34 bytes) = **8.5 bits/peso** (-57.5%) — `:667,678,682`. Causa: sobrecarga de centroides inline (`:682`); justificación de Q8_0 en `lm_head` (`:634`).
- **Método a publicar:** aritmética de layout + `gaje-cli audit` (entropía de centroides, 0 NaN/Inf) + PPL comparativa. Declarar explícito: **patología del layout con codebook por bloque, no del Q4_0 de llama.cpp en general.**
- **Valor:** corrige intuición ("4 < 8 bits siempre") con medición.

### H3. Greedy es requisito funcional en RAG, no preferencia
- **Resultado:** control Canberra 5/5 (100%) en greedy T=0.0 vs 1/5 (20%) a T=0.2 perdiendo 4/5 contra el prior paramétrico Sydney — `:587-591`; conclusión de diseño citada (`:591`); tabla factual greedy-vs-sampling (`:163`); fila de certificación (`:729`). Test: `tests/test_cot_gmem_baseline.rs` (`:579`).
- **Método a publicar:** pipeline RAG `.gmem` + inyección en prompt, 5 pasadas por condición (T=0.0 vs T=0.2/Top-K=5/Top-P=0.9/pen=1.15) sobre Qwen2.5-1.5B y 0.5B, métrica override-autoritativo del contexto.
- **Valor:** regla de diseño citable por practicantes RAG.

## 2. Formato y venues

- **Formato:** 1 tech report de ~6 páginas (las 3 secciones + amenazas) + 1 post técnico espejo. Idioma: inglés para arXiv, ES/EN/ZH para blog/HF (convención del repo).
- **Venues en orden:** blog técnico + Hugging Face (`eaguilar/gaje-models`) → arXiv tech report → workshop sistemas/edge. Conferencia grande solo tras replicar en 2º hardware.
- **Artefacto:** `gaje-cli benchmark/audit/inspect` + scripts de medición + tabla de comandos. Sin publicar pesos propios salvo los ya listados en HF.

## 3. Outline del manuscrito

1. Intro (inferencia edge soberana, 0.1–3B, single-user) · 2. Formato `.gaje` v2 + `FlatHeaderV2` (1 párrafo, resto citado) · 3. H1 zero-copy (tabla antes/después + método `/proc`) · 4. H2 paradoja codebook (aritmética + PPL) · 5. H3 greedy-RAG (tabla 5/5 vs 1/5 + regla) · 6. Amenazas a la validez (abajo) · 7. Trabajo relacionado (llama.cpp, Ollama, Chroma — posicionamiento por huella/arranque, no throughput masivo) · 8. Reproducibilidad (checklist §4).

## 4. Checklist de reproducibilidad (gate antes de enviar)

- [ ] Comandos exactos + commit hash + `gaje-cli doctor` (SIMD/CPU) por tabla.
- [ ] `VmData/VmRSS` con n=5 + p50/p95 de latencia de carga y `retrieve <0.5ms`.
- [ ] Layout bytes/bloque + `audit` + PPL con corpus y seed fijos.
- [ ] Suite RAG 5 pasadas/condición con seeds y prompts congelados (`test_cot_gmem_baseline.rs` en verde).
- [ ] Amenazas declaradas: hardware único, 0.1–3B, baseline FP32 relativo en ablations (`Sc=0.944` vs Q4_0, no vs FP32 puro), corpus factual pequeño, serialización single-hilo.

```gherkin
Feature: Manuscrito enviable
  Scenario: Tablas verificadas
    Given commit fijado y suite verde
    When se re-ejecutan las 3 mediciones con checklist §4
    Then los números del draft coinciden dentro de ±5% o el draft se corrige
```

*SDD → checklist (arriba) → draft. Cierre: entrada en `EMPIRICAL_TRUTH_STATE.md` + enlace en `docs/INDEX.md`.*
