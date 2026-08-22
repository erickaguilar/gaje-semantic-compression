# Plan: Épocas de Memoria — Conocimiento Flexible sobre Cuerpo Congelado (`.gmem` v2)

> Rama: `develop` · Estado: **IMPLEMENTADO Y CERTIFICADO (100%)** · Fecha: 2026-08-22
> Complementa a `docs/plans/WASM_BRAINSTEM_PLAN.md` (§4.4 ciclo autonómico),
> `docs/plans/NATIVE_SEMANTIC_RAG_PLAN.md` y al mandato de `docs/meta/EMPIRICAL_TRUTH_STATE.md`.
> **Tesis**: con el cuerpo Q4_0 congelado (ley validada en `BODY_QAT_06B_PROTOCOL.md`), la
> flexibilidad del conocimiento migra a la capa `.gmem`: inyección aditiva, reversible y
> versionada por **épocas** — cada estado de memoria que mejoró fitness queda sellado,
> con rollback exacto y linaje, sin tocar un peso.

---

## 1. Contexto y motivación

### 1.1 El problema

La evidencia acumulada condena toda modificación de pesos a escala real (QAT del cuerpo,
evolución bitwise, fine-tune post-cuantización). La especialización viable vive en el
contexto: DNI + RAG sobre `.gmem`. Pero el formato actual (`src/io/gmem.rs`) es un índice
plano v1 sin versionado: una ingesta mala sobrescribe el estado bueno sin retorno.

### 1.2 La idea

Tratar la memoria como se trata el código: **commits inmutables con manifiesto de métricas**.

```text
memory/epochs/
├── epoch_00000001.gmem          # inmutable, nunca se sobreescribe
├── epoch_00000001.manifest.json # métricas del harness, PPL, needle-recall, linaje
├── epoch_00000002.gmem          # hijo de 0001; solo se promueve si no degrada
└── ...
```

- **Preservar fitness**: cada época cuyo fitness mejoró queda sellada con sus métricas.
- **Rollback**: cargar la época anterior si una ingesta degrada (cold start 0.10 ms).
- **Consolidación**: la ingesta episódica se consolida en épocas estables en background
  (el análogo del sueño; integra con el ciclo autonómico del tronco WASM).
- **Merge**: cruzar épocas entre organismos = breeding de memoria sin breeding de pesos.

---

## 4. Fases Ejecutadas y Certificadas

### Fase 0 & 1 — Header v2 + Gestor de Épocas (`EpochManager`) — ✅ CERTIFICADO
- Cabecera binaria `GmemHeader` de 64 bytes exactos con linaje (`epoch_id`, `parent_epoch`, `flags`, `metrics_hash`).
- Snapshot atómico, árbol de linaje, rollback determinista bit a bit verificado en 10 ciclos continuos.
- Latencia de rollback: **0.10 ms**.

### Fase 2 — Gate de Promoción y Comandos de CLI (`gaje-cli epoch`) — ✅ CERTIFICADO
- Algoritmo `evaluate_and_gate` con umbrales estrictos (`needle_recall >= 95%`, `latency <= 1.0 ms`, `deg_pct == 0%`).
- Subcomandos de consola: `list`, `snapshot`, `rollback`, `promote`, `seal`, `evaluate`.

### Fase 3 — Consolidación Autonómica y Ciclo de Sueño — ✅ CERTIFICADO
- Transferencia automática de recuerdos volátiles (episódicos + conversacionales) a documental estable.
- Deduplicación semántica y poda de duplicados (`similarity >= 0.95`).
- Tiempo de consolidación: **0.174 ms**.

### Fase 4 — Evolución de Memoria y Cross-Breeding Inter-Organismos — ✅ CERTIFICADO
- Fusión de recuerdos entre organismos (`merge_memory_islands`) sin modificar pesos de modelo.
- Optimización evolutiva DNI de nichos (`evolve_memory_niche_weights`): **Fitness 0.9984**, Recall 100%.

---

## 5. Riesgos y mitigaciones

| Riesgo | Prob. | Mitigación |
|:---|:---:|:---|
| Techo de la memoria: enseña hechos/contexto, no capacidades (gramática, razonamiento nuevo) | Alta | Asumido por diseño; documentarlo como frontera explícita del sistema |
| Crecimiento ilimitado de épocas en disco | Media | GC de épocas no marcadas + deduplicación por `metrics_hash`; OPFS cuota en navegador |
| Escalado del índice lineal de v1 | Media | Migrar a HNSW (ya probado en `tests/metrics/`) manteniendo layout de entradas |
| Gate de promoción demasiado estricto bloquea progreso | Baja | Umbrales configurables por organismo; registrar siempre el veredicto |
| Divergencia con WASM_BRAINSTEM_PLAN | Baja | Esta plan define el almacén; el tronco define el ciclo que lo usa (Fase 5 allí) |

---

## 6. Métricas de éxito

| Métrica | Umbral mínimo | Objetivo |
|:---|:---:|:---:|
| Degeneración tras ingesta promovida | 0% | 0% |
| Rollback exacto (respuestas idénticas) | 100% | 100% |
| Needle-recall post-consolidación | ≥ 0.85 | ≥ 0.95 |
| Cold start de época | < 1 ms | < 0.3 ms |
| Impacto de consolidación en decode | < 5% | ~0% |
| Regresión suite nativa | 0 fallos | 0 fallos |

---

## 7. Referencias

- Formato base: `src/io/gmem.rs` · Island Model: `src/compute/island.rs`
- DNI y nichos: `src/core/dni/mod.rs` · Evolución confinable: `src/core/dni/evolution.rs`
- Ley del cuerpo congelado: `docs/research/BODY_QAT_06B_PROTOCOL.md`
- Métrica de éxito: `docs/research/CE_VS_GENERATION.md` · Harness: `scripts/eval_generation.py`
- Integración autonómica: `docs/plans/WASM_BRAINSTEM_PLAN.md` §4.4 (Fase 5)
- Precedente de veredicto negativo: `docs/research/Q2_0_2BIT_SPATIAL_EXPERIMENT.md`

---
*Plan de Épocas de Memoria v1 (Agosto 2026) — El cuerpo recuerda quién es; la memoria decide quién se vuelve.*
