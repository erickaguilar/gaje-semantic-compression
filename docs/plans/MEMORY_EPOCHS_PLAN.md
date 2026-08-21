# Plan: Épocas de Memoria — Conocimiento Flexible sobre Cuerpo Congelado (`.gmem` v2)

> Rama: `test/experimental` · Estado: **PROPUESTO** · Fecha: 2026-08-20
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
- **Rollback**: cargar la época anterior si una ingesta degrada (cold start 0.12 ms).
- **Consolidación**: la ingesta episódica se consolida en épocas estables en background
  (el análogo del sueño; integra con el ciclo autonómico del tronco WASM).
- **Merge**: cruzar épocas entre organismos = breeding de memoria sin breeding de pesos.

### 1.3 Base estructural existente

`GmemHeader` (64 B, `src/io/gmem.rs:13`) ya trae **40 bytes reservados sin uso**: caben
los campos de linaje sin romper compatibilidad. Las entradas `(id, vector f32, texto)` son
aditivas por diseño — el formato ya es un log de conocimiento.

---

## 2. Objetivo e hipótesis

> **H1 (flexibilidad segura)** — La inyección de conocimiento vía épocas `.gmem` no degrada
> la generación del cuerpo congelado cuando cada época pasa el gate del harness generativo.
>
> **H2 (reversibilidad exacta)** — El rollback a una época previa restaura el estado
> bit a bit (misma recuperación, mismas respuestas en prompts fijos).
>
> **H3 (fitness de memoria)** — La búsqueda evolutiva sobre la CAPA DE MEMORIA (qué entradas
> consolidar/podar/agrupar por nicho), evaluada por calidad de recuperación + harness
> generativo, mejora el fitness sin OOM ni colapso — a diferencia de la evolución de pesos
> (refutada en `EMPIRICAL_TRUTH_STATE.md` §4).

**Hipótesis nula**: si las épocas no mejoran needle-recall sin degradar generación, o el
rollback no es exacto, el versionado se simplifica a backup plano y se documenta.

---

## 3. Diseño

### 3.1 `GmemHeader` v2 (usa los 40 B reservados)

```rust
pub struct GmemHeaderV2 {
    pub magic: [u8; 4],        // b"GMEM"
    pub version: u32,          // 2
    pub dim: u32,
    pub index_type: u8,
    pub _pad: [u8; 3],
    pub num_entries: u64,
    // --- nuevos (antes reserved) ---
    pub epoch_id: u64,         // identificador monotónico de época
    pub parent_epoch: u64,     // 0 = raíz (linaje)
    pub created_at_unix: i64,
    pub metrics_hash: u64,     // hash del manifest asociado
    pub flags: u32,            // bit0: consolidada | bit1: sellada | ...
}
```

Lectores v1 ignoran los campos nuevos (mismo layout total de 64 B): compatibilidad bidireccional.

### 3.2 Manifiesto de época (`manifest.json`)

```json
{
  "epoch_id": 2,
  "parent_epoch": 1,
  "created_at": "2026-08-20T12:00:00Z",
  "entries_added": 42,
  "entries_pruned": 7,
  "metrics": {
    "needle_recall": 0.94,
    "ppl_validation": 12.3,
    "harness_deg_pct": 0.0,
    "rag_latency_ms_p50": 0.71
  },
  "verdict": "PROMOTED"
}
```

### 3.3 Gate de promoción (obligatorio, lección de `CE_VS_GENERATION.md`)

Una época nueva solo se promueve como *actual* si:
1. `harness_deg_pct == 0%` en el banco de prompts fijos;
2. needle-recall de las sesiones ingestadas ≥ umbral (p. ej. ≥ 0.90);
3. latencia RAG p50 < 5 ms (en navegador; < 1 ms nativo).

Si falla: la época se conserva como `REJECTED` (evidencia) y el puntero actual no cambia.
La CE nunca es criterio de promoción.

### 3.4 Evolución sobre la capa de memoria (H3)

Espacio de búsqueda (barato de evaluar, sin forward completo por candidato):
- qué entradas consolidar/podar (por resonancia media y antigüedad);
- asignación de nichos (General/Logic/Grammar/Memory del `DNIEngine`);
- umbrales de inyección (presupuesto de 128 tokens).

Fitness = w1·needle-recall + w2·(1 − deg%) + w3·(−latencia normalizada), medido con el
harness generativo en los finalistas. Confina así la maquinaria evolutiva existente
(`src/core/dni/evolution.rs`, `src/core/evolution_bitwise/`) a un dominio donde su coste
es viable — reviviendo el código archivado para micro-embriones de memoria, no de pesos.

---

## 4. Fases con umbrales de decisión

### Fase 1 — Header v2 + gestor de épocas (1–2 días)
`--epoch snapshot|list|promote|rollback` en `gaje-cli`. **Gate**: suite nativa verde;
cold start de una época < 1 ms; lectores v1 siguen cargando archivos v2 (y viceversa).

### Fase 2 — Manifiesto + gate de promoción (2–3 días)
Integración con `eval_generation.py`. **Gate**: H2 verificada (rollback bit a bit en 10
ciclos ingesta→rollback); una ingesta deliberadamente tóxica es bloqueada por el gate.

### Fase 3 — Consolidación autonómica (3–5 días)
Loop background (timing-wheel existente) que consolida ingesta episódica → época candidata
+ evaluación + promoción. **Gate**: impacto < 5% en throughput de decode durante consolidación.

### Fase 4 — Evolución de memoria en micro-embrión (1 semana)
Search de §3.4 sobre un organismo micro con corpus de ingesta real. **Gate**: mejora ≥ 15%
en needle-recall agregado sin incumplir jamás el gate de promoción. Fracaso ⇒ H3 rechazada
y documentada (patrón Q2_0).

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
