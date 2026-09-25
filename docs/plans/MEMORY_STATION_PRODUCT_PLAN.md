# 📦 Plan Producto Memory Station — Agente Local con Memoria Persistente (SDD)

**Estado:** `draft` · **Versión:** `v1.7.4-alpha` · **Dueño:** `gaje-cli serve` + Web UI + `.gmem`
**Reutiliza:** `src/compute/island.rs` (`add_memory:194`, `retrieve_context:203`, `save_all:655`, `load_paired_for_model:842`, `consolidate_memory:737`), `src/server/mod.rs` (`LoadedModel`, `GET /api/memory:258`)

> Producto: estación personal single-user que carga notas/papers/docs en `.gmem`, persiste hechos entre sesiones y responde con modelo local 0.1–3B sin nube, sin cuentas, sin dependencias. No es plataforma multi-agente ni servidor multi-tenant.

---

## 1. Propuesta de Valor e ICP

**Pitch:** tus notas y papers consultables con un modelo local en tu máquina/teléfono. Sin subir nada a OpenAI.

**ICP:** desarrolladores, investigadores, perfiles con información sensible (legal, médica, empresa). Patrón: cargan `md/txt`, papers y documentación técnica; preguntan Q&A factual corto.

**Diferenciador verificado:** binario único Rust zero-Python, mmap zero-copy (cold-start `<0.75ms`), retrieval hipocampal `<0.5ms` con 0MB extra, ejecución edge (Ryzen/ARM Termux). Frente a Ollama+Open WebUI+Chroma o AnythingLLM: sin Docker/Python, arranque instantáneo, memoria asociativa integrada.

**Lo que no es (verdad empírica, cf. `README.md:209`):**
- No multi-tenant ni multi-hilo masivo: `RwLock` serializa peticiones (`mod.rs:122`). **1 usuario / 1 hilo.**
- No retrieval denso: `Weighted Mean Pooling` sobre `W_E`, depende de solapamiento léxico. **No paráfrasis pura** (P6/P11/P13).
- No modelos >3B en consumo. Franja certificada 0.1–3B (11–32 tok/s Ryzen; ~4–5 tok/s esperado en hardware modesto con 1.5B).

---

## 2. Estado Real (auditoría 2026-09-22)

| Pieza | Estado | Evidencia |
|---|---|---|
| `serve` + Web UI + SSE + hot-swap | ✅ Implementado | `src/server/mod.rs:98`, `streaming.rs` |
| Carga auto `<modelo>_memory/` al seleccionar modelo | ✅ Implementado, sin UI | `load_paired_for_model:842`, `mod.rs:159` |
| `GET /api/memory` info + `GET /api/chat` + `POST /api/chat` | ✅ Implementado | `mod.rs:258,432` |
| Ingesta núcleo `add_memory` + `save_all` + `consolidate_memory` | ✅ Librería y atómico | `island.rs:194,655,737`, `gmem.rs:595` |
| `POST /api/memory/remember` ingesta en caliente | ✅ Implementado y certificado | `src/server/mod.rs`, `tests/test_memory_station_endpoints.rs` (T1, T2, T5, T6) |
| `DELETE /api/memory/entry` curación | ✅ Implementado y certificado | `src/server/mod.rs`, `tests/test_memory_station_endpoints.rs` (T3) |
| `POST /api/memory/consolidate` desduplicación | ✅ Implementado y certificado | `src/server/mod.rs`, `tests/test_memory_station_endpoints.rs` (T4) |
| Selector UI extracción/síntesis | ❌ No expuesto | grep vacío |
| Panel memoria (nichos, τ, whitening, consolidar) | ❌ No existe | — |
| Importador `md/txt` con chunking | ❌ No existe | — |
| Empaquetado `.deb/.rpm/binario` | ❌ Pendiente | — |

**Métricas a recortar antes de vender:** `needle_recall 1.0` y retrieval ~95% son con needle sintética y solapamiento léxico. Falta certificación en corpus sucio (PDFs, notas desordenadas, OCR). El gating `Δ_top ≥ 0.12` rechazará más en datos reales: es comportamiento correcto, no bug.

---

## 3. MVP Producto (alcance cerrado)

### 3.1 API mínima
```
POST /api/memory/remember { "text": "...", "niche": "auto|<nombre>", "tags": [] }
  → { "status": "ok", "niche": "...", "id": 1234, "similarity_selfcheck": 1.0 }
  Envuelve add_memory + save_all. Límite 8KB/llamada. Rechaza ../ en niche.

DELETE /api/memory/entry { "id": 1234, "niche": "..." }
  → { "status": "ok" } + save_all.

GET /api/memory → extender con { niches:[{name,count,threshold}], whitening_missing, calibration_path }

POST /api/memory/consolidate { "dedup_threshold": 0.97 }
  → { "removed": N, "stats": {...} } (expone consolidate_memory:737)
```

### 3.2 Modos UI (la decisión producto correcta)
| Modo | `max_tokens` | `temperature` | `system_prompt` | Uso |
|---|---|---|---|---|
| **Extracción** (defecto) | 128 | **0.0** (Greedy determinista) | "Responde solo con el hecho citado. Si no está en memoria, dilo." | Q&A factual exacto, elimina competencia del prior paramétrico |
| **Síntesis** | 512 | 0.4–0.7 | Asistente conciso estándar | Redacción, tolera latencia y variación creativa |

Toggle en header + persistencia en `localStorage`. Cada respuesta muestra `telemetry` (state 6-estados, `delta_top`, `threshold`, `latency_ms`) y `caveat` léxico.

### 3.3 Panel Memoria (evita vertedero write-only)
- Lista nichos + conteos + τ efectivo + badge `whitening_missing`.
- Botones: consolidar (dedup), exportar nicho (`.jsonl`), borrar entrada.
- Importador `md/txt`: textarea + file input, chunking ~512 chars con overlap 64, preview antes de ingerir. PDF/OCR: fuera de MVP (documentar).

### 3.5 Interacción Matemática: Filtrado (Threshold) vs. Ordenamiento (Niche Weight)

Para evitar la contaminación del contexto por hechos espurios o el bloqueo accidental de recuerdos válidos:
1. **Separación Estricta de Cantidades:**
   - **Filtrado (`pure_sim` $\in [-1.0, 1.0]$):** La decisión binaria de inyectar o rechazar un hecho se evalúa **únicamente sobre la similitud coseno pura no ponderada** (`m.similarity >= threshold && m.similarity >= niche_min`). Jamás se inyecta un hecho cuya similitud real esté por debajo del corte, independientemente del peso de su nicho (e.g. peso 1.2 en Documental no puede rescatar un hecho irrelevante).
   - **Ordenamiento (`pure_sim \times \text{niche\_weight}`):** El score ponderado se emplea exclusivamente como criterio de desempate en el ranking decreciente para priorizar qué hechos ocupan las primeras posiciones en el prompt.
2. **Gating Libre de Inversión de Índice:**
   - Tanto el umbral de entrada (`top_sim >= threshold`) como la brecha de entropía ($\Delta_{\text{top}} = S_1 - S_2 \ge \text{entropy\_gap\_threshold}$) se computan ordenando internamente las similitudes puras en orden descendente. Esto garantiza que $S_1$ sea siempre el máximo real del conjunto y que $\Delta_{\text{top}} \ge 0$, evitando que un hecho con menor similitud pero elevado peso de nicho genere deltas artificialmente negativas o un rechazo prematuro de candidatos afines.

### 3.6 Empaquetado
- Fase 0: `cargo-deb` (`.deb`) + `tar.gz` + guía Termux/APK. Fuente ofrecida por AGPL-3.0.
- Fuera de MVP: estático musl (fricción `tokenizers`/`wgpu`), `.rpm`, AppImage.

---

## 4. Pipeline de Ingesta (definición)
1. Chunk `md/txt` 512/64 → 2. `vector_from_text` (`island.rs:787`) → 3. `add_memory[_orthogonal]` → 4. `save_all` a `<modelo>_memory/` (índice 64B-aligned) → 5. `consolidate_memory` bajo demanda.
- Calibración: `configure_model_memory` por `dim`; si falta `data/calibration/<modelo>.mu.bin` → Opción C (`τ*=0.50` + `whitening_missing:true`), 100% operativo.

---

## 5. BDD (contrato previo a TDD)

```gherkin
Feature: Memory Station personal

  Scenario: Ingesta en caliente y re-query
    Given modelo con .gmem pareada cargado
    When POST /api/memory/remember {"text": "El puerto del relé es 8081", "niche": "notas"}
    And POST /api/chat {"message": "¿puerto del relé?", "use_memory": true}
    Then la respuesta contiene "8081" con telemetry memory_injected

  Scenario: Curación
    Given entrada id=1234 en nicho "notas"
    When DELETE /api/memory/entry {"id": 1234}
    Then re-query del hecho devuelve rejected_low_similarity

  Scenario: Modo extracción por defecto
    Given UI recién abierta
    When usuario envía pregunta factual
    Then se usa preset extracción (max_tokens=128, T=0.1) salvo cambio explícito a síntesis

  Scenario: Corpus sucio documentado
    Given 500 notas reales con ruido
    When se ejecuta suite needle+paráfrasis
    Then el reporte publica recall léxico y tasa rejected_entropy_gap sin prometer 95% denso
```

---

## 6. Gates TDD / Certificación
- `cargo test --test test_memory_station`: remember→query roundtrip, delete→miss, traversal `../` rechazado, límite 8KB, consolidate dedup.
- `tests/ui_e2e`: toggle extracción/síntesis persiste; panel muestra τ y whitening.
- Benchmark: latencia `remember` p50/p95, `retrieve` `<0.5ms` Ryzen + ARM, overhead vs `/api/chat` nativo; tabla en `docs/reports/`.
- Cierre: `GAJE_CLI_GUIDE.md` + guía usuario + entrada `EMPIRICAL_TRUTH_STATE.md` con recall en corpus sucio (no solo needle sintética).

---

## 7. Roadmap
| Fase | Alcance | Coste estimado |
|---|---|---|
| **F0 (MVP)** | `remember`+`delete`+`consolidate`, toggle extracción/síntesis, panel memoria, importador md/txt, `cargo-deb` | 3–5 días |
| F1 | Chunking mejorado, export/import `.jsonl`, backup `<modelo>_memory/`, guía ES/EN/ZH | +1 semana |
| F2 | PDF local (fuera de binario único o feature opt-in), eval corpus sucio publicada | según eval |

*SDD → BDD (arriba) → TDD antes de implementar. Nombre comercial: **Memory Station** (no "agentes").*
