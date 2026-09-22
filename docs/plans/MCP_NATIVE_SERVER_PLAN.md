# 🔌 Plan Servidor MCP Nativo — Memoria `.gmem` como Herramienta Soberana (SDD)

**Estado:** `draft` · **Versión:** `v1.7.4-alpha` · **Alcance:** `gaje-cli mcp` (Rust nativo, zero-Python)
**Reutiliza:** `src/server/mod.rs` (`LoadedModel`, `IslandOrchestrator`), `src/compute/island` (`MemoryConfig`, `prepare_prompt_with_memory`, telemetría 6 estados)

> MCP (Model Context Protocol, Anthropic) es el estándar para exponer herramientas/recursos a modelos vía JSON-RPC (stdio o Streamable HTTP). Este plan especifica un servidor MCP escrito en Rust que exponga la memoria hipocampal `.gmem` sin que los datos salgan de la máquina, y que permita a GAJE actuar como herramienta de otros sistemas (Claude Code, Cline, Goose, Claude Desktop).

---

## 1. Objetivo y No-Objetivos

### Objetivo
- `gaje-cli mcp [--model <ruta.gaje>] [--transport stdio|http]` expone `.gmem` como **recurso** + **2 tools** (`gmem_query`, `gmem_store`) con telemetría honesta.
- Claude Desktop o cualquier cliente MCP consulta memoria local; GAJE actúa como tool de otros sistemas. Sin inventar protocolo nuevo.

### No-objetivos (verdad empírica, cf. `README.md:209`)
- **No** es servidor multi-tenant empresarial: 1 usuario / 1 hilo activo, peticiones serializadas vía `RwLock` (`src/server/mod.rs:122`).
- **No** es retrieval denso de alta precisión: extractor `Weighted Mean Pooling` sobre `W_E`, `<0.5ms`, 0MB extra. Depende de solapamiento léxico; **no recupera paráfrasis pura sin vocabulario común** (casos P6/P11/P13 en `tests/test_threshold_calibration.rs`).
- **No** compite con `llama.cpp` en throughput multi-hilo masivo; compite en mmap zero-copy, cold-start y memoria asociativa en binario único.
- **No** ejecuta modelos >3B en consumo sin degradación severa (franja certificada 0.1–3B).

---

## 2. Decisión de Transporte (MVP stdio-first)

| Transporte | Cliente objetivo | Coste | Decisión |
|---|---|---|---|
| **stdio (JSON-RPC por stdin/stdout)** | Claude Code, Cline, Goose, agentes CLI locales | **1–2 días** | ✅ **Fase 0 (MVP)**. Loop síncrono como `run_server` (`mod.rs:188`), solo `serde_json`. Sin `tokio`, sin engordar binario Termux. |
| Streamable HTTP / SSE | Claude Desktop remoto | +1 semana (auth, CORS, sesiones, concurrencia) | Fase 1. Reutilizar `tiny_http` + CORS ya existente (`mod.rs:199`). |

Justificación: el foco del repo es **PWA móvil soberana / edge** (`AGENTS.md B.2`); stdio local alinea con Termux y agentes CLI. Desktop remoto es Fase 1.

---

## 3. Arquitectura Soberana (regla `AGENTS.md A.2`)

```
gaje-cli mcp --model models/production/<org>.gaje --transport stdio
  └─ load_model_and_tokenizer()          # mmap zero-copy existente
  └─ IslandOrchestrator::load_paired_for_model()  # .gmem pareada existente
  └─ MCP stdio loop (línea = 1 mensaje JSON-RPC 2.0, Content-Length opcional)
       ├─ initialize / notifications/initialized / ping
       ├─ tools/list · tools/call        # gmem_query, gmem_store (+ chat opcional Fase 1)
       ├─ resources/list · resources/read # gmem://niches, gmem://telemetry
       └─ prompts/list · prompts/get     # system_con Memoria (opcional Fase 1)
```

- **Sin dependencias nuevas pesadas.** Solo `serde`/`serde_json` (ya en `Cargo.toml`). Prohibido `rmcp+tokio` en MVP por huella en Android.
- **Reutilización estricta:** `LoadedModel::memory_config()` (`mod.rs:50`), `prepare_prompt_with_memory`, `calibrate_memory_threshold` (`mod.rs:66`), `find_model_path` con sandbox `file_name()` (`mod.rs:71`) para `memory_dir`.
- **Serialización documentada:** el `RwLock` existente serializa `tools/call`; el servidor responde secuencialmente. Documentar como límite, no como bug.

---

## 4. Contrato MCP (JSON-RPC 2.0)

### 4.1 `tools/list`
```json
[
  {
    "name": "gmem_query",
    "description": "Consulta memoria hipocampal .gmem local (<0.5ms). Retrieval léxico por mean-pooling; no paráfrasis pura. Devuelve score, entropía y telemetría.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": {"type": "string"},
        "top_k": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10},
        "threshold": {"type": "number", "description": "Override de τ; por defecto τ calibrado del modelo"},
        "use_whitening": {"type": "boolean", "default": true}
      },
      "required": ["query"]
    }
  },
  {
    "name": "gmem_store",
    "description": "Persiste texto en nicho .gmem local (append + índice 64B-aligned).",
    "inputSchema": {
      "type": "object",
      "properties": {
        "text": {"type": "string"},
        "niche": {"type": "string", "description": "Nicho existente o 'auto'"},
        "tags": {"type": "array", "items": {"type": "string"}}
      },
      "required": ["text"]
    }
  }
]
```

### 4.2 `tools/call gmem_query` → respuesta honesta obligatoria
```json
{
  "results": [{"niche": "rust_std", "text": "...", "similarity": 0.71}],
  "telemetry": {
    "state": "memory_injected | rejected_low_similarity | rejected_entropy_gap | whitening_missing | ...",
    "delta_top": 0.14,
    "threshold": 0.50,
    "whitening_missing": false,
    "latency_ms": 0.31
  },
  "caveat": "lexical-overlap retrieval; pure paraphrase without shared vocabulary may miss (P6/P11/P13)"
}
```
- **Gating obligatorio:** `Δ_top ≥ 0.12` o rechazo con `rejected_entropy_gap` (cf. `README.md:100`). Si falta `data/calibration/<modelo>.mu.bin`, fallback Opción C (`τ*=0.50` + `whitening_missing:true`, cf. `README.md:240`).
- Nunca devolver solo texto sin `telemetry`: es el contrato de verdad empírica.

### 4.3 `resources/list|read`
- `gmem://niches` — lista nichos + conteos + τ por modelo.
- `gmem://telemetry` — últimos N estados de 6-estados.
- `gmem://calibration/<modelo>` — presencia de `.mu.bin`, τ efectivo.

---

## 5. Seguridad
- Sandbox de rutas: reutilizar `find_model_path` (`file_name()`); `resources/read` nunca acepta `../`.
- stdio no requiere auth (proceso hijo local); transporte HTTP Fase 1 exige `127.0.0.1` por defecto + token `--mcp-token` + CORS restrictivo (no `*`).
- `gmem_store` con límite de tamaño por llamada (p.ej. 8KB) y rate-limit secuencial natural del `RwLock`.
- Sin exfiltración: todo `mmap` local; documentar que el cliente MCP (Claude) sí puede reenviar resultados a la nube — fuera del control de GAJE.

---

## 6. Fases y Estimación

| Fase | Alcance | Coste |
|---|---|---|
| **Fase 0 — MVP stdio** | `gaje-cli mcp --transport stdio`, `initialize/ping/tools+resources` arriba, tests `tests/test_mcp_*.rs`, doc cliente (Claude Code/Cline JSON) | **1–2 días** |
| Fase 1 — HTTP/SSE | `POST /mcp` Streamable HTTP sobre `tiny_http`, sesiones, `chat` como tool opcional con SSE, `--mcp-token` | +3–5 días |
| Fase 2 — Hardening | Fuzz JSON-RPC, benchmarks latencia MCP vs `/api/chat`, guía `GAJE_MCP_GUIDE.md` ES/EN/ZH, certificación en `EMPIRICAL_TRUTH_STATE.md` | +2–3 días |

---

## 7. BDD (Given-When-Then) — contrato previo a TDD

```gherkin
Feature: Memoria .gmem vía MCP stdio

  Scenario: Query con hit
    Given modelo <org>.gaje cargado con .gmem pareada y τ calibrado
    When cliente envía tools/call gmem_query {"query": "<término del nicho>"}
    Then responde state=memory_injected con similarity≥τ, delta_top y latency_ms<1

  Scenario: Query trampa ambigua
    Given consulta con Δ_top < 0.12
    When cliente envía gmem_query
    Then responde state=rejected_entropy_gap sin resultados y con threshold efectivo

  Scenario: Calibración ausente
    Given falta data/calibration/<modelo>.mu.bin
    When cliente envía gmem_query
    Then responde con whitening_missing=true y τ*=0.50, sin interrumpir inferencia

  Scenario: Store y re-query
    Given nicho "auto"
    When cliente envía gmem_store {"text": "..."} y luego gmem_query con solapamiento léxico
    Then el texto almacenado es recuperable con similarity documentada
```

---

## 8. Gates TDD / Certificación
- `cargo test --test test_mcp_stdio`: handshake, list, query-hit, query-reject, store-roundtrip, path-traversal rechazado.
- `gaje-cli mcp --help` documentado en `GAJE_CLI_GUIDE.md`; ejemplo de config cliente:
```json
{ "mcpServers": { "gaje": { "command": "gaje-cli", "args": ["mcp", "--model", "models/production/gaje_pico_135m.gaje"] } } }
```
- Benchmark: overhead MCP-stdio vs llamada nativa <1ms p50 en Ryzen 5800H y ARM Termux; tabla en `docs/reports/`.
- Cierre: entrada en `docs/meta/EMPIRICAL_TRUTH_STATE.md` + `docs/INDEX.md`.

---

## 9. Riesgos y Alternativas Consideradas
- **Riesgo expectativa:** clientes esperan RAG denso; mitigación = `caveat` + telemetría obligatoria + guía honesta.
- **Alternativa protocolo propio REST:** descartada — MCP es estándar creciente, REST ya existe (`/api/chat`, `/api/memory`).
- **Alternativa crate `rmcp`:** descartada en MVP por `tokio` y peso en Android; reevaluar en Fase 1 HTTP si el mantenimiento manual supera el coste de la dependencia.

*SDD → BDD (arriba) → TDD (`tests/test_mcp_*.rs`) antes de implementar.*
