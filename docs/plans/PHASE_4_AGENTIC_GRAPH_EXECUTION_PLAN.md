# Plan Fase 4: Grafo de Razonamiento Agéntico Nativo en Rust (`gaje-swarm`)

> Rama: `develop` · Estado: **PROPUESTO** · Fecha: 2026-08-23
> Ejecuta la visión de `docs/plans/AGENTIC_GRAPH_RUST.md` bajo el **Mandato de Verdad Empírica**
> ([`docs/meta/EMPIRICAL_TRUTH_STATE.md`](../meta/EMPIRICAL_TRUTH_STATE.md)).
> **Tesis**: si el motor GAJE ya ejecuta inferencia forward-only a 19–32 tok/s y recuperación
> `.gmem` a 0.75 ms, entonces un grafo agéntico nativo en Rust puede orquestar enjambres de
> micro-organismos especializados con transiciones de nodo en µs — eliminando la frontera
> Python/JSON que domina los frameworks actuales (LangGraph, CrewAI).

---

## 1. Contexto y motivación

### 1.1 Piezas ya certificadas sobre las que se construye

| Primitiva | Ubicación | Certificación |
|:---|:---|:---|
| Inferencia nativa Q4_0+FP32 | `src/nn/llm.rs`, `RustGenomicLLM` | 19–32 tok/s CPU (Ryzen 7 5800H) |
| MCTS / Tree-of-Thoughts base | `src/compute/mcts.rs` | `select/expand/backpropagate` con UCT operativo |
| RAG semántico nativo | `src/compute/rag.rs`, `NativeSemanticRAG` | Recall@10 85.3% (SBERT 768d) |
| Memoria persistente `.gmem` v2 | `src/io/gmem.rs`, `IslandOrchestrator` | Latencia 0.75 ms, lineage/flags/seal |
| SDK Zero-GIL + C-FFI | `src/core/sdk.rs`, `src/io/ffi.rs` | Android/iOS/C++ sin runtime Python |
| Currículo híbrido H3 | `tests/benchmark_spsa_discrete.py` | Especialización de organismos 21.56× más rápida |

### 1.2 El problema de la frontera de orquestación

Los frameworks agénticos en Python pagan impuestos estructurales por cada paso del grafo:
serialización JSON/Pydantic (~5–15 ms/nodo), GIL que bloquea paralelismo real, y duplicación
de pesos al cargar N modelos en N procesos. Para un enjambre de micro-organismos GAJE
(135M × k especialistas + sintetizador 3B) ese impuesto anula la ventaja de la compresión.

**La pregunta central**: ¿puede el paso de estado entre nodos agénticos costar ~10 µs,
con pesos compartidos zero-copy y paralelismo multinúcleo real?

### 1.3 Precedentes

| Hecho | Fuente | Implicación |
|:---|:---|:---|
| LangGraph/LangChain latencia de nodo 5–15 ms | Medición pública de frameworks | Margen de mejora ≥ 500× |
| `timing_wheel.rs` O(1) para 1M+ contextos | CHANGELOG v0.9.0 | Scheduling asíncrono ya probado en el motor |
| Rayon evalúa linajes en paralelo total | CHANGELOG v0.9.0 | Paralelismo CPU real sin Tokio para grafos puros |
| KV-Cache DNA 2-bit por modelo | CHANGELOG v0.6.1 | N agentes concurrentes con huella RAM mínima |

---

## 2. Objetivo e hipótesis

> **H1 (latencia)** — La transición de estado entre nodos nativos (`AgentState` por valor,
> sin serialización) cuesta < 10 µs p50 en grafos de 5 nodos.
>
> **H2 (soberanía RAM)** — Un enjambre de 4 organismos (3× 135M + 1× 3B) con pesos
> compartidos vía `Arc` consume < 5 MB de RAM adicional sobre la carga individual de modelos,
> y ejecuta nodos en paralelo real multinúcleo.
>
> **H3 (calidad de ruteo)** — Un router 135M cuantizado clasifica intención con precisión
> suficiente (≥ 85% en micro-benchmark etiquetado) para dirigir el grafo sin escalada al 3B
> en > 60% de consultas simples.
>
> **Hipótesis nula**: si el overhead del runtime del grafo supera el costo de una llamada
> Python directa (< 1 ms/paso) o el router colapsa bajo distribución real, se documenta el
> veredicto negativo y el frente se congela (patrón Q2_0).

---

## 3. Diseño

### 3.1 Core del grafo (`src/compute/graph.rs`)

```text
StateGraph ── registra AgentNode (trait object, Send+Sync)
    │           └─ StepResult::Next { node_name } | Fork(Vec<...>) | End(state)
    ├─ Executor secuencial   (rayon::scope para forks; SIN Tokio en Fase 4a)
    ├─ DecisionEdge          (router por reglas → luego por LLM)
    └─ AgentState            (struct tipado, Clone barato, cero JSON)
```

**Decisión clave**: empezar con **Rayon, no Tokio**. Los nodos de inferencia son
CPU-bound; async solo aporta cuando existan tool calls con I/O de red (revisitar en 4d).

### 3.2 Tipos de nodos (en orden de implementación)

1. `GmemRetrievalNode` — envuelve `IslandOrchestrator::retrieve_context` (ya medido: 0.75 ms).
2. `GajeModelNode` — envuelve `GenomicLLM` compartido vía `Arc<RwLock<GenomicLLM>>`;
   prefill+decode configurable (temp, penalty, max_tokens).
3. `RuleRouterNode` — clasificación por reglas/keywords (baseline determinista de H3).
4. `LlmRouterNode` — clasificación con organismo 135M + validación contra RuleRouter.
5. `ToTNode` — árbol de pensamiento sobre `MctsTree` existente usando un `GajeModelNode`
   como función de evaluación de hojas.
6. `ToolNode` — (último) funciones Rust whitelisted con timeout duro; I/O externo solo aquí.

### 3.3 Seguridad de tool calling

- Registro explícito de herramientas permitidas (whitelist en `ToolRegistry`).
- Timeout por nodo (hard kill del hilo worker vía `rayon` + canal).
- Sin acceso a filesystem/red fuera de herramientas registradas explícitamente.

### 3.4 Reutilización estricta

| Componente nuevo | Reutiliza |
|:---|:---|
| `graph.rs` (executor) | `StepResult` del plan §4.2 de AGENTIC_GRAPH_RUST.md |
| `GmemRetrievalNode` | `IslandOrchestrator` completo (sin tocar `.gmem`) |
| `ToTNode` | `MctsTree::select/expand/backpropagate` de `mcts.rs` |
| CLI demo | Patrón de subcomandos de `gaje-cli.rs` |

---

## 4. Fases con umbrales de decisión

### Fase 4a — Micro-benchmark del core (2–3 días)
Implementar `graph.rs` mínimo (StateGraph + 3 nodos dummy CPU-bound + executor Rayon).
Comparar contra baseline: misma cadena orquestada desde Python con PyO3 (1 llamada/nodo).
**Gate (H1)**: transición nodo→nodo < 10 µs p50 y ≥ 100× vs baseline Python; si no,
documentar y reevaluar el diseño del executor antes de continuar.

### Fase 4b — Enjambre soberano (1 semana)
Conectar nodos reales: 3× `smollm2_135m.flat` especializados (currículo H3 de Fase 0)
+ 1× `qwen2_5_3b.flat` sintetizador, todos mmap zero-copy compartidos.
**Gate (H2)**: RAM adicional < 5 MB sobre carga individual; fork paralelo de 3 nodos
135M completa sin serialización intermedia; throughput del enjambre ≥ suma de individuos −20%.

### Fase 4c — Router empírico (1 semana)
Micro-benchmark de ruteo: dataset etiquetado de ~200 consultas (3 intenciones: factual/RAG/conversacional).
Comparar `RuleRouterNode` vs `LlmRouterNode` (135M) vs escalada directa al 3B.
**Gate (H3)**: precisión router ≥ 85%; ahorro de cómputo > 60% de consultas resueltas sin 3B;
respuesta final del grafo sin regresión vs pipeline monolítico 3B (evaluación ciega A/B).

### Fase 4d — ToT + Tools + CLI demo (1–2 semanas, condicional a 4a–4c en verde)
`ToTNode` sobre `MctsTree` (búsqueda de 3 profundidad máxima, presupuesto 16 evaluaciones)
y primer `ToolNode` sandboxeado. Demo `examples/agent_swarm.rs` + subcomando
`gaje-cli swarm --graph demo.json`.
**Gate**: ToT mejora respuesta factual certificada (needle multi-salto) vs greedy directo;
tool call E2E < 100 ms overhead sobre la herramienta sola; suite de tests en verde.

---

## 5. Riesgos y mitigaciones

| Riesgo | Prob. | Mitigación |
|:---|:---:|:---|
| Router 135M desvía tareas (distribución abierta) | Alta | Gate H3 con dataset etiquetado; fallback determinista a RuleRouter; escalada al 3B ante baja confianza |
| Contención de `RwLock` en fork paralelo | Media | Modelos inmutables tras carga (solo KV-cache mutable por agente); clones zero-copy `Arc<Vec<u8>>` ya existentes |
| Scope creep hacia framework completo | Media | Fases con gates; cierre temprano si hipótesis nula (patrón Q2_0) |
| Tokio tentación prematura | Baja | Prohibido hasta 4d; solo si tool calls con I/O lo exigen |
| Degradación generativa por prompts de orquestación | Media | Gate generativo permanente: 0% degeneradas en harness (herencia Fase 2); nunca promover checkpoints sin él |

---

## 6. Métricas de éxito

| Métrica | Umbral mínimo | Objetivo |
|:---|:---:|:---:|
| Latencia transición nodo→nodo (Fase 4a) | < 100 µs p50 | < 10 µs p50 |
| Speedup vs orquestación Python (Fase 4a) | ≥ 100× | ≥ 1000× |
| RAM runtime del grafo (Fase 4b) | < 50 MB | < 5 MB |
| Precisión router (Fase 4c) | ≥ 85% | ≥ 92% |
| Consultas resueltas sin sintetizador 3B (4c) | > 60% | > 80% |
| Overhead tool call E2E (Fase 4d) | < 100 ms | < 10 ms |
| Regresión suite nativa | 0 fallos | 0 fallos |

---

## 7. Referencias

- Visión arquitectónica: `docs/plans/AGENTIC_GRAPH_RUST.md` (tipos `AgentState`/`AgentNode`)
- Disciplina de gates: `docs/plans/ZERO_ORDER_NATIVE_TRAINING_PLAN.md` (fases 0–3 completadas)
- Verdad empírica: `docs/meta/EMPIRICAL_TRUTH_STATE.md`
- Primitivas: `src/compute/mcts.rs` · `src/compute/island.rs` · `src/core/sdk.rs`
- Frameworks de comparación: LangGraph, CrewAI (medición pública de latencia de nodo)

---
*Plan Fase 4 v1 (Agosto 2026) — Donde cada nodo es un organismo y el grafo es el ecosistema.*
