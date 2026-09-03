# 🧬 GAJE Agentic Graph: Motor de Orquestación Multi-Agente Nativo en Rust

> **Versión:** v1.6.0-alpha (Silver Adult)
> **Fecha:** 19 de agosto de 2026
> **Estado:** 📝 Plan Aprobado / Diseño de Arquitectura
> **Ubicación:** `docs/plans/AGENTIC_GRAPH_RUST.md`

---

## 1. 🎯 Visión y Objetivos

El **GAJE Agentic Graph (`gaje-swarm`)** es un motor de orquestación de grafos multi-agente asíncrono y de ultra-baja latencia implementado en **Rust puro (Tokio)**.

Está diseñado para reemplazar la sobrecarga de frameworks en Python (como LangGraph o CrewAI), permitiendo coordinar un **enjambre de modelos micro-especializados** (`smollm2_135m` a 4-bits) junto con modelos sintetizadores (`qwen2_5_3b`) con transiciones en **microsegundos** y cero copias de memoria (Zero-Copy).

---

## 2. ⚡ Comparativa Arquitectónica: Python LangGraph vs GAJE Graph en Rust

| Métrica / Característica | Python LangGraph | **GAJE Agentic Graph (Rust)** | Impacto |
| :--- | :---: | :---: | :---: |
| **Latencia entre Nodos (Paso de Estado)** | $5 - 15\text{ ms}$ (Pydantic/JSON) | **$< 0.01\text{ ms}$ ($10\text{ µs}$)** | ⚡ **1000x más rápido** |
| **Sobrecarga de Memoria del Runtime** | $150 - 300\text{ MB}$ | **$< 5\text{ MB}$** | 📉 **Ahorro masivo** |
| **Concurrencia y Paralelismo** | Limitada por el GIL de Python | **Paralelismo real multinúcleo (Tokio)** | 🚀 **Inferencia paralela real** |
| **Acceso a Pesos de Modelos** | Múltiples procesos / Duplicación | **Zero-Copy `mmap` compartido (Arc/RwLock)** | 🛡️ **Huella de RAM unificada** |
| **Seguridad de Tipos** | Dinámica / Runtime Errors | **Estricta en tiempo de compilación** | 🔒 **Cero caídas en producción** |

---

## 3. 🏛️ Arquitectura del Grafo y Flujo de Datos

El sistema modela la colaboración entre agentes como un **Grafo Dirigido Acíclico o Cíclico con Estado (StateGraph)**:

```
                                [ Usuario Query ]
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │   NODO ROUTER (135M)      │ ⚡ 25 ms
                          │   Clasificación de tarea  │
                          └─────────────┬─────────────┘
                                        │
                ┌───────────────────────┼───────────────────────┐
                ▼                       ▼                       ▼
     ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
     │ NODO EXTRACTOR      │ │ NODO TRADUCTOR      │ │ NODO RAG .GMEM      │
     │ (135M Especializado)│ │ (135M Especializado)│ │ (Recuperación <1ms) │
     └──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘
                │                       │                       │
                └───────────────────────┼───────────────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │  NODO SINTETIZADOR (3B)   │ ⚡ Si requiere
                          │  (Redacción y lógica)     │    razonamiento
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                                    [ FIN ]
```

---

## 4. 🧩 Especificación de Tipos y Traits en Rust

### 4.1 Estado Global Tipado (`AgentState`)
```rust
use std::collections::HashMap;

/// Estado fuertemente tipado que fluye y se transforma en cada nodo
#[derive(Debug, Clone, Default)]
pub struct AgentState {
    pub user_query: String,
    pub intent: Option<String>,
    pub extracted_entities: Vec<String>,
    pub context_retrieved: Vec<String>,
    pub tool_outputs: HashMap<String, String>,
    pub final_response: Option<String>,
    pub step_history: Vec<String>,
}
```

### 4.2 Trait de Nodo y Transición (`AgentNode`)
```rust
pub enum StepResult {
    /// Pasa al siguiente nodo especificado por nombre
    Next { node_name: String, state: AgentState },
    /// Ejecuta múltiples nodos en paralelo y recombina
    Fork(Vec<(String, AgentState)>),
    /// Finaliza la ejecución del grafo
    End(AgentState),
}

#[async_trait::async_trait]
pub trait AgentNode: Send + Sync {
    /// Nombre identificador del nodo
    fn name(&self) -> &str;
    /// Ejecución asíncrona del paso
    async fn process(&self, state: AgentState) -> Result<StepResult, String>;
}
```

---

## 5. 🛠️ Tipos de Nodos Nativos Soportados

1. **Inference Nodes (`GajeModelNode`):**
   * Ejecutan inferencia auto-regresiva nativa usando modelos `.gaje.flat`.
   * Permiten configurar temperatura, penalties y número de tokens de salida.
2. **Memory & RAG Nodes (`GmemPersistenceNode`):**
   * Consultan el índice plano `.gmem` de GAJE con latencia submilisegundo ($<750\text{ µs}$) para inyectar contexto relevante al estado.
3. **Tool Execution Nodes (`NativeToolNode`):**
   * Invocan funciones nativas en Rust (consultas SQL/SQLite, peticiones HTTP asíncronas con `reqwest`, o llamadas a herramientas de sistema).
4. **Conditional Router Nodes (`DecisionEdge`):**
   * Evalúan el contenido del estado para derivar la ejecución hacia el agente especializado correspondiente.

---

## 6. 📅 Plan de Implementación

| Fase | Tarea | Entregables |
| :---: | :--- | :--- |
| **Fase 1** | **Core Graph Engine** | Módulo `src/compute/graph.rs` con `StateGraph`, `AgentNode` y tests unitarios. |
| **Fase 2** | **Nodos GAJE Inferencia** | Integración directa de `GenomicLLM` como nodos de grafo compartidos vía `Arc<RwLock<GenomicLLM>>`. |
| **Fase 3** | **Enrutamiento Condicional** | Soporte para bifurcaciones paralelas (`tokio::spawn`) y combinación de estados (*join/reduce*). |
| **Fase 4** | **CLI & Demos** | Binario de ejemplo `examples/agent_swarm.rs` ejecutando un flujo conversacional multi-agente en CPU. |
