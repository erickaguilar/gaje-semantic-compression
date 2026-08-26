//! # 🕸️ GAJE Agentic Graph (Fase 4a): Core del Grafo de Razonamiento
//!
//! Motor de orquestación nativo de nodos agénticos con estado tipado.
//! El paso de estado entre nodos se hace por VALOR (struct Rust, cero
//! serializacion): la tesis de latencia < 10 µs/transicion que separa a este
//! motor de los frameworks Python (JSON/Pydantic por nodo).
//!
//! Plan: docs/plans/PHASE_4_AGENTIC_GRAPH_EXECUTION_PLAN.md (Fase 4a)
//! Gate H1: transicion nodo→nodo < 10 µs p50 y >= 100x vs orquestacion Python
//! con handoff serializado.

use std::sync::Arc;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Estado fuertemente tipado que fluye y se transforma en cada nodo.
/// Clone barato por diseno (payload corto + contadores): nunca JSON.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentState {
    pub user_query: String,
    pub intent: Option<String>,
    pub context: Vec<String>,
    pub tool_outputs: Vec<(String, String)>,
    pub response: Option<String>,
    /// Contador de pasos dados (telemetria del grafo).
    pub hops: u64,
}

impl AgentState {
    #[inline]
    pub fn with_query(query: impl Into<String>) -> Self {
        Self {
            user_query: query.into(),
            ..Default::default()
        }
    }

    /// Trabajo minimo representativo de un nodo real: tocar el payload
    /// (hash barato) para que el benchmark mida transicion + trabajo, no nada.
    #[inline]
    pub fn touch(&mut self) -> u64 {
        let mut h = 1469598103934665603u64;
        for &b in self.user_query.as_bytes() {
            h = (h ^ b as u64).wrapping_mul(1099511628211);
        }
        h ^= self.hops;
        self.hops += 1;
        h
    }
}

/// Resultado de procesar un nodo.
#[derive(Debug)]
pub enum StepResult {
    /// Continua hacia el nodo indice `next` con el estado transformado.
    Next { next: usize, state: AgentState },
    /// Ejecuta varios nodos en paralelo sobre estados derivados y recombina.
    Fork(Vec<(usize, AgentState)>),
    /// Termina la ejecucion del grafo.
    End(AgentState),
}

/// Trait de nodo agéntico. Send+Sync para forks paralelos via rayon.
pub trait AgentNode: Send + Sync {
    fn name(&self) -> &str;
    fn process(&self, state: AgentState) -> Result<StepResult, String>;
}

/// Error de ejecucion del grafo.
#[derive(Debug)]
pub enum GraphError {
    NodeIndex(usize),
    MaxSteps(u64),
    NodeFailed { node: usize, err: String },
}

/// Registro de nodos + executor secuencial con soporte de fork paralelo.
#[cfg_attr(feature = "python", pyclass)]
pub struct StateGraph {
    nodes: Vec<Arc<dyn AgentNode>>,
    max_steps: u64,
}

impl StateGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            max_steps: 10_000,
        }
    }

    pub fn with_max_steps(mut self, max_steps: u64) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn add_node(&mut self, node: Arc<dyn AgentNode>) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Ejecuta el grafo desde `start` con el estado inicial. Devuelve el estado
    /// final y el numero de transiciones efectuadas.
    pub fn run(&self, start: usize, state: AgentState) -> Result<(AgentState, u64), GraphError> {
        if start >= self.nodes.len() {
            return Err(GraphError::NodeIndex(start));
        }
        let mut current = start;
        let mut st = state;
        let mut transitions: u64 = 0;

        loop {
            if transitions >= self.max_steps {
                return Err(GraphError::MaxSteps(self.max_steps));
            }
            transitions += 1;
            let node = &self.nodes[current];
            match node
                .process(st.clone())
                .map_err(|e| GraphError::NodeFailed {
                    node: current,
                    err: e,
                })? {
                StepResult::End(s) => return Ok((s, transitions)),
                StepResult::Next { next, state } => {
                    if next >= self.nodes.len() {
                        return Err(GraphError::NodeIndex(next));
                    }
                    st = state;
                    current = next;
                }
                StepResult::Fork(branches) => {
                    // Ejecucion paralela real de ramas; los estados resultantes
                    // se fusionan en orden (join/reduce trivial en 4a).
                    use rayon::prelude::*;
                    let merged: Vec<AgentState> = branches
                        .into_par_iter()
                        .map(|(node_idx, s)| {
                            let mut cur = node_idx;
                            let mut s = s;
                            for _ in 0..64 {
                                match self.nodes[cur].process(s.clone()) {
                                    Ok(StepResult::End(done)) => return done,
                                    Ok(StepResult::Next { next, state }) => {
                                        s = state;
                                        cur = next;
                                    }
                                    _ => return s,
                                }
                            }
                            s
                        })
                        .collect();
                    st = merge_states(merged);
                    current = current; // permanece en el nodo despues del join
                }
            }
            st.touch();
        }
    }
}

fn merge_states(states: Vec<AgentState>) -> AgentState {
    let mut out = states.first().cloned().unwrap_or_default();
    for s in states.iter().skip(1) {
        out.context.extend(s.context.iter().cloned());
        out.tool_outputs.extend(s.tool_outputs.iter().cloned());
        out.hops += s.hops;
        if out.response.is_none() {
            out.response = s.response.clone();
        }
    }
    out
}

// --- Nodos dummy para benchmarks y tests (trabajo CPU minimo real) ----------

/// Nodo que toca el estado y avanza al siguiente indice (o termina en el ultimo).
pub struct ChainNode {
    pub idx: usize,
    pub next: usize,
    pub label: String,
}

impl AgentNode for ChainNode {
    fn name(&self) -> &str {
        &self.label
    }
    fn process(&self, mut state: AgentState) -> Result<StepResult, String> {
        state.touch();
        // Convencion de cadena: next == idx marca el nodo terminal.
        if self.next == self.idx {
            Ok(StepResult::End(state))
        } else {
            Ok(StepResult::Next {
                next: self.next,
                state,
            })
        }
    }
}

/// Construye una cadena de `n` nodos conectados secuencialmente.
pub fn build_chain(n: usize) -> StateGraph {
    let mut g = StateGraph::new();
    for i in 0..n {
        let next = if i + 1 == n { i } else { i + 1 };
        g.add_node(Arc::new(ChainNode {
            idx: i,
            next,
            label: format!("node_{i}"),
        }));
    }
    g
}

// --- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoNode {
        next: Option<usize>,
    }
    impl AgentNode for EchoNode {
        fn name(&self) -> &str {
            "echo"
        }
        fn process(&self, mut state: AgentState) -> Result<StepResult, String> {
            state.context.push(format!("echo:{}", state.user_query));
            match self.next {
                Some(n) => Ok(StepResult::Next { next: n, state }),
                None => Ok(StepResult::End(state)),
            }
        }
    }

    #[test]
    fn test_chain_runs_all_nodes_and_ends() {
        let g = build_chain(5);
        assert_eq!(g.len(), 5);
        let (state, transitions) = g.run(0, AgentState::with_query("hola")).unwrap();
        assert_eq!(transitions, 5);
        assert_eq!(state.hops >= 5, true);
    }

    #[test]
    fn test_state_flows_between_nodes() {
        let mut g = StateGraph::new();
        let a = g.add_node(Arc::new(EchoNode { next: Some(1) }));
        let _b = g.add_node(Arc::new(EchoNode { next: None }));
        let (state, _) = g.run(a, AgentState::with_query("x")).unwrap();
        assert_eq!(state.context.len(), 2, "ambos nodos tocaron el contexto");
    }

    #[test]
    fn test_out_of_range_start_errors() {
        let g = build_chain(2);
        assert!(matches!(
            g.run(99, AgentState::default()),
            Err(GraphError::NodeIndex(99))
        ));
    }

    #[test]
    fn test_max_steps_guard_against_cycle() {
        // Nodo que salta a si mismo: sin guardia seria bucle infinito.
        struct LoopNode;
        impl AgentNode for LoopNode {
            fn name(&self) -> &str {
                "loop"
            }
            fn process(&self, s: AgentState) -> Result<StepResult, String> {
                Ok(StepResult::Next { next: 0, state: s })
            }
        }
        let mut g = StateGraph::new().with_max_steps(100);
        g.add_node(Arc::new(LoopNode));
        assert!(matches!(
            g.run(0, AgentState::default()),
            Err(GraphError::MaxSteps(100))
        ));
    }
}

// --- Exposicion PyO3 (Fase 4a: micro-benchmark del gate H1) -----------------

/// Paso de nodo que cruza la frontera PyO3 con estado serializado (JSON).
/// Emula el handoff por-nodo de los frameworks Python (LangGraph-style):
/// serializar -> cruzar FFI -> deserializar. Es el baseline honesto.
#[cfg(feature = "python")]
#[pyfunction]
pub fn boundary_step_py(payload: &str, hops: u64) -> PyResult<(String, u64)> {
    let mut st = AgentState::with_query(payload);
    st.hops = hops;
    st.touch();
    Ok((st.user_query, st.hops))
}

/// Resultado del benchmark nativo.
#[cfg(feature = "python")]
#[derive(Clone)]
#[pyclass(get_all)]
pub struct GraphBenchResult {
    pub transitions: u64,
    pub total_ms: f64,
    pub ns_per_transition: f64,
    pub final_hops: u64,
}

/// Ejecuta `iterations` carreras de una cadena de `chain_len` nodos en Rust
/// puro y devuelve estadisticas por transicion.
#[cfg(feature = "python")]
#[pyfunction]
pub fn graph_bench_native_py(chain_len: usize, iterations: u64) -> PyResult<GraphBenchResult> {
    use std::time::Instant;
    let g = build_chain(chain_len);
    let start_state = AgentState::with_query("benchmark-query");
    // Warmup
    let _ = g.run(0, start_state.clone());

    let t0 = Instant::now();
    let mut last_hops = 0u64;
    for _ in 0..iterations {
        let (s, _) = g
            .run(0, start_state.clone())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        last_hops = s.hops;
    }
    let total_ns = t0.elapsed().as_nanos() as f64;
    let total_transitions = iterations * chain_len as u64;
    Ok(GraphBenchResult {
        transitions: total_transitions,
        total_ms: total_ns / 1e6,
        ns_per_transition: total_ns / total_transitions as f64,
        final_hops: last_hops,
    })
}
