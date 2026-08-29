use std::sync::Arc;
use _impl::compute::graph::{
    AgentNode, AgentState, GmemRetrievalNode, RoutingDecision, RuleRouterNode,
    StateGraph, StepResult, SwarmExecutor, SwarmIntent, SwarmRouterNode, ToolNode,
};
use _impl::compute::island::{IslandNiche, IslandOrchestrator};

#[test]
fn test_agentic_swarm_multi_node_execution() {
    let mut orchestrator = IslandOrchestrator::new(32);
    orchestrator.add_memory(
        IslandNiche::Documental,
        101,
        vec![0.1; 32],
        "GAJE es un protocolo de compresión semántica genómica y memoria persistente.".to_string(),
    );
    let orch_arc = Arc::new(std::sync::RwLock::new(orchestrator));

    let mut graph = StateGraph::new();

    // Node 0: Synthesizer (terminal)
    struct SynthesizerNode;
    impl AgentNode for SynthesizerNode {
        fn name(&self) -> &str {
            "synthesizer"
        }
        fn process(&self, mut state: AgentState) -> Result<StepResult, String> {
            state.touch();
            let mut final_answer = format!("Resultado sintetizado para '{}'", state.user_query);
            if !state.context.is_empty() {
                final_answer.push_str(&format!(" [Contexto: {}]", state.context.join(" | ")));
            }
            if !state.tool_outputs.is_empty() {
                final_answer.push_str(&format!(" [Tools: {:?}]", state.tool_outputs));
            }
            state.response = Some(final_answer);
            Ok(StepResult::End(state))
        }
    }
    let synth_idx = graph.add_node(Arc::new(SynthesizerNode));

    // Node 1: RAG Specialist -> Synthesizer (synth_idx = 0)
    let rag_node = GmemRetrievalNode::new("rag_specialist", orch_arc, 3, synth_idx);
    let rag_idx = graph.add_node(Arc::new(rag_node));

    // Node 2: Math Tool -> Synthesizer (synth_idx = 0)
    let math_node = ToolNode::new("math_evaluator", synth_idx, |st| {
        Ok(format!("Evaluated math for query: {}", st.user_query))
    });
    let math_idx = graph.add_node(Arc::new(math_node));

    // Node 3: Router
    let router = RuleRouterNode::new("intent_router", synth_idx)
        .add_route(vec!["documento".to_string(), "gaje".to_string(), "memoria".to_string()], rag_idx)
        .add_route(vec!["calcular".to_string(), "math".to_string(), "+".to_string()], math_idx);
    let router_idx = graph.add_node(Arc::new(router));

    // Turno 1: Ruta RAG
    let (state_rag, hops_rag) = graph
        .run(router_idx, AgentState::with_query("¿Qué es GAJE y cómo funciona su memoria?"))
        .unwrap();

    assert!(hops_rag >= 2);
    assert!(!state_rag.context.is_empty());
    assert!(state_rag.response.as_ref().unwrap().contains("Contexto"));

    // Turno 2: Ruta Tool
    let (state_tool, hops_tool) = graph
        .run(router_idx, AgentState::with_query("calcular 50 * 20"))
        .unwrap();

    assert!(hops_tool >= 2);
    assert!(!state_tool.tool_outputs.is_empty());
    assert!(state_tool.response.as_ref().unwrap().contains("math_evaluator"));
}

#[test]
fn test_agentic_swarm_fork_and_merge() {
    let mut graph = StateGraph::new();

    // Node 0: Worker Alpha (terminal)
    struct WorkerAlpha;
    impl AgentNode for WorkerAlpha {
        fn name(&self) -> &str {
            "worker_alpha"
        }
        fn process(&self, mut state: AgentState) -> Result<StepResult, String> {
            state.touch();
            state.tool_outputs.push(("alpha".to_string(), "ok".to_string()));
            Ok(StepResult::End(state))
        }
    }
    let a_idx = graph.add_node(Arc::new(WorkerAlpha));

    // Node 1: Worker Beta (terminal)
    struct WorkerBeta;
    impl AgentNode for WorkerBeta {
        fn name(&self) -> &str {
            "worker_beta"
        }
        fn process(&self, mut state: AgentState) -> Result<StepResult, String> {
            state.touch();
            state.tool_outputs.push(("beta".to_string(), "ok".to_string()));
            Ok(StepResult::End(state))
        }
    }
    let b_idx = graph.add_node(Arc::new(WorkerBeta));

    // Node 2: Forking Dispatcher
    struct ForkDispatcher {
        a: usize,
        b: usize,
    }
    impl AgentNode for ForkDispatcher {
        fn name(&self) -> &str {
            "fork_dispatcher"
        }
        fn process(&self, state: AgentState) -> Result<StepResult, String> {
            let mut s1 = state.clone();
            s1.context.push("rama_alpha".to_string());
            let mut s2 = state.clone();
            s2.context.push("rama_beta".to_string());
            Ok(StepResult::Fork {
                branches: vec![(self.a, s1), (self.b, s2)],
                next: None,
            })
        }
    }

    let d_idx = graph.add_node(Arc::new(ForkDispatcher { a: a_idx, b: b_idx }));

    let (res, transitions) = graph.run(d_idx, AgentState::with_query("paralelo")).unwrap();
    assert_eq!(transitions, 1);
    assert_eq!(res.tool_outputs.len(), 2);
}

#[test]
fn test_swarm_router_and_executor_batch() {
    let mut graph = StateGraph::new();

    // Node 0: Factual Specialist
    struct FactualSpecialist;
    impl AgentNode for FactualSpecialist {
        fn name(&self) -> &str { "factual_specialist" }
        fn process(&self, mut state: AgentState) -> Result<StepResult, String> {
            state.touch();
            state.response = Some(format!("Respuesta fáctica directa para: {}", state.user_query));
            Ok(StepResult::End(state))
        }
    }
    let factual_idx = graph.add_node(Arc::new(FactualSpecialist));

    // Node 1: Deep Reasoning Specialist (3B Synthesizer)
    struct DeepReasoningSpecialist;
    impl AgentNode for DeepReasoningSpecialist {
        fn name(&self) -> &str { "deep_reasoning_specialist" }
        fn process(&self, mut state: AgentState) -> Result<StepResult, String> {
            state.touch();
            state.response = Some(format!("Razonamiento profundo multi-paso para: {}", state.user_query));
            Ok(StepResult::End(state))
        }
    }
    let deep_idx = graph.add_node(Arc::new(DeepReasoningSpecialist));

    // Node 2: SwarmRouterNode
    let router = SwarmRouterNode::new("swarm_router", factual_idx, deep_idx, 0.70)
        .add_intent_route(vec!["simple".to_string(), "capital".to_string()], SwarmIntent::DirectFactual, factual_idx);
    let router_idx = graph.add_node(Arc::new(router));

    let executor = SwarmExecutor::new(Arc::new(graph));

    // 1. Consulta simple -> Factual Specialist
    let (st_simple, hops_s, elapsed_s) = executor
        .execute_profiled(router_idx, AgentState::with_query("capital de Francia"))
        .unwrap();
    assert!(hops_s >= 2);
    assert!(elapsed_s >= 0.0);
    assert!(st_simple.response.as_ref().unwrap().contains("Respuesta fáctica directa"));

    // 2. Consulta compleja larga -> Deep Reasoning
    let long_query = "Explica detalladamente la relación entre la teoría de la relatividad general, la mecánica cuántica y la transducción sintergial de Jacobo Grinberg en 500 palabras estructuradas paso a paso.";
    let (st_deep, hops_d, _) = executor
        .execute_profiled(router_idx, AgentState::with_query(long_query))
        .unwrap();
    assert!(hops_d >= 2);
    assert!(st_deep.response.as_ref().unwrap().contains("Razonamiento profundo multi-paso"));

    // 3. Batch masivo concurrente con Rayon
    let batch_queries = vec![
        "capital de Italia".to_string(),
        "simple query 1".to_string(),
        "simple query 2".to_string(),
        long_query.to_string(),
    ];
    let batch_results = executor.execute_batch(router_idx, batch_queries);
    assert_eq!(batch_results.len(), 4);
    for res in batch_results {
        assert!(res.is_ok());
    }
}
