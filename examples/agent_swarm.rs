//! 🧬 GAJE Swarm Demo — Orquestación Multi-Agente Nativa en Rust
//!
//! Demuestra:
//! 1. Ruteo en microsegundos (SwarmRouterNode)
//! 2. Recuperación RAG contextual .gmem (GmemRetrievalNode)
//! 3. Ejecución de herramientas sandboxeadas (ToolNode)
//! 4. Razonamiento en árbol Tree-of-Thoughts sobre MCTS (ToTNode)
//! 5. Síntesis y fork paralelo zero-copy con Rayon

use _impl::compute::graph::*;
use _impl::compute::island::{IslandNiche, IslandOrchestrator};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==================================================================");
    println!("🧬 GAJE AGENTIC SWARM (Fase 4d Demo) — Enjambre de Razonamiento");
    println!("==================================================================");

    // 1. Inicializar Memoria Asociativa .gmem con conocimiento genómico
    let mut orchestrator = IslandOrchestrator::new(32);
    orchestrator.add_memory(
        IslandNiche::Documental,
        1,
        vec![0.25; 32],
        "GAJE Protocol v1.7.0: Memoria genética persistente zero-copy con latencia <0.75ms.".to_string(),
    );
    orchestrator.add_memory(
        IslandNiche::Episodic,
        2,
        vec![0.50; 32],
        "Needle Multi-Salto: La clave del genoma GAJE reside en el acoplamiento K-WTA toroidal.".to_string(),
    );
    let orch_arc = Arc::new(std::sync::RwLock::new(orchestrator));

    // 2. Construir Grafo de Estado (StateGraph)
    let mut graph = StateGraph::new();

    // Node 0: Sintetizador Final
    struct FinalSynthesizerNode;
    impl AgentNode for FinalSynthesizerNode {
        fn name(&self) -> &str { "final_synthesizer" }
        fn process(&self, mut state: AgentState) -> Result<StepResult, String> {
            state.touch();
            let mut summary = format!("Respuesta consolidada para: \"{}\"\n", state.user_query);
            if !state.context.is_empty() {
                summary.push_str(&format!("  [Contexto/RAG/ToT]: {}\n", state.context.join(" | ")));
            }
            if !state.tool_outputs.is_empty() {
                summary.push_str(&format!("  [Herramientas]: {:?}\n", state.tool_outputs));
            }
            state.response = Some(summary);
            Ok(StepResult::End(state))
        }
    }
    let synth_idx = graph.add_node(Arc::new(FinalSynthesizerNode));

    // Node 1: Recuperador RAG .gmem
    let rag_node = GmemRetrievalNode::new("rag_memory_specialist", orch_arc, 2, synth_idx);
    let rag_idx = graph.add_node(Arc::new(rag_node));

    // Node 2: Herramienta de Cálculo Matemático (Sandbox)
    let tool_node = ToolNode::new("math_calculator", synth_idx, |st| {
        let t0 = std::time::Instant::now();
        let result = format!("eval_sandbox(input='{}')", st.user_query);
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        println!("    [ToolNode: math_calculator] Ejecutado en {:.3} ms (overhead < 100 ms)", elapsed);
        Ok(result)
    });
    let tool_idx = graph.add_node(Arc::new(tool_node));

    // Node 3: ToT Reasoner sobre MctsTree
    let tot_node = ToTNode::new("tot_mcts_reasoner", 3, 16, 1.41, synth_idx);
    let tot_idx = graph.add_node(Arc::new(tot_node));

    // Node 4: Swarm Router
    let router = SwarmRouterNode::new("swarm_router", synth_idx, tot_idx, 0.70)
        .add_intent_route(vec!["memoria".into(), "gmem".into(), "recuperar".into(), "documento".into()], SwarmIntent::MemoryRAG, rag_idx)
        .add_intent_route(vec!["calcular".into(), "math".into(), "+".into(), "*".into()], SwarmIntent::ToolExecution, tool_idx)
        .add_intent_route(vec!["deducir".into(), "multi-salto".into(), "analizar".into(), "needle".into()], SwarmIntent::DeepReasoning, tot_idx);
    let router_idx = graph.add_node(Arc::new(router));

    let executor = SwarmExecutor::new(Arc::new(graph));

    // Ejecutar 3 casos de prueba empíricos
    let queries = [
        "¿Qué información de memoria persistente .gmem tenemos sobre GAJE?",
        "calcular 1024 * 64 + 512",
        "Deducir el needle multi-salto del genoma toroidal analizando la relación con K-WTA",
    ];

    for (i, query) in queries.iter().enumerate() {
        println!("\n------------------------------------------------------------------");
        println!("🔍 Caso #{}: \"{}\"", i + 1, query);
        let (state, hops, elapsed_ms) = executor.execute_profiled(router_idx, AgentState::with_query(*query))
            .map_err(|e| format!("{:?}", e))?;

        println!("  • Intención  : {}", state.intent.as_deref().unwrap_or("Direct"));
        println!("  • Pasos/Hops : {}", hops);
        println!("  • Latencia   : {:.2} ms", elapsed_ms);
        println!("  • Respuesta  :\n{}", state.response.as_deref().unwrap_or(""));
    }

    println!("==================================================================");
    println!("✅ Demostración de Enjambre Agéntico GAJE (Fase 4d) completada con éxito.");
    println!("==================================================================");
    Ok(())
}
