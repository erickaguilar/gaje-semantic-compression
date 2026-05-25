use _impl::io::loader::NativeLoader;
use _impl::compute::mcts::MctsTree;
use _impl::compute::math::calculate_genomic_mse;
use std::time::Instant;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: gaje-mcts-optimize <model_path> [--iterations 5000]");
        return Ok(());
    }

    let model_path = &args[1];
    let mut iterations = 5000;
    if args.len() > 3 && args[2] == "--iterations" {
        iterations = args[3].parse().unwrap_or(5000);
    }

    println!("🎲 Iniciando Optimización MCTS-Genómica para: {}", model_path);

    // 1. Cargar el modelo
    let loader = NativeLoader::new(model_path)?;
    let model = loader.load_llm()?;
    
    // Para la validación del Gold Embryo, optimizaremos los centroides de la capa de embeddings
    // ya que es la base del vocabulario y la más crítica para la coherencia inicial.
    let target_layer = &model.embeddings;
    println!("   Optimizando capa: token_embd ({} x {})", target_layer.out_features, target_layer.in_features);

    // 2. Extraer datos para evaluación (Simulamos una distribución objetivo basada en los centroides actuales)
    // En una implementación de producción, aquí usaríamos los pesos F32 originales si estuvieran disponibles,
    // o muestras del dataset para maximizar la resonancia.
    let initial_centroids = target_layer.centroids[0..4].to_vec();
    
    // 3. Configurar Árbol MCTS
    let mut mcts = MctsTree::new(initial_centroids.clone(), 1.0);
    let start_time = Instant::now();

    // Simulamos un conjunto de pesos "ideales" para que el MCTS tenga un objetivo de convergencia.
    // En el futuro, esto se conectará con el Activation Drift del maestro.
    let weights = vec![0.0f32; 1000]; // Placeholder para pesos de referencia

    for i in 0..iterations {
        let selected_node_idx = mcts.select(0, 1.41);

        if mcts.nodes[selected_node_idx].n_visits > 0 {
            mcts.expand(selected_node_idx, 4, 0.1);
            let last_idx = mcts.nodes.len() - 1;
            evaluate_and_backpropagate(&mut mcts, last_idx, &weights);
        } else {
            evaluate_and_backpropagate(&mut mcts, selected_node_idx, &weights);
        }

        if (i + 1) % 1000 == 0 {
            let best_score = mcts.nodes[0].q_value;
            println!("   [Iter {}] Mejor Score de Resonancia: {:.6}", i + 1, best_score);
        }
    }

    let duration = start_time.elapsed();
    
    // 4. Encontrar los mejores centroides
    let mut best_node_idx = 0;
    let mut max_q = -1.0;
    for (idx, node) in mcts.nodes.iter().enumerate() {
        if node.q_value > max_q && node.n_visits > 0 {
            max_q = node.q_value;
            best_node_idx = idx;
        }
    }

    let best_centroids = &mcts.nodes[best_node_idx].state;

    println!("\n✅ Optimización completada en {:?}", duration);
    println!("   Centroides Originales: {:?}", initial_centroids);
    println!("   Centroides Optimizados: {:?}", best_centroids);
    println!("   Mejora en Score: {:.2}%", ((max_q - mcts.nodes[0].q_value) / mcts.nodes[0].q_value.max(1e-6)) * 100.0);
    
    println!("\n[!] Nota: En la v1.0, estos centroides se guardarán automáticamente en el archivo .gaje.");

    Ok(())
}

fn evaluate_and_backpropagate(tree: &mut MctsTree, node_idx: usize, _weights: &[f32]) {
    let centroids = &tree.nodes[node_idx].state;
    
    // Heurística de fitness: premiar la dispersión y el equilibrio de los centroides
    // para evitar el colapso de la señal en 2 bits.
    let mut score = 0.0;
    
    // 1. Penalizar centroides demasiado cercanos (colisión)
    for i in 0..3 {
        let diff = (centroids[i+1] - centroids[i]).abs();
        if diff < 0.2 { score -= 10.0; }
        else { score += diff * 5.0; }
    }
    
    // 2. Premiar simetría respecto al cero
    let symmetry = (centroids[0] + centroids[3]).abs() + (centroids[1] + centroids[2]).abs();
    score += 1.0 / (symmetry + 0.1);

    tree.backpropagate(node_idx, score);
}
