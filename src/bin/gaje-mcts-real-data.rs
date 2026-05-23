use _impl::compute::mcts::MctsTree;
use _impl::compute::math::calculate_genomic_mse;
use std::time::Instant;
use std::fs;

fn main() {
    println!("🧪 Validando MCTS con Dataset Real: tiny_shakespeare.txt");

    // 1. Cargar el dataset y convertirlo en "pesos" simulados (bytes -> f32 normalizado)
    let path = "data/datasets/tiny_shakespeare.txt";
    let content = fs::read(path).expect("No se pudo leer el dataset");
    
    // Tomamos una muestra para la optimización (ej. primeros 20,000 bytes)
    let weights: Vec<f32> = content.iter()
        .take(20000)
        .map(|&b| (b as f32 / 255.0) - 0.5) // Normalizar a rango [-0.5, 0.5]
        .collect();

    println!("   Dataset cargado: {} muestras", weights.len());

    // 2. Estado inicial (Centroides clásicos - asunción Gaussiana)
    let initial_centroids = vec![-0.4, -0.15, 0.15, 0.4];
    let initial_mse = calculate_genomic_mse(weights.clone(), initial_centroids.clone());
    println!("   MSE Inicial (Clásico): {:.6}", initial_mse);

    // 3. Ejecutar MCTS
    let mut mcts = MctsTree::new(initial_centroids.clone(), 1.0);
    let iterations = 10000; // Aumentamos iteraciones para el dataset real
    let start_time = Instant::now();

    for i in 0..iterations {
        let selected_node_idx = mcts.select(0, 1.41);

        if mcts.nodes[selected_node_idx].n_visits > 0 {
            mcts.expand(selected_node_idx, 4, 0.04);
            let last_idx = mcts.nodes.len() - 1;
            let centroids = mcts.nodes[last_idx].state.clone();
            let mse = calculate_genomic_mse(weights.clone(), centroids);
            mcts.backpropagate(last_idx, 1.0 / (mse + 1e-10));
        } else {
            let centroids = mcts.nodes[selected_node_idx].state.clone();
            let mse = calculate_genomic_mse(weights.clone(), centroids);
            mcts.backpropagate(selected_node_idx, 1.0 / (mse + 1e-10));
        }

        if (i + 1) % 2000 == 0 {
            println!("   [Iter {}] Explorando...", i + 1);
        }
    }

    let duration = start_time.elapsed();
    
    // 4. Encontrar el mejor resultado
    let mut best_node_idx = 0;
    let mut max_q = -1.0;
    for (idx, node) in mcts.nodes.iter().enumerate() {
        if node.q_value > max_q && node.n_visits > 0 {
            max_q = node.q_value;
            best_node_idx = idx;
        }
    }

    let best_centroids = &mcts.nodes[best_node_idx].state;
    let final_mse = calculate_genomic_mse(weights.clone(), best_centroids.clone());

    println!("\n✅ Validación completada en {:?}", duration);
    println!("   Centroides Óptimos: {:?}", best_centroids);
    println!("   MSE Final (MCTS): {:.6}", final_mse);
    println!("   Mejora: {:.2}%", ((initial_mse - final_mse) / initial_mse) * 100.0);
}
