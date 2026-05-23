use _impl::compute::mcts::MctsTree;
use _impl::compute::math::calculate_genomic_mse;
use std::time::Instant;
use rand::Rng;

fn main() {
    println!("🎲 Iniciando Optimización MCTS-Genómica (Nivel AlphaZero)...");

    // 1. Generar datos de ejemplo (simulando pesos de una capa)
    let mut rng = rand::thread_rng();
    let weights: Vec<f32> = (0..10000)
        .map(|_| rng.gen_range(-0.5..0.5) + rng.gen_range(-0.1..0.1))
        .collect();

    // 2. Estado inicial (Centroides clásicos)
    let initial_centroids = vec![-0.3, -0.1, 0.1, 0.3];
    let mut mcts = MctsTree::new(initial_centroids.clone(), 1.0);

    // 3. Loop Principal de MCTS
    let iterations = 5000;
    let start_time = Instant::now();

    for i in 0..iterations {
        // A. Selección
        let selected_node_idx = mcts.select(0, 1.41);

        // B. Expansión (si el nodo es una hoja o ha sido visitado)
        if mcts.nodes[selected_node_idx].n_visits > 0 {
            mcts.expand(selected_node_idx, 4, 0.05);
            // Seleccionar uno de los nuevos hijos para evaluar
            let last_idx = mcts.nodes.len() - 1;
            evaluate_and_backpropagate(&mut mcts, last_idx, &weights);
        } else {
            evaluate_and_backpropagate(&mut mcts, selected_node_idx, &weights);
        }

        if (i + 1) % 1000 == 0 {
            let best_mse = 1.0 / mcts.nodes[0].q_value; // Inverso del score
            println!("   [Iter {}] Mejor MSE estimado (vía Q-Root): {:.6}", i + 1, best_mse);
        }
    }

    let duration = start_time.elapsed();
    
    // 4. Encontrar el mejor nodo final
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

    println!("\n✅ Optimización completada en {:?}", duration);
    println!("   Centroides Iniciales: {:?}", initial_centroids);
    println!("   Centroides Óptimos (MCTS): {:?}", best_centroids);
    println!("   MSE Final: {:.6}", final_mse);
    println!("   Nodos Explorados: {}", mcts.nodes.len());
}

fn evaluate_and_backpropagate(tree: &mut MctsTree, node_idx: usize, weights: &[f32]) {
    let centroids = tree.nodes[node_idx].state.clone();
    // Clonamos weights para la función (en producción usaríamos referencias si no fuera por PyO3)
    let mse = calculate_genomic_mse(weights.to_vec(), centroids);
    
    // Convertir MSE a Score de Fitness (mayor es mejor)
    let score = 1.0 / (mse + 1e-10);
    
    tree.backpropagate(node_idx, score);
}
