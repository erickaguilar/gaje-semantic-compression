use _impl::nn::spiking::layer::GajeNeuromorphicLayer;
use _impl::compute::scheduler::NeuromorphicScheduler;
use std::time::Instant;

fn main() {
    println!("🧬 GAJE-Flow: Demo de Entrenamiento Genómico Nativo (Life-long Learning)");
    println!("   Objetivo: Aprender una asociación sin usar evolución poblacional.");

    let centroides = [-1.5, -0.5, 0.5, 1.5];
    let vocab_size = 10;
    let hidden_dim = 16;
    
    // 1. Inicializar un único organismo
    let l0 = GajeNeuromorphicLayer::new(hidden_dim, vocab_size, 0.5, 0.9);
    let l1 = GajeNeuromorphicLayer::new(vocab_size, hidden_dim, 0.5, 0.9);
    let mut layers = vec![l0, l1];

    // 2. Tarea: Asociar Input 3 con Output 7
    let input_id = 3;
    let target_output = 7;
    let learning_rate = 0.5;
    let epochs = 50;

    println!("🔥 Entrenando asociación local: Input {} -> Output {}...", input_id, target_output);
    let start = Instant::now();

    for epoch in 1..=epochs {
        let mut scheduler = NeuromorphicScheduler::new(centroides, [0.0; 4], 1);
        scheduler.inject_spike(0, input_id, 0, 0, 1.0);
        
        let output_events = scheduler.run_to_completion(&mut layers);
        
        // Calcular Error (Delta)
        // Queremos que la neurona target_output dispare, y las demás no.
        let mut layer1_deltas = vec![-1.0; vocab_size]; // Inhibir todo por defecto
        layer1_deltas[target_output] = 1.0; // Reforzar target
        
        // Refinar Capa 1 basándose en los disparos de la Capa 0
        // Para simplificar, asumimos que todas las neuronas de la capa oculta contribuyeron
        let hidden_spikes: Vec<usize> = (0..hidden_dim).collect(); // Heurística simple
        for &h_idx in &hidden_spikes {
            layers[1].refine_step(h_idx, layer1_deltas.clone(), learning_rate);
        }

        // Refinar Capa 0: Queremos que la capa oculta responda al input
        let layer0_deltas = vec![1.0; hidden_dim];
        layers[0].refine_step(input_id, layer0_deltas, learning_rate);

        // Homeostasis para estabilidad
        layers[0].apply_homeostasis(1.0);
        layers[1].apply_homeostasis(1.0);

        if epoch % 10 == 0 || epoch == 1 {
            let hit = output_events.iter().any(|e| e.source_neuron_id == target_output);
            println!("   [Epoch {:2}] Spikes detectados: {} | Acierto: {}", epoch, output_events.len(), hit);
        }
    }

    println!("\n✅ Entrenamiento completado en {:?}", start.elapsed());
    
    // Verificación final
    let mut scheduler = NeuromorphicScheduler::new(centroides, [0.0; 4], 1);
    scheduler.inject_spike(0, input_id, 0, 0, 1.0);
    let final_outputs = scheduler.run_to_completion(&mut layers);
    let success = final_outputs.iter().any(|e| e.source_neuron_id == target_output);
    
    if success {
        println!("🚀 ¡ÉXITO! El organismo ha aprendido la asociación nativamente.");
    } else {
        println!("❌ El organismo no logró converger (se requiere ajuste de hiperparámetros).");
    }
}
