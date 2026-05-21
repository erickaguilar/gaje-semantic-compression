use _impl::nn::spiking::{SpikingNeuron, GajeWeight2Bit};
use _impl::core::evolution_bitwise::SpikingEvolutionEngine;
use std::fs;
use std::time::Instant;
use std::collections::HashMap;

fn main() {
    // 1. Configuración de Parámetros (Simulando gaje_train.sh)
    let dataset_path = "data/datasets/dataset_1000.txt";
    let pop_size = 50;
    let generations = 100;
    let dim = 64;
    let mutation_rate = 0.2;
    let centroides = [-1.0, -0.2, 0.2, 1.0];

    println!("🧬 GAJE-Flow: Iniciando Nacimiento de Modelo Neuromórfico");
    println!("   Dataset: {}", dataset_path);
    println!("   Población: {}, Gen: {}, Dim: {}", pop_size, generations, dim);

    // 2. Carga y Tokenización del Dataset
    let content = fs::read_to_string(dataset_path).expect("Error al leer el dataset");
    let mut word_to_id = HashMap::new();
    let mut id_counter = 0;
    let mut sequences = Vec::new();

    for line in content.lines().take(50) { // Tomar 50 líneas para la demo
        let mut seq = Vec::new();
        for word in line.split_whitespace() {
            let clean = word.to_lowercase().replace(".", "").replace(",", "");
            let id = *word_to_id.entry(clean).or_insert_with(|| {
                let id = id_counter;
                id_counter += 1;
                id
            });
            seq.push(id);
        }
        if !seq.is_empty() {
            sequences.push(seq);
        }
    }

    println!("   Vocabulario: {} tokens únicos.", id_counter);

    // 3. Inicialización del Modelo "Born"
    // L0: Capa de Entrada (Embeddings de Spikes)
    // L1: Capa de Procesamiento Neuromórfico
    let mut layers = vec![
        Vec::new(), // L0
        Vec::new(), // L1
    ];

    for _ in 0..id_counter {
        layers[0].push(SpikingNeuron::new(0.05, 0.8, 1)); 
    }
    for _ in 0..dim {
        layers[1].push(SpikingNeuron::new(0.05, 0.8, id_counter));
    }

    let mut engine = SpikingEvolutionEngine::new(layers, pop_size, centroides, mutation_rate);

    // 4. Protocolo de Entrenamiento (Evolución)
    println!("\n🔥 Iniciando Entrenamiento Genómico (Resonancia Bitwise)...");
    let start = Instant::now();

    let input_spikes: Vec<(usize, usize)> = sequences.iter()
        .map(|seq| (0, seq[0])) // Inyectar la primera palabra de cada secuencia
        .collect();
    let targets = vec![1.0; sequences.len()];

    for gen in 0..=generations {
        engine.evaluate(&input_spikes, &targets);

        if gen % 20 == 0 {
            let best_fitness = engine.population[0].fitness;
            println!("   [Gen {:3}] Fitness: {:.4} | Tiempo: {:?}", gen, best_fitness, start.elapsed());
        }

        if gen < generations {
            engine.evolve();
        }
    }

    let duration = start.elapsed();
    let best = &engine.population[0];

    // 5. Finalización y Exportación (Simulada)
    println!("\n✅ Entrenamiento Completado en {:?}", duration);
    println!("   Fitness Final: {:.4}", best.fitness);
    println!("   Modelo 'Born' generado exitosamente.");
    println!("   Pesos de 2-bits optimizados para el dataset.");

    // Aquí se llamaría a NativeLoader::save_genomic_model en una versión completa
    println!("\n[*] Modelo guardado virtualmente como: born_neuromorphic_v1.gaje");
}
