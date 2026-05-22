use _impl::nn::spiking::GajeNeuromorphicLayer;
use _impl::core::evolution_bitwise::SpikingEvolutionEngine;
use _impl::compute::scheduler::NeuromorphicScheduler;
use std::collections::HashMap;

fn main() {
    println!("🧬 Gaje Identity Cloner - Búsqueda de Resonancia Total (1.00 Fitness)");
    
    // 1. Dataset Real
    let sentences = vec![
        "Rust es un lenguaje de programación.",
        "El ADN contiene información genética.",
        "GAJE permite compresión extrema."
    ];

    // 2. Tokenización
    let mut word_to_id = HashMap::new();
    let mut id_counter = 0;
    for s in &sentences {
        for word in s.split_whitespace() {
            let clean = word.to_lowercase().replace(".", "");
            word_to_id.entry(clean).or_insert_with(|| {
                let id = id_counter;
                id_counter += 1;
                id
            });
        }
    }

    let first_words: Vec<usize> = sentences.iter()
        .map(|s| *word_to_id.get(&s.split_whitespace().next().unwrap().to_lowercase()).unwrap())
        .collect();

    println!("   Vocabulario: {} palabras.", id_counter);

    // 3. Configuración del Motor Industrial (SoA)
    let dim = 128; 
    let centroides = [-1.0, -0.2, 0.2, 1.0];
    
    // Crear capas SoA
    // L0 ahora es una capa densa que recibe el ID de la palabra como un spike broadcast
    let l0 = GajeNeuromorphicLayer::new(id_counter, id_counter, 0.05, 0.8);
    let l1 = GajeNeuromorphicLayer::new(dim, id_counter, 0.05, 0.8);
    let layers = vec![l0, l1];

    let mut engine = SpikingEvolutionEngine::new(layers, 200, centroides, 0.3);

    // 4. Evolución Intensiva
    println!("\n🚀 Evolución Intensiva (Población: 200, Gen: 500) buscando 1.00 Fitness...");

    for gen in 0..=500 {
        let input_spikes: Vec<(usize, usize)> = first_words.iter().map(|&id| (0, id)).collect();
        let targets = vec![1.0; first_words.len()];
        
        engine.evaluate(&input_spikes, &targets);

        let best_fitness = engine.population[0].fitness;
        if gen % 100 == 0 {
            println!("   [Gen {:3}] Fitness: {:.4}", gen, best_fitness);
        }

        if best_fitness >= 1.0 {
            println!("   ✨ ¡HITO ALCANZADO! 1.00 Fitness en la generación {}.", gen);
            break;
        }

        if gen < 500 {
            engine.evolve();
        }
    }

    // 5. Verificación de "Pensamiento" Industrial
    println!("\n🧠 Verificación de Inferencia (Palabras Semilla):");
    let best_organism = &mut engine.population[0];
    
    for word in &["rust", "el", "gaje"] {
        let mut scheduler = NeuromorphicScheduler::new(centroides, 1);
        let id = *word_to_id.get(*word).unwrap();
        
        // En SoA, el id de la palabra activa una neurona específica de la capa de entrada
        scheduler.inject_spike(0, id, 0, 0, 1.0);
        
        let outputs = scheduler.run_to_completion(&mut best_organism.layers);
        println!("   Al procesar '{}', la red generó {} disparos.", word, outputs.len());
    }
}
