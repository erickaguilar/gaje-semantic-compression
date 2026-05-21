use _impl::nn::spiking::GajeNeuromorphicLayer;
use _impl::core::evolution_bitwise::SpikingEvolutionEngine;
use std::fs;
use std::time::Instant;
use std::collections::HashMap;

fn main() {
    // 1. Configuración de Parámetros (Ajustada para Born-Genomic)
    let dataset_path = "data/datasets/dataset_born_2000.txt";
    let pop_size = 64;
    let generations = 200;
    let mutation_rate = 0.15;
    let centroides = [-1.5, -0.5, 0.5, 1.5]; // Centroides Born estándar

    println!("🧬 GAJE-Flow: Iniciando Crianza de Micro-Modelo Born-Genomic");
    println!("   Dataset: {}", dataset_path);
    println!("   Estrategia: Evolución Masiva Bitwise (Industrial SoA)");
    println!("   Población: {}, Gen: {}", pop_size, generations);

    // 2. Carga y Tokenización del Dataset
    let content = fs::read_to_string(dataset_path).expect("Error al leer el dataset");
    let mut word_to_id = HashMap::new();
    let mut id_counter = 0;
    let mut sequences = Vec::new();

    // Procesar dataset completo para construir el vocabulario
    for line in content.lines() {
        let mut seq = Vec::new();
        for word in line.split_whitespace() {
            let clean = word.to_lowercase().replace(".", "").replace(",", "").replace("?", "").replace("¿", "");
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
    println!("   Secuencias cargadas: {}.", sequences.len());

    // 3. Inicialización del Modelo Industrial (SoA)
    // Usamos una dimensión oculta pequeña (128) para rapidez en el nacimiento
    let hidden_dim = 128;
    let l0 = GajeNeuromorphicLayer::new(hidden_dim, id_counter, 0.05, 0.8);
    let l1 = GajeNeuromorphicLayer::new(id_counter, hidden_dim, 0.05, 0.8);
    let layers = vec![l0, l1];

    let mut engine = SpikingEvolutionEngine::new(layers, pop_size, centroides, mutation_rate);

    // 4. Protocolo de Entrenamiento (Resonancia Semántica)
    println!("\n🔥 Iniciando Simulación Evolutiva...");
    let start = Instant::now();

    // Preparar inputs de prueba (Primeros 100 tokens para resonancia inicial)
    let input_spikes: Vec<(usize, usize)> = sequences.iter()
        .take(100)
        .map(|seq| (0, seq[0]))
        .collect();
    let targets = vec![1.0; input_spikes.len()];

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

    // 5. Finalización
    println!("\n✅ Crianza Finalizada en {:?}", duration);
    println!("   Fitness Final: {:.4}", best.fitness);
    println!("   El organismo ha nacido exitosamente en el motor SoA.");
}
