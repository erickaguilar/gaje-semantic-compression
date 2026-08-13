use _impl::compute::scheduler::NeuromorphicScheduler;
use _impl::core::evolution_bitwise::SpikingEvolutionEngine;
use _impl::nn::spiking::GajeNeuromorphicLayer;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::time::Instant;

fn main() {
    let dataset_path = "data/datasets/dataset_born_2000.txt";
    let c_r = [-1.5, -0.5, 0.5, 1.5];
    let c_im = [0.0, 0.0, 0.0, 0.0];

    println!("🧬 GAJE-Flow: Iniciando Crianza Just-In-Time (JIT) para el Chat...");

    // 1. Carga y Vocabulario
    let content = fs::read_to_string(dataset_path).expect("Error al leer el dataset");
    let mut word_to_id = HashMap::new();
    let mut id_to_word = Vec::new();
    let mut id_counter = 0;
    let mut sequences = Vec::new();

    for line in content.lines() {
        let mut seq = Vec::new();
        for word in line.split_whitespace() {
            let clean = word
                .to_lowercase()
                .replace(".", "")
                .replace(",", "")
                .replace("?", "")
                .replace("¿", "");
            let id = *word_to_id.entry(clean.clone()).or_insert_with(|| {
                let id = id_counter;
                id_to_word.push(clean);
                id_counter += 1;
                id
            });
            seq.push(id);
        }
        if !seq.is_empty() {
            sequences.push(seq);
        }
    }

    // 2. Crianza Rápida (100 Generaciones)
    let hidden_dim = 128;
    let l0 = GajeNeuromorphicLayer::new(hidden_dim, id_counter, 0.05, 0.8);
    let l1 = GajeNeuromorphicLayer::new(id_counter, hidden_dim, 0.05, 0.8);
    let mut layers = vec![l0, l1];

    let mut engine = SpikingEvolutionEngine::new(layers.clone(), 64, c_r, c_im, 0.1);

    let input_spikes: Vec<(usize, usize)> =
        sequences.iter().take(100).map(|seq| (0, seq[0])).collect();
    let targets = vec![1.0; input_spikes.len()];

    print!("🔥 Incubando organismo...");
    io::stdout().flush().unwrap();
    let start_train = Instant::now();
    for _ in 0..100 {
        engine.evaluate(&input_spikes, &targets);
        engine.evolve();
    }
    layers = engine.population[0].layers.clone();
    println!(" ¡Nacido en {:?}!", start_train.elapsed());

    println!("\n--- 🗨️ CHAT GENÓMICO (BORN-GENOMIC) ---");
    println!("(Escribe 'salir' para terminar)");

    loop {
        print!("\nUsuario: ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input == "salir" {
            break;
        }

        let mut words: Vec<String> = input
            .split_whitespace()
            .map(|s| {
                s.to_lowercase()
                    .replace(".", "")
                    .replace(",", "")
                    .replace("?", "")
                    .replace("¿", "")
            })
            .collect();

        print!("Asistente:");

        // Inferencia Autoregresiva Genómica
        for _ in 0..15 {
            // Generar hasta 15 palabras
            let mut scheduler = NeuromorphicScheduler::new(c_r, c_im, 1);

            // Inyectar contexto (última palabra)
            if let Some(last_word) = words.last() {
                if let Some(&id) = word_to_id.get(last_word) {
                    scheduler.inject_spike(0, id, 0, 0, 1.0);
                } else {
                    // Si no conoce la palabra, inyectar una aleatoria del vocabulario
                    let rand_id = rand::random::<usize>() % id_counter;
                    scheduler.inject_spike(0, rand_id, 0, 0, 1.0);
                }
            }

            let output_spikes = scheduler.run_to_completion(&mut layers);

            if let Some(next_event) = output_spikes.last() {
                let next_id = next_event.source_neuron_id;
                if next_id < id_counter {
                    let next_word = &id_to_word[next_id];
                    print!(" {}", next_word);
                    io::stdout().flush().unwrap();
                    words.push(next_word.clone());

                    // Pequeña pausa para efecto de "escritura"
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            } else {
                // Si no hay spikes, el organismo "se queda en blanco"
                print!(" ...");
                break;
            }
        }
        println!();
    }
}
