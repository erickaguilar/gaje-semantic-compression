use _impl::nn::spiking::GajeNeuromorphicLayer;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::time::Instant;

fn main() {
    // 1. Configuración de Parámetros (Estándar SMG-1)
    let dataset_path = "data/datasets/dataset_born_2000.txt";
    let generations = 150;
    let lr = 0.5;
    let centroides = [-1.5, -0.5, 0.5, 1.5];

    println!("🧬 GAJE-Flow: Estandarizando Micro-Genoma (Arquitectura SMG-1)");
    println!("   Dataset: {}", dataset_path);
    println!("   Estrategia: Refinamiento Genómico Local (Nativo)");
    println!("   Configuración: 3 Capas (256/128), Gen: {}", generations);

    // 2. Carga y Tokenización del Dataset
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

    println!("   Vocabulario: {} tokens únicos.", id_counter);
    println!("   Secuencias cargadas: {}.", sequences.len());

    // 3. Inicialización SMG-1 (Standard Micro-Genome)
    let dim_latent = 256;
    let dim_logic = 128;

    let l0 = GajeNeuromorphicLayer::new(dim_latent, id_counter, 0.4, 0.8);
    let l1 = GajeNeuromorphicLayer::new(dim_logic, dim_latent, 0.4, 0.8);
    let l2 = GajeNeuromorphicLayer::new(id_counter, dim_logic, 0.4, 0.8);
    let mut layers = vec![l0, l1, l2];

    // 4. Protocolo de Entrenamiento Nativo (Life-long Learning)
    println!("\n🔥 Iniciando Nacimiento Genómico...");
    let start = Instant::now();

    // Solo tomamos una muestra representativa para el nacimiento rápido
    let train_sequences: Vec<Vec<usize>> = sequences.iter().take(200).cloned().collect();

    for gen in 1..=generations {
        let mut total_hits = 0;
        let mut total_tokens = 0;

        for seq in &train_sequences {
            for i in 0..seq.len() - 1 {
                let input_id = seq[i];
                let target_id = seq[i + 1];
                total_tokens += 1;

                // Forward/Refinement Step (SMG-1 Logic)
                for layer in &mut layers {
                    layer.reset_potentials();
                }

                // REFORZAR L0
                let mut l0_deltas = vec![-0.1; dim_latent];
                let offset_l0 = (input_id * 16) % dim_latent;
                for j in 0..16 {
                    l0_deltas[(offset_l0 + j) % dim_latent] = 1.0;
                }
                layers[0].refine_step(input_id, l0_deltas, 1.0);

                layers[0].integrate_batch(input_id, centroides, [0.0; 4], 1.0);
                let s0 = layers[0].check_spikes();

                // REFORZAR L1
                let mut l1_deltas = vec![-0.1; dim_logic];
                let offset_l1 = (input_id * 8) % dim_logic;
                for j in 0..8 {
                    l1_deltas[(offset_l1 + j) % dim_logic] = 1.0;
                }
                for &(idx, _, _) in &s0 {
                    layers[1].refine_step(idx, l1_deltas.clone(), 1.0);
                }

                for &(idx, intensity, _) in &s0 {
                    layers[1].integrate_batch(idx, centroides, [0.0; 4], intensity);
                }
                let s1 = layers[1].check_spikes();

                // REFORZAR L2 (Output)
                let mut l2_deltas = vec![-1.0; id_counter];
                l2_deltas[target_id] = 1.0;
                for &(idx, _, _) in &s1 {
                    layers[2].refine_step(idx, l2_deltas.clone(), lr);
                }

                for &(idx, intensity, _) in &s1 {
                    layers[2].integrate_batch(idx, centroides, [0.0; 4], intensity);
                }
                let s2 = layers[2].check_spikes();

                if s2.iter().any(|&(idx, _, _)| idx == target_id) {
                    total_hits += 1;
                }

                for layer in &mut layers {
                    layer.apply_homeostasis(2.0);
                }
            }
        }

        if gen % 30 == 0 || gen == 1 {
            let accuracy = (total_hits as f32 / total_tokens as f32) * 100.0;
            println!(
                "   [Gen {:3}] Precisión de Resonancia: {:.2}% | Tiempo: {:?}",
                gen,
                accuracy,
                start.elapsed()
            );
            io::stdout().flush().unwrap();
        }
    }

    println!("\n✅ Micro-Genoma SMG-1 nacido en {:?}", start.elapsed());
    println!("   El organismo ha sido estandarizado y está listo para evolución continua.");
}
