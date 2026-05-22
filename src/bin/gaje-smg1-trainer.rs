use _impl::nn::spiking::layer::GajeNeuromorphicLayer;
use _impl::compute::scheduler::NeuromorphicScheduler;
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    println!("🧬 GAJE-Flow: Inicializando Micro-Genoma Estándar (SMG-1)");
    println!("   Arquitectura: 3 Capas (Latente: 256, Lógica: 128)");
    
    // 1. Dataset: Un texto con más complejidad estructural
    let text = "el protocolo gaje es un motor neuromorfico soberano que utiliza adn de dos bits para procesar lenguaje natural en dispositivos moviles con alta eficiencia energetica y coherencia semantica local";
    let words: Vec<&str> = text.split_whitespace().collect();
    
    let mut word_to_id = HashMap::new();
    let mut id_to_word = Vec::new();
    let mut vocab_size = 0;
    
    for &word in &words {
        word_to_id.entry(word).or_insert_with(|| {
            let id = vocab_size;
            id_to_word.push(word);
            vocab_size += 1;
            id
        });
    }
    
    println!("   Vocabulario: {} tokens únicos.", vocab_size);

    // 2. Construcción del SMG-1
    let dim_latent = 256;
    let dim_logic = 128;
    let centroides = [-1.5, -0.5, 0.5, 1.5];
    
    // Capas
    let l0 = GajeNeuromorphicLayer::new(dim_latent, vocab_size, 0.4, 0.8);
    let l1 = GajeNeuromorphicLayer::new(dim_logic, dim_latent, 0.4, 0.8);
    let l2 = GajeNeuromorphicLayer::new(vocab_size, dim_logic, 0.4, 0.8);
    let mut layers = vec![l0, l1, l2];

    // 3. Entrenamiento SMG-1
    let epochs = 300;
    let lr = 0.5;
    let start = Instant::now();
    
    println!("\n🔥 Entrenando SMG-1 (Resonancia Tri-Capa)...");

    for epoch in 1..=epochs {
        let mut total_hits = 0;
        
        for i in 0..words.len() - 1 {
            let input_id = word_to_id[words[i]];
            let target_id = word_to_id[words[i+1]];
            
            // RESET
            for layer in &mut layers { layer.membrane_potentials.fill(0.0); }

            // 1. REFORZAR L0 (Input -> Latent)
            // Asignamos un patrón robusto (16 neuronas) por cada input
            let mut l0_deltas = vec![-0.1; dim_latent];
            let offset_l0 = (input_id * 16) % dim_latent;
            for j in 0..16 { l0_deltas[(offset_l0 + j) % dim_latent] = 1.0; }
            layers[0].refine_step(input_id, &l0_deltas, 1.0);

            // 2. FORWARD L0
            layers[0].integrate_batch(input_id, &centroides, 1.0);
            let s0 = layers[0].check_spikes();
            
            // 3. REFORZAR L1 (Latent -> Logic)
            // Hacemos que la capa lógica responda con un patrón denso si L0 disparó
            let mut l1_deltas = vec![-0.1; dim_logic];
            let offset_l1 = (input_id * 8) % dim_logic;
            for j in 0..8 { l1_deltas[(offset_l1 + j) % dim_logic] = 1.0; }
            for &(idx, _, _) in &s0 {
                layers[1].refine_step(idx, &l1_deltas, 1.0);
            }
            
            // 4. FORWARD L1
            for &(idx, intensity, _) in &s0 {
                layers[1].integrate_batch(idx, &centroides, intensity);
            }
            let s1 = layers[1].check_spikes();

            // 5. REFORZAR L2 (Logic -> Output)
            let mut l2_deltas = vec![-1.0; vocab_size];
            l2_deltas[target_id] = 1.0;
            for &(idx, _, _) in &s1 {
                layers[2].refine_step(idx, &l2_deltas, lr);
            }

            // 6. FORWARD L2 (Final check for hit)
            for &(idx, intensity, _) in &s1 {
                layers[2].integrate_batch(idx, &centroides, intensity);
            }
            let s2 = layers[2].check_spikes();

            if s2.iter().any(|&(idx, _, _)| idx == target_id) {
                total_hits += 1;
            }
            
            // Homeostasis
            for layer in &mut layers { layer.apply_homeostasis(2.0); }
        }
        
        if epoch % 50 == 0 || epoch == 1 {
            let accuracy = (total_hits as f32 / (words.len() - 1) as f32) * 100.0;
            println!("   [Epoch {:3}] Precisión: {:.2}%", epoch, accuracy);
        }
    }

    println!("\n✅ Entrenamiento SMG-1 finalizado en {:?}", start.elapsed());

    // 4. Generación
    println!("\n🗨️ Generación SMG-1 (Seed: '{}'):", words[0]);
    let mut current_id = word_to_id[words[0]];
    print!("{}", id_to_word[current_id]);
    
    for _ in 0..words.len() + 2 {
        for layer in &mut layers { layer.membrane_potentials.fill(0.0); }
        
        layers[0].integrate_batch(current_id, &centroides, 1.0);
        let s0 = layers[0].check_spikes();
        
        for &(idx, intensity, _) in &s0 {
            layers[1].integrate_batch(idx, &centroides, intensity);
        }
        let s1 = layers[1].check_spikes();
        
        for &(idx, intensity, _) in &s1 {
            layers[2].integrate_batch(idx, &centroides, intensity);
        }
        let s2 = layers[2].check_spikes();
        
        if let Some(&(next_id, _, _)) = s2.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
            if next_id < vocab_size {
                print!(" {}", id_to_word[next_id]);
                current_id = next_id;
            } else { break; }
        } else {
            // Intento de recuperación por potencial máximo
            let mut max_p = 0.0;
            let mut max_idx = 0;
            for (idx, &p) in layers[2].membrane_potentials.iter().enumerate() {
                if p > max_p { max_p = p; max_idx = idx; }
            }
            if max_p > 0.0 && max_idx < vocab_size {
                print!(" [{}]", id_to_word[max_idx]);
                current_id = max_idx;
            } else { break; }
        }
    }
    println!("\n");
}
