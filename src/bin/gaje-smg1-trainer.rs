use _impl::nn::spiking::layer::GajeNeuromorphicLayer;
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dataset_path = if args.len() > 2 && args[1] == "--dataset" { Some(&args[2]) } else { None };

    println!("🧬 GAJE-Flow: Inicializando Micro-Genoma Estándar (SMG-1)");
    
    // 1. Dataset: Carga desde archivo o texto por defecto
    let text = if let Some(path) = dataset_path {
        std::fs::read_to_string(path).expect("No se pudo leer el dataset")
    } else {
        "el protocolo gaje es un motor neuromorfico soberano que utiliza adn de dos bits para procesar lenguaje natural en dispositivos moviles con alta eficiencia energetica y coherencia semantica local".to_string()
    };
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

    // 2. Construcción del SMG-2 (Calibrado para selectividad)
    let dim_latent = 512;
    let dim_logic = 256;
    let centroides = [-1.5, -0.5, 0.5, 1.5];
    
    let l0 = GajeNeuromorphicLayer::new(dim_latent, vocab_size, 0.4, 0.8);
    let l1 = GajeNeuromorphicLayer::new(dim_logic, dim_latent, 0.4, 0.8);
    let l2 = GajeNeuromorphicLayer::new(vocab_size, dim_logic, 0.4, 0.8);
    let mut layers = vec![l0, l1, l2];

    // 3. Entrenamiento SMG-2
    let epochs = 150;
    let start = Instant::now();
    
    println!("\n🔥 Entrenando SMG-2 (Refuerzo Diferencial)...");

    for epoch in 1..=epochs {
        let mut total_hits = 0;
        
        for i in 0..words.len() - 1 {
            let input_id = word_to_id[words[i]];
            let target_id = word_to_id[words[i+1]];
            
            for layer in &mut layers { layer.membrane_potentials.fill(0.0); }

            // 1. REFORZAR L0
            let mut l0_deltas = vec![-0.5; dim_latent];
            for j in 0..32 { 
                let idx = (input_id * 17 + j) % dim_latent;
                l0_deltas[idx] = 1.5; 
            }
            layers[0].refine_step(input_id, l0_deltas, 1.0);

            // 2. FORWARD L0
            layers[0].integrate_batch(input_id, centroides, 1.0);
            let mut s0 = layers[0].check_spikes();
            s0.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let s0 = if s0.len() > 8 { s0[..8].to_vec() } else { s0 };
            
            // 3. REFORZAR L1
            let mut l1_deltas = vec![-0.5; dim_logic];
            for j in 0..16 { 
                let idx = (input_id * 13 + j) % dim_logic;
                l1_deltas[idx] = 1.5; 
            }
            for &(idx, _, _) in &s0 {
                layers[1].refine_step(idx, l1_deltas.clone(), 1.0);
            }
            
            // 4. FORWARD L1
            for &(idx, intensity, _) in &s0 {
                layers[1].integrate_batch(idx, centroides, intensity);
            }
            let mut s1 = layers[1].check_spikes();
            s1.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let s1 = if s1.len() > 8 { s1[..8].to_vec() } else { s1 };

            // 5. REFORZAR L2 (Diferencial)
            let mut l2_deltas = vec![-1.0; vocab_size];
            l2_deltas[target_id] = 2.0;
            for &(idx, _, _) in &s1 {
                layers[2].refine_step(idx, l2_deltas.clone(), 1.0);
            }

            // 6. FORWARD L2
            for &(idx, intensity, _) in &s1 {
                layers[2].integrate_batch(idx, centroides, intensity);
            }
            let mut s2 = layers[2].check_spikes();
            s2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let s2 = if s2.len() > 4 { s2[..4].to_vec() } else { s2 };

            if s2.iter().any(|&(idx, _, _)| idx == target_id) {
                total_hits += 1;
            }
            for layer in &mut layers { layer.apply_homeostasis(1.5); }
        }
        
        if epoch % 10 == 0 || epoch == 1 {
            let accuracy = (total_hits as f32 / (words.len() - 1) as f32) * 100.0;
            println!("   [Epoch {:3}] Precisión: {:.2}%", epoch, accuracy);
        }
    }

    println!("\n✅ Entrenamiento SMG-2 finalizado en {:?}", start.elapsed());

    // 4. Generación
    println!("\n🗨️ Generación SMG-2 (Seed: '{}'):", words[0]);
    let mut current_id = word_to_id[words[0]];
    print!("{}", id_to_word[current_id]);
    
    for _ in 0..words.len() + 2 {
        for layer in &mut layers { layer.membrane_potentials.fill(0.0); }
        layers[0].integrate_batch(current_id, centroides, 1.0);
        let mut s0 = layers[0].check_spikes();
        s0.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let s0 = if s0.len() > 8 { s0[..8].to_vec() } else { s0 };
        
        for &(idx, intensity, _) in &s0 { layers[1].integrate_batch(idx, centroides, intensity); }
        let mut s1 = layers[1].check_spikes();
        s1.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let s1 = if s1.len() > 8 { s1[..8].to_vec() } else { s1 };
        
        for &(idx, intensity, _) in &s1 { layers[2].integrate_batch(idx, centroides, intensity); }
        let mut s2 = layers[2].check_spikes();
        s2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        if let Some(&(next_id, _, _)) = s2.first() {
            if next_id < vocab_size {
                print!(" {}", id_to_word[next_id]);
                current_id = next_id;
            } else { break; }
        } else {
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
