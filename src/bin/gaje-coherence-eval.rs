use _impl::nn::spiking::layer::GajeNeuromorphicLayer;
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    println!("🧬 GAJE-Flow: Test de Coherencia y Entrenamiento Secuencial (Full Evaluation)");
    
    // 1. Dataset de Prueba (Un fragmento con estructura semántica)
    let text = "la inteligencia genomica permite compresion extrema en dispositivos moviles mediante neuronas de dos bits";
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
    
    println!("   Vocabulario: {} tokens.", vocab_size);
    println!("   Secuencia: {:?}", words);

    // 2. Arquitectura del Organismo
    let hidden_dim = 128; // Mayor capacidad
    let centroides = [-1.5, -0.5, 0.5, 1.5];
    let l0 = GajeNeuromorphicLayer::new(hidden_dim, vocab_size, 0.5, 0.8);
    let l1 = GajeNeuromorphicLayer::new(vocab_size, hidden_dim, 0.5, 0.8);
    let mut layers = [l0, l1];

    // 3. Bucle de Entrenamiento Supervisado Nativo
    let epochs = 500; // Más épocas para estabilidad en 2-bits
    let lr = 0.5;
    let start = Instant::now();
    
    println!("\n🔥 Entrenando resonancia secuencial (GenomicNorm activo)...");

    for epoch in 1..=epochs {
        let mut total_hits = 0;
        
        for i in 0..words.len() - 1 {
            let input_id = word_to_id[words[i]];
            let target_id = word_to_id[words[i+1]];
            
            // RESET potentials for clean step
            layers[0].membrane_potentials.fill(0.0);
            layers[1].membrane_potentials.fill(0.0);

            // 1. Reforzar asociación Input -> Hidden (Capa 0)
            // Asignamos neuronas ocultas específicas a cada input para evitar colisiones
            let mut l0_deltas = vec![-0.1; hidden_dim];
            let start_h = (input_id * 8) % hidden_dim;
            for j in 0..8 { l0_deltas[(start_h + j) % hidden_dim] = 1.0; }
            layers[0].refine_step(input_id, l0_deltas, 1.0);
            
            // 2. Forward pass para obtener spikes
            layers[0].integrate_batch(input_id, centroides, 1.0);
            let l0_spikes = layers[0].check_spikes();
            
            for &(idx, intensity, _) in &l0_spikes {
                layers[1].integrate_batch(idx, centroides, intensity);
            }
            let l1_spikes = layers[1].check_spikes();
            
            // 3. Reforzar asociación Hidden -> Target (Capa 1)
            let mut l1_deltas = vec![-0.5; vocab_size];
            l1_deltas[target_id] = 1.0;
            
            for &(h_idx, _, _) in &l0_spikes {
                layers[1].refine_step(h_idx, l1_deltas.clone(), lr);
            }
            
            if l1_spikes.iter().any(|&(idx, _, _)| idx == target_id) {
                total_hits += 1;
            }
        }
        
        if epoch % 100 == 0 || epoch == 1 {
            let accuracy = (total_hits as f32 / (words.len() - 1) as f32) * 100.0;
            println!("   [Epoch {:3}] Precisión: {:.2}%", epoch, accuracy);
        }
    }

    println!("\n✅ Evaluación finalizada en {:?}", start.elapsed());

    // 4. Prueba de Generación (Inferencia Autoregresiva)
    println!("\n🗨️ Generación del Organismo (Seed: '{}'):", words[0]);
    let mut current_id = word_to_id[words[0]];
    print!("{}", id_to_word[current_id]);
    
    for _ in 0..words.len() {
        // Simular dinámica exacta del entrenamiento
        layers[0].membrane_potentials.fill(0.0);
        layers[1].membrane_potentials.fill(0.0);
        
        layers[0].integrate_batch(current_id, centroides, 1.0);
        let l0_spikes = layers[0].check_spikes();
        
        for &(idx, intensity, _) in &l0_spikes {
            layers[1].integrate_batch(idx, centroides, intensity);
        }
        let l1_spikes = layers[1].check_spikes();
        
        if let Some(&(next_id, _, _)) = l1_spikes.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
            if next_id < vocab_size {
                print!(" {}", id_to_word[next_id]);
                current_id = next_id;
            }
        } else {
            // Intento de recuperación si no hay spikes directos (búsqueda de potencial máximo)
            if let Some((max_idx, _)) = layers[1].membrane_potentials.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) 
            {
                if max_idx < vocab_size && layers[1].membrane_potentials[max_idx] > 0.0 {
                    print!(" [{}]", id_to_word[max_idx]);
                    current_id = max_idx;
                } else {
                    print!(" ..."); break;
                }
            } else {
                print!(" ..."); break;
            }
        }
    }
    println!("\n");
}
