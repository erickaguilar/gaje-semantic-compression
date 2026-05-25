use _impl::nn::spiking::layer::GajeNeuromorphicLayer;
use _impl::io::smg1::{save_smg1_model, load_smg1_model, Smg1Model, Smg1Config};
use std::collections::HashMap;
use std::time::Instant;
use std::path::Path;

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
enum CurriculumPhase {
    A, // Identidad y Ontología
    B, // Lógica Relacional
    C, // Conocimiento Técnico (Dataset Completo)
}

impl CurriculumPhase {
    fn get_dataset_path(&self) -> String {
        match self {
            CurriculumPhase::A => "data/training/curriculum/phase_a_identity.txt".to_string(),
            CurriculumPhase::B => "data/training/curriculum/phase_b_logic.txt".to_string(),
            CurriculumPhase::C => "data/datasets/dataset_es_ext.txt".to_string(),
        }
    }
    fn next(&self) -> Option<Self> {
        match self {
            CurriculumPhase::A => Some(CurriculumPhase::B),
            CurriculumPhase::B => Some(CurriculumPhase::C),
            CurriculumPhase::C => None,
        }
    }
    fn name(&self) -> &str {
        match self {
            CurriculumPhase::A => "Fase A: Identidad y Ontología",
            CurriculumPhase::B => "Fase B: Lógica Relacional",
            CurriculumPhase::C => "Fase C: Conocimiento Técnico",
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.iter().position(|x| x == "--model")
        .and_then(|idx| args.get(idx + 1).map(|s| s.as_str()))
        .unwrap_or("GoldEmbryo-v1.gaje");

    println!("🧬 GAJE-Flow: Inicializando Micro-Genoma Estándar (SMG-1)");
    
    let mut word_to_id = HashMap::new();
    let mut id_to_word = Vec::new();
    let mut layers = Vec::new();
    let mut current_phase = CurriculumPhase::A;
    let mut vocab_size = 0;
    
    // 1. Cargar o Inicializar Modelo
    let mut model_exists = false;
    if Path::new(model_path).exists() {
        println!("   📂 Cargando modelo existente desde: {}", model_path);
        match load_smg1_model(model_path) {
            Ok((model, _config)) => {
                word_to_id = model.word_to_id;
                id_to_word = model.id_to_word;
                layers = model.layers;
                vocab_size = id_to_word.len();
                model_exists = true;
                // Por ahora, asumimos que si se carga, el usuario podría querer forzar una fase o continuar.
                // Implementación simple: si tiene vocabulario básico, empezamos en A o B.
                if vocab_size > 50 { current_phase = CurriculumPhase::C; }
                else if vocab_size > 20 { current_phase = CurriculumPhase::B; }
            }
            Err(e) => println!("   ⚠️ Error cargando modelo: {}. Iniciando desde cero.", e),
        }
    }

    if !model_exists {
        println!("   ✨ Inicializando nuevo organismo genómico...");
        // Inicialización mínima de vocabulario para Phase A
        let initial_text = std::fs::read_to_string(current_phase.get_dataset_path()).unwrap_or_default();
        for word in initial_text.split_whitespace() {
            word_to_id.entry(word.to_string()).or_insert_with(|| {
                let id = vocab_size;
                id_to_word.push(word.to_string());
                vocab_size += 1;
                id
            });
        }
        
        let dim_latent = 512;
        let dim_logic = 256;
        let l0 = GajeNeuromorphicLayer::new(dim_latent, vocab_size, 0.4, 0.8);
        let l1 = GajeNeuromorphicLayer::new(dim_logic, dim_latent, 0.4, 0.8);
        let l2 = GajeNeuromorphicLayer::new(vocab_size, dim_logic, 0.4, 0.8);
        layers = vec![l0, l1, l2];
    }
    
    println!("   Vocabulario Inicial: {} tokens únicos.", vocab_size);

    // 2. Bucle de Currículo
    let centroides = [-1.5, -0.5, 0.5, 1.5];
    let start_time = Instant::now();
    let mut best_accuracy = 0.0f32;

    loop {
        println!("\n🚀 Iniciando {}", current_phase.name());
        let dataset_text = match std::fs::read_to_string(current_phase.get_dataset_path()) {
            Ok(t) => t,
            Err(_) => {
                println!("   ⚠️ Dataset no encontrado para {}. Saltando...", current_phase.name());
                if let Some(next) = current_phase.next() { current_phase = next; continue; }
                else { break; }
            }
        };

        // Actualizar vocabulario si es necesario (solo expansión)
        for word in dataset_text.split_whitespace() {
            if !word_to_id.contains_key(word) {
                let id = vocab_size;
                word_to_id.insert(word.to_string(), id);
                id_to_word.push(word.to_string());
                vocab_size += 1;
            }
        }
        // Redimensionar L2 si el vocabulario creció
        if vocab_size > layers[2].num_neurons {
            println!("   📈 Expandiendo capa de salida a {} neuronas...", vocab_size);
            let mut new_l2 = GajeNeuromorphicLayer::new(vocab_size, layers[2].weights_per_neuron, 0.4, 0.8);
            
            // Copiar pesos existentes con cuidado de la estructura de empaquetado
            let old_row_size = (layers[2].num_neurons + 3) / 4;
            let new_row_size = (vocab_size + 3) / 4;
            
            for input_idx in 0..layers[2].weights_per_neuron {
                let old_start = input_idx * old_row_size;
                let new_start = input_idx * new_row_size;
                let bytes_to_copy = old_row_size.min(new_row_size);
                new_l2.packed_weights[new_start..new_start + bytes_to_copy]
                    .copy_from_slice(&layers[2].packed_weights[old_start..old_start + bytes_to_copy]);
            }
            layers[2] = new_l2;
        }

        let words: Vec<&str> = dataset_text.split_whitespace().collect();
        let mut consecutive_high_acc = 0;

        for epoch in 1..=1000 {
            let mut total_hits = 0;
            for i in 0..words.len() - 1 {
                let input_id = word_to_id[words[i]];
                let target_id = word_to_id[words[i+1]];
                
                for layer in &mut layers { layer.membrane_potentials.fill(0.0); }
                
                // Refuerzo y Forward (Lógica SMG-1 estándar)
                let l0_n = layers[0].num_neurons;
                let mut l0_deltas = vec![-0.5; l0_n];
                for j in 0..32 { l0_deltas[(input_id * 17 + j) % l0_n] = 1.5; }
                layers[0].refine_step(input_id, l0_deltas, 1.0);
                layers[0].integrate_batch(input_id, centroides, 1.0);
                let mut s0 = layers[0].check_spikes();
                s0.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let s0 = if s0.len() > 8 { s0[..8].to_vec() } else { s0 };

                let l1_n = layers[1].num_neurons;
                let mut l1_deltas = vec![-0.5; l1_n];
                for j in 0..16 { l1_deltas[(input_id * 13 + j) % l1_n] = 1.5; }
                for &(idx, _, _) in &s0 { layers[1].refine_step(idx, l1_deltas.clone(), 1.0); }
                for &(idx, intensity, _) in &s0 { layers[1].integrate_batch(idx, centroides, intensity); }
                let mut s1 = layers[1].check_spikes();
                s1.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let s1 = if s1.len() > 8 { s1[..8].to_vec() } else { s1 };

                let mut l2_deltas = vec![-1.0; vocab_size];
                l2_deltas[target_id] = 2.0;
                for &(idx, _, _) in &s1 { layers[2].refine_step(idx, l2_deltas.clone(), 1.0); }
                for &(idx, intensity, _) in &s1 { layers[2].integrate_batch(idx, centroides, intensity); }
                let mut s2 = layers[2].check_spikes();
                s2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let s2 = if s2.len() > 4 { s2[..4].to_vec() } else { s2 };

                if s2.iter().any(|&(idx, _, _)| idx == target_id) { total_hits += 1; }
                for layer in &mut layers { layer.apply_homeostasis(1.5); }
            }

            let accuracy = (total_hits as f32 / (words.len() - 1) as f32) * 100.0;
            if epoch % 10 == 0 { println!("   [Epoch {:3}] Precisión: {:.2}%", epoch, accuracy); }

            // Auto-Save
            if accuracy > best_accuracy {
                best_accuracy = accuracy;
                let config = Smg1Config { vocab_size, layer_dims: layers.iter().map(|l| (l.num_neurons, l.weights_per_neuron)).collect(), thresholds: layers[0].thresholds.clone(), decays: layers[0].decays.clone() };
                let model = Smg1Model { layers: layers.clone(), word_to_id: word_to_id.clone(), id_to_word: id_to_word.clone() };
                let _ = save_smg1_model(model_path, &model, &config);
            }

            // Lógica de Progresión: 90% de precisión constante durante 3 épocas
            if accuracy >= 90.0 { consecutive_high_acc += 1; }
            else { consecutive_high_acc = 0; }

            if consecutive_high_acc >= 3 {
                println!("   🎯 Fase completada con éxito ({:.2}% acc).", accuracy);
                break;
            }
        }

        if let Some(next) = current_phase.next() { current_phase = next; best_accuracy = 0.0; }
        else { break; }
    }

    println!("\n✅ Crianza por Currículo finalizada en {:?}", start_time.elapsed());
}
