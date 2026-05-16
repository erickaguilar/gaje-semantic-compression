use _impl::loader::NativeLoader;
use _impl::nn::RustGenomicLLM;
use _impl::kernels;
use std::env;
use std::path::Path;
use std::time::Instant;
use std::io::{self, Write};
use rand::distributions::{Distribution, WeightedIndex};
use rayon::prelude::*;
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    unsafe {
        kernels::init_shuffle_table();
    }
    pyo3::prepare_freethreaded_python();
    
    let args: Vec<String> = env::args().collect();
    
    let mut model_path = String::new();
    let mut prompt_arg = None;
    let mut rollback_target = None;
    let mut i = 1;
    let mut evolve_target = None;
    let mut train_target = None;
    let mut generations = 2000;
    let mut train_epochs = 10;
    let mut scale = 0.02;
    let mut save_path = None;

    while i < args.len() {
        if args[i] == "--model" && i + 1 < args.len() {
            model_path = args[i+1].clone();
            i += 2;
        } else if args[i] == "--prompt" && i + 1 < args.len() {
            prompt_arg = Some(args[i+1].clone());
            i += 2;
        } else if args[i] == "--evolve" && i + 1 < args.len() {
            evolve_target = Some(args[i+1].clone());
            i += 2;
        } else if args[i] == "--train" && i + 1 < args.len() {
            train_target = Some(args[i+1].clone());
            i += 2;
        } else if args[i] == "--epochs" && i + 1 < args.len() {
            train_epochs = args[i+1].parse().unwrap_or(10);
            i += 2;
        } else if args[i] == "--gens" && i + 1 < args.len() {
            generations = args[i+1].parse().unwrap_or(2000);
            i += 2;
        } else if args[i] == "--scale" && i + 1 < args.len() {
            scale = args[i+1].parse().unwrap_or(0.02);
            i += 2;
        } else if args[i] == "--save" && i + 1 < args.len() {
            save_path = Some(args[i+1].clone());
            i += 2;
        } else if args[i] == "--rollback" && i + 1 < args.len() {
            rollback_target = Some(args[i+1].parse::<u64>().map_err(|e| e.to_string())?);
            i += 2;
        } else if model_path.is_empty() {
            if !args[i].starts_with("--") {
                model_path = args[i].clone();
            }
            i += 1;
        } else {
            i += 1;
        }
    }

    if model_path.is_empty() {
        println!("Usage: gaje-cli <model_path> [--prompt \"...\"] [--evolve \"target\"] [--train \"dataset.txt\"] [--save output.gaje]");
        return Ok(());
    }

    println!("🧬 GAJE Native Runtime (v0.6.5)");
    
    let mut gaje_loader_opt = None;
    
    let (mut model, tokenizer, config) = if model_path.ends_with(".gguf") {
        let mut loader = _impl::loader::GGUFLoader::new(&model_path)?;
        let config = loader.infer_config()?;
        println!("[+] GGUF Config Inferred: {} layers, {} dim", config.n_blocks, config.n_embd);
        
        let model = loader.load_genomic_llm(config.clone(), -1.0)?;
        let tokenizer_path = Path::new(&model_path).parent().unwrap().join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
             tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?
        } else {
             return Err(format!("tokenizer.json not found in {}", Path::new(&model_path).parent().unwrap().display()).into());
        };
        (model, tokenizer, config)
    } else {
        let loader = NativeLoader::new(&model_path)?;
        println!("[*] Extracting tokenizer from GAJE DB...");
        let tokenizer = loader.load_tokenizer().map_err(|e| e.to_string())?;
        let config = loader.load_config()?;
        let model = loader.load_llm()?;
        let l = NativeLoader::new(&model_path)?;
        gaje_loader_opt = Some(l);
        (model, tokenizer, config)
    };

    println!("[*] Model & Tokenizer loaded.");

    if let Some(ref target_text) = evolve_target {
        println!("[*] Iniciando Crianza por Integración de Caminos (Poblacional) para: '{}'", target_text);
        let _start_evolve = Instant::now();

        let encoding = tokenizer.encode(target_text.clone(), false).map_err(|e| e.to_string())?;
        let tokens = encoding.get_ids();

        if tokens.len() < 2 {
            return Err("Target text too short for evolution".into());
        }

        let evaluate = |m: &mut RustGenomicLLM, tokens: &[u32]| -> f32 {
            m.clear_cache().unwrap();
            let mut total_log_prob = 0.0f32;
            for i in 0..tokens.len() - 1 {
                let current_token = tokens[i] as usize;
                let target_token = tokens[i+1] as usize;
                let logits = m.forward(current_token, false).unwrap();
                let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum_exp = 0.0f32;
                for &l in &logits {
                    if !l.is_nan() { sum_exp += (l - max_l).exp(); }
                }
                let prob = (logits[target_token] - max_l).exp() / (sum_exp + 1e-12);
                total_log_prob += (prob + 1e-12).ln();
            }
            total_log_prob
        };

        let mut best_fitness = evaluate(&mut model, tokens);
        println!("[Gen 0] Log-Fitness Inicial: {:.4}", best_fitness);

        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        let mut layers = vec!["lm_head".to_string()];
        if model.blocks.len() > 0 {
            let last = model.blocks.len() - 1;
            layers.push(format!("blk.{}.attn_v", last));
            layers.push(format!("blk.{}.attn_output", last));
            layers.push(format!("blk.{}.ffn_up", last));
        }

        let population_size = 8;

        for gen in 1..=generations {
            let layer_name = layers.choose(&mut rng).unwrap();
            let current_scale = (scale * (1.0 - (gen as f32 / generations as f32))).max(1e-5);
            
            // Generar clones mutantes (costo cero en RAM gracias a Arc<Vec<u8>>)
            let mut population = Vec::new();
            for _ in 0..population_size {
                population.push(model.clone());
            }

            // Evaluar los caminos evolutivos en todos los núcleos de CPU (AVX2/Rayon)
            let results: Vec<Option<(Vec<f32>, f32)>> = population.into_par_iter().map(|mut p_model| {
                if let Ok(delta) = p_model.mutate_layer(layer_name, current_scale) {
                    let fitness = evaluate(&mut p_model, tokens);
                    if fitness > best_fitness {
                        return Some((delta, fitness));
                    }
                }
                None
            }).collect();

            let mut paths: Vec<(Vec<f32>, f32)> = results.into_iter().flatten().collect();
            
            if !paths.is_empty() {
                paths.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let n_integrate = paths.len().min(3);
                let weight = 1.0 / (n_integrate as f32);
                for k in 0..n_integrate {
                    model.apply_weighted_layer_mutation(layer_name, paths[k].0.clone(), weight).unwrap();
                }
                best_fitness = evaluate(&mut model, tokens);
                if gen % 10 == 0 || best_fitness > -10.0 {
                    println!("[Gen {}] Camino Integrado en {}: Fitness = {:.4} ({} caminos)", gen, layer_name, best_fitness, n_integrate);
                }
            }

            if best_fitness > -0.05 {
                println!("🔥 ¡Propagador de Inteligencia Alcanzado!");
                break;
            }
        }
        println!("[+] Crianza completada. Log-Fitness Final: {:.4}", best_fitness);
    }

    if let Some(dataset_path) = train_target {
        println!("[*] Iniciando Auto-Grad Nativo (Hybrid-Training) con texto: {}", dataset_path);
        let start_train = Instant::now();
        
        // If the target is an existing file, read it. Otherwise treat it as a raw string.
        let text = std::fs::read_to_string(&dataset_path).unwrap_or_else(|_| dataset_path.clone());
        let encoding = tokenizer.encode(text, false).map_err(|e| e.to_string())?;
        let tokens = encoding.get_ids();

        if tokens.len() < 2 {
            return Err("Dataset too short for training".into());
        }

        let mut current_lr = scale;
        for epoch in 1..=train_epochs {
            model.clear_cache().unwrap();
            let mut total_loss = 0.0;
            
            for i in 0..tokens.len() - 1 {
                let token_id = tokens[i] as usize;
                let target_id = tokens[i+1] as usize;
                
                let loss = model.train_step(token_id, target_id, current_lr)?;
                total_loss += loss;
            }
            
            let avg_loss = total_loss / (tokens.len() - 1) as f32;
            println!("[Epoch {}] Avg Loss: {:.4} (LR: {:.4})", epoch, avg_loss, current_lr);
            
            // Simple decay
            current_lr *= 0.9;
        }
        
        println!("[+] Entrenamiento completado en {:?}", start_train.elapsed());
    }

    if let Some(ref path) = save_path {
        println!("[*] Guardando organismo genómico en: {}", path);
        let start_save = Instant::now();
        _impl::loader::save_genomic_model(&path, &model, &config, Some(&tokenizer))?;
        println!("[+] Modelo guardado exitosamente en {:?}", start_save.elapsed());
    }

    if let Some(target) = rollback_target {
        if let Some(loader) = gaje_loader_opt {
            println!("[*] Initiating rollback to timestamp: {}", target);
            let mutations = loader.list_mutations().map_err(|e| e.to_string())?;
            let mut count = 0;
            for (ts, data) in mutations.into_iter().rev() {
                if ts > target {
                    let mutation: _impl::db::Mutation = bincode::deserialize(&data).map_err(|e| e.to_string())?;
                    model.apply_mutation(&mutation.layer_name, mutation.delta_centroids, true).map_err(|e| e.to_string())?;
                    count += 1;
                }
            }
            println!("[+] Rollback complete. Undone {} mutations.", count);
        }
    }

    if let Some(prompt) = prompt_arg {
        generate(&mut model, &tokenizer, &prompt, 50)?;
    } else if evolve_target.is_none() {
        loop {
            print!("\n👤 User: ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let prompt = input.trim();
            if prompt.is_empty() { continue; }
            if prompt == "exit" || prompt == "quit" { break; }
            print!("🤖 GAJE: ");
            io::stdout().flush()?;
            generate(&mut model, &tokenizer, prompt, 100)?;
            println!();
        }
    }

    Ok(())
}

fn sample_logits(logits: &[f32], temperature: f32, top_k: usize, top_p: f32) -> usize {
    if temperature == 0.0 {
        return logits.iter().enumerate().filter(|(_, &a)| !a.is_nan())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
    }

    let mut indexed_logits: Vec<(usize, f32)> = logits.iter().enumerate()
        .filter(|(_, &l)| !l.is_nan())
        .map(|(i, &l)| (i, l / temperature))
        .collect();

    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if top_k > 0 && top_k < indexed_logits.len() {
        indexed_logits.truncate(top_k);
    }

    let max_logit = indexed_logits.first().map(|&(_, l)| l).unwrap_or(0.0);
    let mut probs: Vec<(usize, f32)> = indexed_logits.iter()
        .map(|&(i, l)| (i, (l - max_logit).exp()))
        .collect();

    let sum_probs: f32 = probs.iter().map(|&(_, p)| p).sum();
    for item in &mut probs {
        item.1 /= sum_probs;
    }

    if top_p < 1.0 {
        let mut cumulative = 0.0;
        let mut cutoff = probs.len();
        for (idx, &(_, p)) in probs.iter().enumerate() {
            cumulative += p;
            if cumulative > top_p {
                cutoff = idx + 1;
                break;
            }
        }
        probs.truncate(cutoff);
    }

    let weights: Vec<f32> = probs.iter().map(|&(_, p)| p).collect();
    if let Ok(dist) = WeightedIndex::new(&weights) {
        let mut rng = rand::thread_rng();
        probs[dist.sample(&mut rng)].0
    } else {
        probs[0].0
    }
}

fn generate(model: &mut RustGenomicLLM, tokenizer: &tokenizers::Tokenizer, prompt: &str, max_tokens: usize) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let encoding = tokenizer.encode(prompt, false).map_err(|e| e.to_string())?;
    let tokens = encoding.get_ids();
    model.clear_cache().unwrap();
    let mut current_tokens = tokens.to_vec();
    if current_tokens.is_empty() { return Ok(()); }
    let mut logits = Vec::new();
    for &tid in &current_tokens {
        logits = model.forward(tid as usize, false).unwrap();
    }
    
    let temperature = 0.7;
    let top_k = 40;
    let top_p = 0.9;
    
    for _ in 0..max_tokens {
        let next_token = sample_logits(&logits, temperature, top_k, top_p);
        if next_token == 0 { break; }
        let decoded = tokenizer.decode(&[next_token as u32], true).map_err(|e| e.to_string())?;
        print!("{}", decoded);
        io::stdout().flush()?;
        current_tokens.push(next_token as u32);
        logits = model.forward(next_token, false).unwrap();
    }
    Ok(())
}
