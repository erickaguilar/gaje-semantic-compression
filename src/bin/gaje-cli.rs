use _impl::loader::NativeLoader;
use _impl::nn::RustGenomicLLM;
use _impl::kernels;
use std::env;
use std::path::Path;
use std::time::Instant;
use std::io::{self, Write};

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
    let mut generations = 2000;
    let mut scale = 0.02;

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
        } else if args[i] == "--gens" && i + 1 < args.len() {
            generations = args[i+1].parse().unwrap_or(2000);
            i += 2;
        } else if args[i] == "--scale" && i + 1 < args.len() {
            scale = args[i+1].parse().unwrap_or(0.02);
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
        println!("Usage: gaje-cli <model_path> [--prompt \"...\"] [--evolve \"target\"] [--gens 2000] [--scale 0.02]");
        return Ok(());
    }

    println!("🧬 GAJE Native Runtime (v0.6.5)");
    
    let mut gaje_loader_opt = None;
    
    let (mut model, tokenizer) = if model_path.ends_with(".gguf") {
        let mut loader = _impl::loader::GGUFLoader::new(&model_path)?;
        let config = loader.infer_config()?;
        println!("[+] GGUF Config Inferred: {} layers, {} dim", config.n_blocks, config.n_embd);
        
        let model = loader.load_genomic_llm(config, -1.0)?;
        let tokenizer_path = Path::new(&model_path).parent().unwrap().join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
             tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?
        } else {
             return Err(format!("tokenizer.json not found in {}", Path::new(&model_path).parent().unwrap().display()).into());
        };
        (model, tokenizer)
    } else {
        let loader = NativeLoader::new(&model_path)?;
        println!("[*] Extracting tokenizer from GAJE DB...");
        let tokenizer = loader.load_tokenizer().map_err(|e| e.to_string())?;
        let model = loader.load_llm()?;
        let l = NativeLoader::new(&model_path)?;
        gaje_loader_opt = Some(l);
        (model, tokenizer)
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

        let population_size = 8; // Número de 'caminos' simultáneos

        for gen in 1..=generations {
            let layer_name = layers.choose(&mut rng).unwrap();
            let current_scale = (scale * (1.0 - (gen as f32 / generations as f32))).max(1e-5);
            
            let mut paths = Vec::new();
            
            // 1. Muestreo de Caminos (Monte Carlo masivo)
            for _ in 0..population_size {
                let delta = model.mutate_layer(layer_name, current_scale).unwrap();
                let fitness = evaluate(&mut model, tokens);
                
                if fitness > best_fitness {
                    paths.push((delta.clone(), fitness));
                }
                
                // Limpiar mutación para la siguiente muestra de la población
                model.undo_layer_mutation(layer_name, delta).unwrap();
            }
            
            // 2. Integración de Caminos (Merge de historias exitosas)
            if !paths.is_empty() {
                // Ordenar por fitness descendente
                paths.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                
                // Solo integramos los 3 mejores caminos (Elite integration)
                let n_integrate = paths.len().min(3);
                let weight = 1.0 / (n_integrate as f32);
                
                for k in 0..n_integrate {
                    model.apply_weighted_layer_mutation(layer_name, paths[k].0.clone(), weight).unwrap();
                }
                
                // Re-evaluar el modelo combinado
                best_fitness = evaluate(&mut model, tokens);
                
                if gen % 10 == 0 || best_fitness > -10.0 {
                    println!("[Gen {}] Camino Integrado en {}: Fitness = {:.4} ({} caminos)", gen, layer_name, best_fitness, n_integrate);
                }
            }

            if best_fitness > -0.05 {
                println!("🔥 ¡Propagador de Inteligencia Alcanzado (Path Integral Coherence)!");
                break;
            }
        }
        println!("[+] Crianza completada. Log-Fitness Final: {:.4}", best_fitness);
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
    for _ in 0..max_tokens {
        let next_token = logits.iter().enumerate().filter(|(_, &a)| !a.is_nan())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);
        let next_token = match next_token { Some(t) => t, None => break };
        if next_token == 0 { break; }
        let decoded = tokenizer.decode(&[next_token as u32], true).map_err(|e| e.to_string())?;
        print!("{}", decoded);
        io::stdout().flush()?;
        current_tokens.push(next_token as u32);
        logits = model.forward(next_token, false).unwrap();
    }
    Ok(())
}
