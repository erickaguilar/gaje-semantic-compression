use _impl::io::loader::NativeLoader;
use _impl::nn::llm::GenomicLLM;
use _impl::compute::kernels;
use _impl::core::tokenizer::GajeTokenizer;
use std::env;
use std::path::Path;
use std::io::{self, Write};
use rand::distributions::{Distribution, WeightedIndex};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    unsafe { kernels::init_shuffle_table(); }
    let args: Vec<String> = env::args().collect();
    let mut model_path = String::new();
    let mut prompt_arg = None;
    let mut i = 1;
    let mut evolve_target = None;
    let mut train_target = None;
    let mut generations = 2000;
    let mut train_epochs = 10;
    let mut scale = 0.02;
    let mut resonance_weight = 0.05;
    let mut save_path = None;
    let mut init_path = None;
    let mut import_path = None;
    let mut output_path = None;
    let mut init_preset = "default".to_string();
    let mut tokenize_text = None;
    let mut inspect_model = false;

    while i < args.len() {
        if args[i] == "--model" && i + 1 < args.len() { model_path = args[i+1].clone(); i += 2; }
        else if args[i] == "--import" && i + 1 < args.len() { import_path = Some(args[i+1].clone()); i += 2; }
        else if args[i] == "--output" && i + 1 < args.len() { output_path = Some(args[i+1].clone()); i += 2; }
        else if args[i] == "--preset" && i + 1 < args.len() { init_preset = args[i+1].clone(); i += 2; }
        else if args[i] == "--inspect" { inspect_model = true; i += 1; }
        else if args[i] == "--init" && i + 1 < args.len() { init_path = Some(args[i+1].clone()); i += 2; }
        else if args[i] == "--tokenize" && i + 1 < args.len() { tokenize_text = Some(args[i+1].clone()); i += 2; }
        else if args[i] == "--prompt" && i + 1 < args.len() { prompt_arg = Some(args[i+1].clone()); i += 2; }
        else if args[i] == "--evolve" && i + 1 < args.len() { evolve_target = Some(args[i+1].clone()); i += 2; }
        else if args[i] == "--train" && i + 1 < args.len() { train_target = Some(args[i+1].clone()); i += 2; }
        else if args[i] == "--epochs" && i + 1 < args.len() { train_epochs = args[i+1].parse().unwrap_or(10); i += 2; }
        else if args[i] == "--gens" && i + 1 < args.len() { generations = args[i+1].parse().unwrap_or(2000); i += 2; }
        else if args[i] == "--scale" && i + 1 < args.len() { scale = args[i+1].parse().unwrap_or(0.02); i += 2; }
        else if args[i] == "--resonance" && i + 1 < args.len() { resonance_weight = args[i+1].parse().unwrap_or(0.05); i += 2; }
        else if args[i] == "--save" && i + 1 < args.len() { save_path = Some(args[i+1].clone()); i += 2; }
        else if model_path.is_empty() { if !args[i].starts_with("--") { model_path = args[i].clone(); } i += 1; }
        else { i += 1; }
    }

    if let Some(path) = init_path {
        println!("[*] Creando nuevo organismo genómico 100% nativo en: {} (Preset: {})", path, init_preset);
        let (n_embd, n_blocks, n_head, vocab_size) = match init_preset.as_str() {
            "gold_embryo" => (384, 8, 6, 49152), 
            "micro_organism" => (128, 2, 4, 1024),
            "silver_fetus" => (512, 12, 8, 32768), // Fase 5.0: 10MB High-Fidelity
            _ => (768, 6, 12, 49152),
        };
        let config = _impl::io::loader::ModelConfig {
            config: _impl::io::loader::ArchConfig {
                name: format!("GAJE-{}-Organism", init_preset), version: "0.9.5".to_string(), tokenizer_id: "tokenizer".to_string(),
                rope_base: 1000000.0, ffn_act: "swiglu".to_string(), use_genomic_norm: true, rope_style: "split".to_string(),
                anchor_threshold: 0.1, ffn_anchor_threshold: 0.1, unpermute_weights: false, apply_smollm_rope_patch: false,
            },
            n_embd, n_head, n_head_kv: n_head, n_blocks, vocab_size: Some(vocab_size), eps: 1e-6,
        };
        let model = _impl::io::loader::init_born_genomic_model(&path, config.clone(), vocab_size)?;
        if Path::new("models/core/tokenizer.json").exists() {
            let tok = GajeTokenizer::from_file("models/core/tokenizer.json").map_err(|e| e.to_string())?;
            _impl::io::loader::save_genomic_model(&path, &model, &config, Some(&tok))?;
            println!("[+] Tokenizador 'models/core/tokenizer.json' integrado en el organismo.");
        }
        println!("[+] Nuevo organismo inicializado exitosamente.");
        return Ok(());
    }

    if let Some(path) = import_path {
        let out = output_path.ok_or("Debe especificar --output <path.gaje> al importar")?;
        println!("[*] Importando modelo GGUF a formato GAJE nativo...");
        let loader = _impl::io::loader::GGUFLoader::new(&path)?;
        let config = loader.infer_config()?;
        let model = loader.load_genomic_llm(config.clone(), -1.0)?;
        let mut tokenizer = None;
        let tokenizer_path = Path::new(&path).parent().unwrap().join("tokenizer.json");
        if tokenizer_path.exists() {
            tokenizer = Some(GajeTokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?);
            println!("[+] Tokenizador detectado e integrado.");
        }
        _impl::io::loader::save_genomic_model(&out, &model, &config, tokenizer.as_ref())?;
        println!("[+] Importación completada exitosamente.");
        return Ok(());
    }

    if model_path.is_empty() {
        println!("Usage: gaje-cli <model_path> [--prompt \"...\"] [--evolve \"target\"] [--train \"dataset.txt\"] [--save output.gaje] [--init new.gaje] [--inspect] [--import path.gguf --output path.gaje]");
        return Ok(());
    }

    if inspect_model {
        let loader = NativeLoader::new(&model_path)?;
        let config = loader.load_config()?;
        println!("--- Metadata for {} ---", model_path);
        println!("{}", serde_json::to_string_pretty(&config).unwrap());
        return Ok(());
    }

    println!("🧬 GAJE Native Runtime (v0.7.0)");
    
    let (mut model, tokenizer, config) = if model_path.ends_with(".gguf") {
        let loader = _impl::io::loader::GGUFLoader::new(&model_path)?;
        let config = loader.infer_config()?;
        let model = loader.load_genomic_llm(config.clone(), -1.0)?;
        let tokenizer_path = Path::new(&model_path).parent().unwrap().join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() { GajeTokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())? }
        else { return Err("tokenizer.json not found".to_string().into()); };
        (model, tokenizer, config)
    } else {
        let loader = NativeLoader::new(&model_path)?;
        let tokenizer = loader.load_tokenizer().map_err(|e| e.to_string())?;
        let config = loader.load_config()?;
        let model = loader.load_llm()?;
        (model, tokenizer, config)
    };

    println!("[*] Model & Tokenizer loaded.");

    if let Some(text) = tokenize_text {
        println!("[*] Tokenizando texto de forma nativa: \"{}\"", text);
        let ids = tokenizer.encode(&text, true).map_err(|e| e.to_string())?;
        println!("    IDs de Tokens: {:?}", ids);
        for &id in &ids { let piece = tokenizer.decode(&[id], true).map_err(|e| e.to_string())?; println!("      [{:>6}] -> \"{}\"", id, piece); }
        return Ok(());
    }

    if let Some(ref target_text) = evolve_target {
        println!("[*] Iniciando Crianza por Integración de Caminos (Poblacional) para: '{}'", target_text);
        let tokens = tokenizer.encode(target_text, false).map_err(|e| e.to_string())?;
        if tokens.len() < 2 { return Err("Target text too short for evolution".into()); }
        let evaluate = |m: &mut GenomicLLM, tokens: &[u32]| -> f32 {
            m.clear_cache_core(); let mut total_log_prob = 0.0f32;
            for i in 0..tokens.len() - 1 {
                let logits = m.forward_core(tokens[i] as usize, false).unwrap();
                let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum_exp = 0.0f32; for &l in &logits { sum_exp += (l - max_l).exp(); }
                let prob = (logits[tokens[i+1] as usize] - max_l).exp() / (sum_exp + 1e-12);
                total_log_prob += (prob + 1e-12).ln();
            }
            total_log_prob
        };

        let mut best_fitness = evaluate(&mut model, &tokens);
        println!("[Gen 0] Log-Fitness Inicial: {:.4}", best_fitness);
        let mut layers = vec!["lm_head".to_string()];
        if !model.blocks.is_empty() { let last = model.blocks.len() - 1; layers.push(format!("blk.{}.attn_output", last)); layers.push(format!("blk.{}.ffn_down", last)); }
        
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        for gen in 1..=generations {
            let layer_name = layers.choose(&mut rng).unwrap();
            let current_scale = (scale * (1.0 - (gen as f32 / generations as f32))).max(1e-5);
            let mut candidate_model = model.clone();
            if candidate_model.mutate_layer_core(layer_name, current_scale).is_ok() {
                let fitness = evaluate(&mut candidate_model, &tokens);
                if fitness > best_fitness {
                    model = candidate_model; best_fitness = fitness;
                    if gen % 10 == 0 || best_fitness > -10.0 { println!("[Gen {}] Mejora en {}: Fitness = {:.4}", gen, layer_name, best_fitness); }
                }
            }
            if best_fitness > -0.05 { println!("🔥 ¡Propagador de Inteligencia Alcanzado!"); break; }
        }
    }

    if let Some(ref dataset_path) = train_target {
        println!("[*] Iniciando Entrenamiento Born-Genomic Nativo (Resonancia: {:.3})", resonance_weight);
        let text = std::fs::read_to_string(dataset_path).unwrap_or_else(|_| dataset_path.clone());
        let dataset: Vec<Vec<usize>> = text.lines().map(|l| l.trim()).filter(|l| l.len() > 5).map(|l| {
            tokenizer.encode(l, false).unwrap_or_default().into_iter().map(|id| id as usize).collect()
        }).filter(|tokens: &Vec<usize>| tokens.len() >= 2).collect();
        if dataset.is_empty() { return Err("Dataset empty or too short".into()); }
        let trainer = _impl::nn::trainer::GenomicTrainerCore::new(scale, resonance_weight);
        trainer.fit(&mut model, &dataset, train_epochs).map_err(|e| e.to_string())?;
        println!("[+] Entrenamiento completado.");
    }

    if let Some(ref path) = save_path { _impl::io::loader::save_genomic_model(path, &model, &config, Some(&tokenizer))?; println!("[+] Modelo guardado exitosamente."); }

    if let Some(prompt) = prompt_arg { generate(&mut model, &tokenizer, &prompt, 50)?; } 
    else if evolve_target.is_none() && train_target.is_none() { println!("\n[!] Modo interactivo no disponible en TTY reducido. Use --prompt."); }

    Ok(())
}

fn sample_logits(logits: &[f32], temperature: f32, top_k: usize, top_p: f32) -> usize {
    if temperature == 0.0 { return logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap_or(0); }
    let mut indexed_logits: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &l)| (i, l / temperature)).collect();
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    if top_k > 0 && top_k < indexed_logits.len() { indexed_logits.truncate(top_k); }
    let max_logit = indexed_logits[0].1;
    let mut probs: Vec<(usize, f32)> = indexed_logits.iter().map(|&(i, l)| (i, (l - max_logit).exp())).collect();
    let sum: f32 = probs.iter().map(|&(_, p)| p).sum();
    for item in &mut probs { item.1 /= sum; }
    if top_p < 1.0 { let mut cum = 0.0; let mut cutoff = probs.len(); for (idx, &(_, p)) in probs.iter().enumerate() { cum += p; if cum > top_p { cutoff = idx + 1; break; } } probs.truncate(cutoff); }
    let weights: Vec<f32> = probs.iter().map(|&(_, p)| p).collect();
    if let Ok(dist) = WeightedIndex::new(&weights) { probs[dist.sample(&mut rand::thread_rng())].0 } else { probs[0].0 }
}

fn generate(model: &mut GenomicLLM, tokenizer: &GajeTokenizer, prompt: &str, max_tokens: usize) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tokens = tokenizer.encode(prompt, false).map_err(|e| e.to_string())?;
    model.clear_cache_core();
    let mut logits = Vec::new();
    for &tid in &tokens { logits = model.forward_phase_gaje_core(tid as usize, 64).map_err(|e| e.to_string())?; }
    for _ in 0..max_tokens {
        let next_token = sample_logits(&logits, 0.4, 40, 0.9);
        if next_token == 0 || next_token == 151643 { break; } 
        let decoded = tokenizer.decode(&[next_token as u32], true).map_err(|e| e.to_string())?;
        print!("{}", decoded); io::stdout().flush()?;
        logits = model.forward_phase_gaje_core(next_token, 64).map_err(|e| e.to_string())?;
    }
    Ok(())
}
