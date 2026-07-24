use _impl::compute::kernels;
use _impl::core::tokenizer::GajeTokenizer;
use _impl::io::loader::NativeLoader;
use _impl::nn::llm::GenomicLLM;
use std::env;
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    unsafe {
        kernels::init_shuffle_table();
    }

    // Manejador de interrupción (Graceful Shutdown)
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("\n[!] Interrupción detectada (Ctrl+C). Finalizando de forma segura...");
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error configurando el manejador de señales");

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
    let mut save_path: Option<String> = None;
    let mut init_path: Option<String> = None;
    let mut import_path: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut init_preset = "default".to_string();
    let mut tokenize_text: Option<String> = None;
    let mut eval_corpus: Option<String> = None;
    let mut inspect_model = false;
    let mut dni_path: Option<String> = None;
    let mut dni_intensity = 0.01;
    let mut dni_pop = 16;
    let mut anchor_threshold_arg = 0.1;
    let mut target_layers_arg = Vec::new();
    let mut is_ingest_mode = false;
    let mut teacher_path = None;
    let mut teacher_tok_path = None;
    let mut do_iqat = false;
    let mut iqat_lr = 0.001;

    while i < args.len() {
        if args[i] == "ingest" {
            is_ingest_mode = true;
            i += 1;
        } else if args[i] == "--model" && i + 1 < args.len() {
            model_path = args[i + 1].clone();
            i += 2;
        } else if args[i] == "--eval" && i + 1 < args.len() {
            eval_corpus = Some(args[i + 1].clone());
            i += 2;
        } else if (args[i] == "--file" || args[i] == "--dni-ingest") && i + 1 < args.len() {
            dni_path = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--teacher" && i + 1 < args.len() {
            teacher_path = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--teacher-tok" && i + 1 < args.len() {
            teacher_tok_path = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--iqat" {
            do_iqat = true;
            i += 1;
        } else if args[i] == "--iqat-lr" && i + 1 < args.len() {
            iqat_lr = args[i + 1].parse().unwrap_or(0.001);
            i += 2;
        } else if (args[i] == "REMOVED") && i + 1 < args.len() {
            save_path = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--import" && i + 1 < args.len() {
            import_path = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--threshold" && i + 1 < args.len() {
            anchor_threshold_arg = args[i + 1].parse().unwrap_or(0.1);
            i += 2;
        } else if args[i] == "--preset" && i + 1 < args.len() {
            init_preset = args[i + 1].clone();
            i += 2;
        } else if args[i] == "--output" && i + 1 < args.len() {
            output_path = Some(args[i + 1].clone());
            save_path = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--save" && i + 1 < args.len() {
            save_path = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--inspect" {
            inspect_model = true;
            i += 1;
        } else if args[i] == "--init" && i + 1 < args.len() {
            init_path = Some(args[i + 1].clone());
            i += 2;
        } else if (args[i] == "--intensity" || args[i] == "--dni-intensity") && i + 1 < args.len() {
            dni_intensity = args[i + 1].parse().unwrap_or(0.01);
            i += 2;
        } else if (args[i] == "--pop" || args[i] == "--dni-pop") && i + 1 < args.len() {
            dni_pop = args[i + 1].parse().unwrap_or(16);
            i += 2;
        } else if args[i] == "--target-layers" && i + 1 < args.len() {
            target_layers_arg = args[i + 1].split(',').map(|s| s.to_string()).collect();
            i += 2;
        } else if args[i] == "--tokenize" && i + 1 < args.len() {
            tokenize_text = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--prompt" && i + 1 < args.len() {
            prompt_arg = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--evolve" && i + 1 < args.len() {
            evolve_target = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--train" && i + 1 < args.len() {
            train_target = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--epochs" && i + 1 < args.len() {
            train_epochs = args[i + 1].parse().unwrap_or(10);
            i += 2;
        } else if args[i] == "--gens" && i + 1 < args.len() {
            generations = args[i + 1].parse().unwrap_or(2000);
            i += 2;
        } else if args[i] == "--scale" && i + 1 < args.len() {
            scale = args[i + 1].parse().unwrap_or(0.02);
            i += 2;
        } else if args[i] == "--resonance" && i + 1 < args.len() {
            resonance_weight = args[i + 1].parse().unwrap_or(0.05);
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

    if let Some(path) = init_path {
        println!(
            "[*] Creando nuevo organismo genómico 100% nativo en: {} (Preset: {})",
            path, init_preset
        );
        let (n_embd, n_blocks, n_head, vocab_size) = match init_preset.as_str() {
            "gold_embryo" => (384, 8, 6, 49152),
            "micro_organism" => (128, 2, 4, 32768),
            "silver_fetus" => (512, 12, 8, 32768),
            "silver_adult" => (512, 12, 8, 32768), // Fase 5.5: 10MB Circular
            "silver_adult_32m" => (512, 8, 8, 32768), // 32MB Toroidal (67M parameters)
            "platinum" => (768, 24, 12, 32768),    // Fase 5.8: 20-25MB Platinum
            "titan" => (1024, 36, 16, 49152),      // Fase 6.0: 50MB Toroidal (Titan)
            _ => (768, 6, 12, 49152),
        };
        let mut config = _impl::io::loader::ModelConfig {
            config: _impl::io::loader::ArchConfig {
                name: format!("GAJE-{}-Organism", init_preset),
                version: "1.0.0-alpha".to_string(),
                tokenizer_id: "tokenizer".to_string(),
                rope_base: 1000000.0,
                ffn_act: "swiglu".to_string(),
                use_genomic_norm: true,
                rope_style: "split".to_string(),
                anchor_threshold: 0.1,
                ffn_anchor_threshold: 0.1,
                rna_threshold: 0.5,
                unpermute_weights: false,
                apply_smollm_rope_patch: false,
                tie_word_embeddings: false,
                dni: String::new(), // Se generará automáticamente
                state: "born".to_string(),
            },
            n_embd,
            n_head,
            n_head_kv: n_head,
            n_blocks,
            vocab_size: Some(vocab_size),
            eps: 1e-6,
        };
        // Forzar generación de DNI
        config.config.dni = _impl::io::loader::py_new_dni();
        let model = _impl::io::loader::init_born_genomic_model(&path, config.clone(), vocab_size)?;
        if Path::new("models/core/tokenizer.json").exists() {
            let tok = GajeTokenizer::from_file("models/core/tokenizer.json")
                .map_err(|e| e.to_string())?;
            _impl::io::loader::save_genomic_model(&path, &model, &config, Some(&tok))?;
            println!("[+] Tokenizador 'models/core/tokenizer.json' integrado en el organismo.");
        }
        println!("[+] Nuevo organismo inicializado exitosamente.");
        return Ok(());
    }

    if let Some(path) = import_path {
        let out = output_path.ok_or("Debe especificar --output <path.gaje> al importar")?;
        println!(
            "[*] Importando modelo GGUF a formato GAJE nativo (Threshold: {})...",
            anchor_threshold_arg
        );
        let loader = _impl::io::loader::GGUFLoader::new(&path)?;
        let config = loader.infer_config()?;
        let model = loader.load_genomic_llm(config.clone(), anchor_threshold_arg)?;
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

    if is_ingest_mode && (dni_path.is_none() || model_path.is_empty()) {
        println!("Usage: gaje-cli ingest --model <model_path> --file <doc.txt> [--output <output.gaje>] [--intensity 0.01] [--pop 16] [--gens 2000]");
        return Ok(());
    }

    if model_path.is_empty() {
        println!("Usage: gaje-cli <model_path> [--prompt \"...\"] [--evolve \"target\"] [--train \"dataset.txt\"] [--save output.gaje] [--init new.gaje] [--inspect] [--import path.gguf --output path.gaje]");
        println!(
            "Or: gaje-cli ingest --model <model_path> --file <doc.txt> [--output <output.gaje>]"
        );
        return Ok(());
    }

    if inspect_model {
        let loader = NativeLoader::new(&model_path)?;
        let config = loader.load_config()?;
        println!("--- Metadata for {} ---", model_path);
        println!("{}", serde_json::to_string_pretty(&config).unwrap());
        return Ok(());
    }

    println!("🧬 GAJE Native Runtime (v1.0.0-alpha)");

    let (mut model, tokenizer, config) = if model_path.ends_with(".gguf") {
        let loader = _impl::io::loader::GGUFLoader::new(&model_path)?;
        let config = loader.infer_config()?;
        let model = loader.load_genomic_llm(config.clone(), -1.0)?;
        let tokenizer_path = Path::new(&model_path)
            .parent()
            .unwrap()
            .join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            GajeTokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?
        } else {
            return Err("tokenizer.json not found".to_string().into());
        };
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
        for &id in &ids {
            let piece = tokenizer.decode(&[id], true).map_err(|e| e.to_string())?;
            println!("      [{:>6}] -> \"{}\"", id, piece);
        }
        return Ok(());
    }

    if let Some(ref target_text) = evolve_target {
        println!(
            "[*] Iniciando Crianza por Integración de Caminos (Poblacional) para: '{}'",
            target_text
        );
        let tokens = tokenizer
            .encode(target_text, false)
            .map_err(|e| e.to_string())?;
        if tokens.len() < 2 {
            return Err("Target text too short for evolution".into());
        }
        let evaluate = |m: &mut GenomicLLM, tokens: &[u32]| -> f32 {
            m.clear_cache_core();
            let mut total_log_prob = 0.0f32;
            for i in 0..tokens.len() - 1 {
                let logits = m.forward_core(tokens[i] as usize, false).unwrap();
                let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum_exp = 0.0f32;
                for &l in &logits {
                    sum_exp += (l - max_l).exp();
                }
                let prob = (logits[tokens[i + 1] as usize] - max_l).exp() / (sum_exp + 1e-12);
                total_log_prob += (prob + 1e-12).ln();
            }
            total_log_prob
        };

        let mut best_fitness = evaluate(&mut model, &tokens);
        println!("[Gen 0] Log-Fitness Inicial: {:.4}", best_fitness);
        let mut layers = vec!["lm_head".to_string()];
        if !model.blocks.is_empty() {
            let last = model.blocks.len() - 1;
            layers.push(format!("blk.{}.attn_output", last));
            layers.push(format!("blk.{}.ffn_down", last));
        }

        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        for gen in 1..=generations {
            if !running.load(Ordering::SeqCst) {
                println!("    [!] Evolución interrumpida por el usuario.");
                break;
            }

            let layer_name = layers.choose(&mut rng).unwrap();
            let current_scale = (scale * (1.0 - (gen as f32 / generations as f32))).max(1e-5);
            let mut candidate_model = model.clone();
            if candidate_model
                .mutate_layer_core(layer_name, current_scale)
                .is_ok()
            {
                let fitness = evaluate(&mut candidate_model, &tokens);
                if fitness > best_fitness {
                    model = candidate_model;
                    best_fitness = fitness;
                    if gen % 10 == 0 || best_fitness > -10.0 {
                        println!(
                            "[Gen {}] Mejora en {}: Fitness = {:.4}",
                            gen, layer_name, best_fitness
                        );
                    }
                }
            }
            if best_fitness > -0.05 {
                println!("🔥 ¡Propagador de Inteligencia Alcanzado!");
                break;
            }
        }
    }

    if let Some(path) = dni_path.clone() {
        println!(
            "[*] Iniciando Direct Neural Ingestion (DNI) Scalable: {}",
            path
        );
        let paths = if Path::new(&path).is_dir() {
            std::fs::read_dir(&path)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_file())
                .collect::<Vec<_>>()
        } else {
            vec![Path::new(&path).to_path_buf()]
        };

        use _impl::core::dni::{DNIEngine, SemanticNiche};
        let mut engine = DNIEngine {
            model: model.clone(),
            tokenizer: Arc::new(tokenizer.clone()),
            council: None,
            intensity: dni_intensity,
            target_layers: target_layers_arg,
            validation_tokens: Vec::new(),
            original_dna_hash: Vec::new(),
            niche: SemanticNiche::General,
        };
        engine.initialize_original_hash();
        engine.set_validation_text(
            "Hola, soy una inteligencia artificial basada en GAJE. ¿En qué puedo ayudarte?"
                .to_string(),
        );

        let start = std::time::Instant::now();
        for p in paths {
            println!("  [+] Inyectando: {:?}", p);
            if let Ok(content) = std::fs::read_to_string(&p) {
                match engine.ingest_document(content, generations, dni_pop) {
                    Ok(fitness) => println!("    - Completado. Fitness Final: {:.4}", fitness),
                    Err(e) => println!("    [!] Error en {:?}: {}", p, e),
                }
            }
        }

        model = engine.model;
        println!("[+] Proceso DNI finalizado en {:?}.", start.elapsed());
    }

    if do_iqat {
        let t_path = teacher_path.ok_or("Debe especificar --teacher <path.gguf> para IQAT")?;
        let t_tok =
            teacher_tok_path.ok_or("Debe especificar --teacher-tok <tokenizer.json> para IQAT")?;
        let teacher =
            _impl::nn::distiller::Teacher::new("Master".to_string(), &t_path, &t_tok, &tokenizer)
                .map_err(|e| e.to_string())?;

        println!("[*] Iniciando Refinamiento IQAT (Activation Drift) bloque a bloque...");
        let texts = if let Some(ref d_path) = dni_path {
            if Path::new(d_path).is_file() {
                vec![std::fs::read_to_string(d_path).unwrap_or_default()]
            } else {
                vec!["El lenguaje es la geodésica del pensamiento.".to_string()]
            }
        } else {
            vec!["El lenguaje es la geodésica del pensamiento.".to_string()]
        };

        let iqat = _impl::nn::iqat::IQATEngine::new(iqat_lr);
        iqat.run_iqat_cycle(&mut model, &teacher, &texts, train_epochs)
            .map_err(|e| e.to_string())?;
    }

    if let Some(ref dataset_path) = train_target {
        println!(
            "[*] Iniciando Entrenamiento Born-Genomic Nativo (Resonancia: {:.3})",
            resonance_weight
        );
        let text = std::fs::read_to_string(dataset_path).unwrap_or_else(|_| dataset_path.clone());
        let dataset: Vec<Vec<usize>> = text
            .lines()
            .map(|l| l.trim())
            .filter(|l| l.len() > 5)
            .map(|l| {
                tokenizer
                    .encode(l, false)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|id| id as usize)
                    .collect()
            })
            .filter(|tokens: &Vec<usize>| tokens.len() >= 2)
            .collect();
        if dataset.is_empty() {
            return Err("Dataset empty or too short".into());
        }

        let trainer = _impl::nn::trainer::GenomicTrainerCore::new(scale, resonance_weight);
        let p1_end = (train_epochs as f32 * 0.2) as usize;
        let p2_end = (train_epochs as f32 * 0.7) as usize;

        // Recuperar el paso guardado de los metadatos DNI
        let mut start_step: usize = config.config.dni.parse().unwrap_or(0);
        let mut current_config = config.clone();

        'epoch_loop: for epoch in 0..train_epochs {
            if !running.load(Ordering::SeqCst) {
                println!(
                    "    [!] Entrenamiento interrumpido por el usuario antes de la época {}.",
                    epoch + 1
                );
                break 'epoch_loop;
            }

            let phase = if epoch < p1_end {
                1
            } else if epoch < p2_end {
                2
            } else {
                3
            };

            // Entrenar época con callback de guardado intra-época (cada 100 muestras)
            let s_path = save_path.clone();
            let tok = tokenizer.clone();
            let run_flag = running.clone();
            let mut epoch_config = current_config.clone();

            let res = trainer.fit_epoch(
                &mut model,
                &dataset,
                epoch,
                train_epochs,
                phase,
                start_step,
                |m, count, loss| {
                    if !run_flag.load(Ordering::SeqCst) {
                        return Err("INTERRUPTED".to_string());
                    }

                    if let Some(ref path) = s_path {
                        // Actualizar el marcador de paso en la config antes de guardar
                        epoch_config.config.dni = count.to_string();
                        _impl::io::loader::save_genomic_model(path, m, &epoch_config, Some(&tok))
                            .map_err(|e| e.to_string())?;
                        println!(
                            "      [Intra-Epoch Checkpoint] Muestra #{} | Loss: {:.4}",
                            count, loss
                        );
                    }
                    Ok(())
                },
            );

            // Después de una época completa, reiniciamos el marcador para la siguiente
            start_step = 0;
            current_config.config.dni = "0".to_string();

            if let Err(e) = res {
                if e == "INTERRUPTED" {
                    println!("    [!] Abortando época {} por interrupción.", epoch + 1);
                    break 'epoch_loop;
                } else {
                    return Err(e.into());
                }
            }

            // Checkpoint final de época
            if let Some(ref path) = save_path {
                _impl::io::loader::save_genomic_model(
                    path,
                    &model,
                    &current_config,
                    Some(&tokenizer),
                )?;
                println!(
                    "    [Final-Epoch Checkpoint] Época {} completada y guardada.",
                    epoch + 1
                );
            }
        }
        println!("[+] Ciclo de entrenamiento finalizado.");
    }

    // Guardado final (por si no se guardó en el bucle o para confirmar éxito total)
    if let Some(ref path) = save_path {
        println!("[*] Ejecutando guardado final de seguridad en: {}", path);
        _impl::io::loader::save_genomic_model(path, &model, &config, Some(&tokenizer))?;
        println!("[+] Proceso completado exitosamente.");
    }

    if let Some(corpus_path) = eval_corpus {
        println!("[*] Evaluando perplejidad nativa sobre: {}", corpus_path);
        let text = std::fs::read_to_string(&corpus_path)
            .map_err(|e| format!("No se pudo leer el corpus: {}", e))?;

        let lines: Vec<&str> = text
            .lines()
            .map(|l| l.trim())
            .filter(|l| l.len() > 5)
            .collect();

        if lines.is_empty() {
            return Err("Corpus vacío o sin líneas válidas".into());
        }

        let mut total_log_prob = 0.0f32;
        let mut total_tokens = 0usize;
        let mut skipped = 0usize;

        for (idx, line) in lines.iter().enumerate() {
            let tokens = match tokenizer.encode(line, false) {
                Ok(t) if t.len() >= 2 => t,
                _ => {
                    skipped += 1;
                    continue;
                }
            };

            model.clear_cache_core();
            let mut line_log_prob = 0.0f32;

            for i in 0..tokens.len() - 1 {
                let logits = model
                    .forward_core(tokens[i] as usize, false)
                    .map_err(|e| format!("Error en forward pass línea {}: {}", idx, e))?;

                let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let sum_exp: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
                let prob = (logits[tokens[i + 1] as usize] - max_l).exp() / (sum_exp + 1e-12);
                line_log_prob += (prob + 1e-12).ln();
            }

            total_log_prob += line_log_prob;
            total_tokens += tokens.len() - 1;

            if idx % 50 == 0 {
                let ppl_parcial = (-(total_log_prob / total_tokens as f32)).exp();
                println!(
                    "  [Línea {:>4}/{}] PPL parcial: {:.4}",
                    idx + 1,
                    lines.len(),
                    ppl_parcial
                );
            }
        }

        if total_tokens == 0 {
            return Err("No se procesaron tokens válidos".into());
        }

        let ppl = (-(total_log_prob / total_tokens as f32)).exp();

        println!("\n╔══════════════════════════════════════╗");
        println!("║  RESULTADO — EVALUACIÓN NATIVA       ║");
        println!("╠══════════════════════════════════════╣");
        println!("║  Modelo  : {}", corpus_path);
        println!(
            "║  Líneas  : {} procesadas, {} omitidas",
            lines.len() - skipped,
            skipped
        );
        println!("║  Tokens  : {}", total_tokens);
        println!("║  PPL     : {:.4}", ppl);
        println!("║  Log-P   : {:.4}", total_log_prob);
        println!("╚══════════════════════════════════════╝");

        return Ok(());
    }

    if let Some(prompt) = prompt_arg {
        generate(&mut model, &tokenizer, &prompt, 50)?;
    } else if evolve_target.is_none() && train_target.is_none() {
        println!("\n[!] Modo interactivo no disponible en TTY reducido. Use --prompt.");
    }

    Ok(())
}

fn generate(
    model: &mut GenomicLLM,
    tokenizer: &GajeTokenizer,
    prompt: &str,
    max_tokens: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tokens = tokenizer.encode(prompt, false).map_err(|e| e.to_string())?;
    model.clear_cache_core();
    let mut logits = Vec::new();
    for &tid in &tokens {
        logits = model
            .forward_core(tid as usize, false)
            .map_err(|e| e.to_string())?;
    }

    let mut generated_history = Vec::new();
    let penalty = 1.5f32; // Penalización fuerte para romper bucles en Android

    println!("\n[*] Generando con Repetition Penalty (Anti-Loop)...");
    for _ in 0..max_tokens {
        // 1. Aplicar penalización a los logits basados en la historia
        for &past_token in &generated_history {
            if logits[past_token] > 0.0 {
                logits[past_token] /= penalty;
            } else {
                logits[past_token] *= penalty;
            }
        }

        let mut indexed_logits: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        /*
        println!("[Debug] Top-5 candidates:");
        for k in 0..5 {
            let (tid, score) = indexed_logits[k];
            let word = tokenizer.decode(&[tid as u32], true).unwrap_or("???".to_string());
            println!("   {}. [{:>5}] {:<15} score: {:.4}", k+1, tid, word, score);
        }
        */

        let next_token = indexed_logits[0].0;
        generated_history.push(next_token);

        if next_token == 0 || next_token == 151643 {
            break;
        }
        let decoded = tokenizer
            .decode(&[next_token as u32], true)
            .map_err(|e| e.to_string())?;
        print!("{}", decoded);
        io::stdout().flush()?;

        logits = model
            .forward_core(next_token, false)
            .map_err(|e| e.to_string())?;
    }
    println!();
    Ok(())
}
