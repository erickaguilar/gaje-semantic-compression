use _impl::nn::distiller::{CouncilOfTeachers, Teacher, GenomicDistiller};
use _impl::core::tokenizer::GajeTokenizer;
use _impl::io::loader::NativeLoader;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let global_start = Instant::now();
    println!("🧪 INICIANDO MICRO-DESTILACIÓN OPTIMIZADA (V2) 🧪");
    println!("--------------------------------------------------");

    // Inicializar tablas de aceleración SIMD (CRÍTICO para evitar NaNs)
    unsafe { _impl::compute::kernels::init_shuffle_table(); }
    println!("[*] Tablas de aceleración SIMD inicializadas.");

    // Manejador de interrupción (Graceful Shutdown)
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("\n[!] Interrupción detectada (Ctrl+C). Finalizando de forma segura...");
        r.store(false, Ordering::SeqCst);
    }).expect("Error configurando el manejador de señales");

    // 1. Configuración de Rutas (Volvemos a F16 por integridad, pero con Streaming)
    let student_path = "models/micro_distilled_coherence.gaje";
    let teacher_model_path = "models/gguf/smollm2-135m-f16.gguf";
    let teacher_tokenizer_path = "models/core/tokenizer.json";
    let dataset_path = "data/datasets/mosaic_dataset.txt";
    let output_path = "models/micro_distilled_coherence.gaje";

    // 2. Cargar Tokenizador y Recursos
    println!("[*] Cargando tokenizador...");
    let tokenizer = Arc::new(GajeTokenizer::from_file(teacher_tokenizer_path)?);
    
    // 3. Cargar Estudiante
    println!("[*] Cargando Estudiante (Micro-Organismo)...");
    let (mut student, config) = {
        let loader = NativeLoader::new(student_path)?;
        let student = loader.load_llm()?;
        let config = loader.load_config()?;
        (student, config)
    };
    let student_vocab = student.lm_head.out_features;
    println!("    [+] Vocabulario Estudiante: {}", student_vocab);

    // --- CIRUGÍA DE EMERGENCIA PROFUNDA (Anti-NaN Recursivo) ---
    fn check_linear(l: &_impl::nn::linear::GenomicLinear) -> bool {
        l.centroids.iter().any(|&x| x.is_nan() || x.is_infinite() || x.abs() > 1e10)
    }

    let mut corruption_found = check_linear(&student.embeddings) || check_linear(&student.lm_head);
    if !corruption_found {
        for block in &student.blocks {
            if check_linear(&block.q_gen) || check_linear(&block.k_gen) || check_linear(&block.v_gen) ||
               check_linear(&block.w_o) || check_linear(&block.gate_gen) || check_linear(&block.up_gen) ||
               check_linear(&block.w_down) {
                corruption_found = true;
                break;
            }
        }
    }
    
    let bad_norm = student.output_norm.iter().any(|&x| x.is_nan() || x.is_infinite() || x == 0.0) || student.output_norm.is_empty();

    if corruption_found || bad_norm {
        println!("[!] LIMPIEZA PROFUNDA DE NaNs (Corrupción: {}, BadNorm: {}). Reseteando Estructura...", corruption_found, bad_norm);
        
        student.eps = 1e-6;
        student.output_norm = vec![1.0; config.n_embd];

        let codebook_path = "models/core/algebraic_codebook.json";
        if let Ok(f) = std::fs::File::open(codebook_path) {
            let val: serde_json::Value = serde_json::from_reader(f)?;
            if let Some(arr) = val.get("centroids").and_then(|c| c.as_array()) {
                let algebraic_c: [f32; 4] = [
                    arr[0].as_f64().unwrap_or(0.01) as f32,
                    arr[1].as_f64().unwrap_or(-0.01) as f32,
                    arr[2].as_f64().unwrap_or(0.02) as f32,
                    arr[3].as_f64().unwrap_or(-0.02) as f32,
                ];
                
                let reset_layer = |layer: &mut _impl::nn::linear::GenomicLinear| {
                    for i in 0..layer.centroids.len() {
                        layer.centroids[i] = algebraic_c[i % 4];
                    }
                };

                reset_layer(&mut student.embeddings);
                reset_layer(&mut student.lm_head);
                for block in &mut student.blocks {
                    reset_layer(&mut block.q_gen);
                    reset_layer(&mut block.k_gen);
                    reset_layer(&mut block.v_gen);
                    reset_layer(&mut block.w_o);
                    reset_layer(&mut block.gate_gen);
                    reset_layer(&mut block.up_gen);
                    reset_layer(&mut block.w_down);
                    block.attn.clear_cache_core();
                }
                println!("    [✔] Estructura saneada. El organismo ha renacido.");
            }
        }
    }
    // ------------------------------------------------------------
    
    // --- VERIFICACIÓN DE VITALIDAD FINAL (Pre-entrenamiento) ---
    let (_test_logits, test_h) = student.forward_with_hidden_core(1, true)?;
    let final_nan = test_h.iter().any(|&x| x.is_nan());
    
    if final_nan {
        println!("[!!!] FALLO CRÍTICO DE INTEGRIDAD. Generando modelo base desde cero para evitar NaNs...");
        // Re-inicializamos los centroides con Ruptura de Simetría (Symmetry Breaker)
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let safe_c = [0.01, -0.01, 0.02, -0.02];
        
        let mut reset_l = |l: &mut _impl::nn::linear::GenomicLinear, scale: f32| {
            for i in 0..l.centroids.len() { 
                let noise = (rng.gen::<f32>() - 0.5) * scale;
                l.centroids[i] = safe_c[i % 4] + noise; 
            }
            // LIMPIEZA DE BIAS, ANCLAS Y CENTROIDES SECUNDARIOS
            for x in &mut l.bias { if x.is_nan() || x.is_infinite() { *x = 0.0; } }
            for x in &mut l.epigenetic_centroids { if x.is_nan() || x.is_infinite() { *x = 0.0; } }
            for x in &mut l.triplet_centroids { if x.is_nan() || x.is_infinite() { *x = 0.0; } }
            
            // Si hay anclas corruptas, las ponemos a cero
            let mut a_vals = (*l.anchor_values).clone();
            let mut a_changed = false;
            for v in &mut a_vals {
                let f = v.to_f32();
                if f.is_nan() || f.is_infinite() { *v = half::f16::from_f32(0.0); a_changed = true; }
            }
            if a_changed { l.anchor_values = std::sync::Arc::new(a_vals); }
        };

        reset_l(&mut student.embeddings, 0.01);
        reset_l(&mut student.lm_head, 0.05); // Mayor escala en salida para romper simetría
        student.output_norm = vec![1.0; config.n_embd];
        student.eps = 1e-4;
        for block in &mut student.blocks {
            reset_l(&mut block.q_gen, 0.01); reset_l(&mut block.k_gen, 0.01); reset_l(&mut block.v_gen, 0.01);
            reset_l(&mut block.w_o, 0.01); reset_l(&mut block.gate_gen, 0.01); reset_l(&mut block.up_gen, 0.01);
            reset_l(&mut block.w_down, 0.01);
            
            // RESET CRÍTICO: Normas internas, Homeostasis y Epsilon
            block.attn.rmsnorm_weight = vec![1.0; config.n_embd];
            block.ffn_norm = vec![1.0; config.n_embd];
            block.eps = 1e-4;
            block.attn.eps = 1e-4;
            if block.h_scale.is_nan() || block.h_scale.is_infinite() { block.h_scale = 1.0; }
            
            block.attn.clear_cache_core();
        }
        println!("    [✔] Modelo base generado con Ruptura de Simetría y Normas Limpias. Procediendo...");
    }

    // 4. Configurar Profesor Único (Versión Ligera Q8)
    println!("[*] Configurando Profesor (SmolLM2 Q8 - 42MB)...");
    let mut council = CouncilOfTeachers::new();
    let teacher = Teacher::new(
        "SmolLM2-Master".to_string(),
        teacher_model_path,
        teacher_tokenizer_path,
        &tokenizer
    )?;
    println!("    [+] Vocabulario Profesor: {}", teacher.tokenizer.vocab_size());
    council.add_teacher(teacher);

    // 5. Configurar Destilador
    let distiller = GenomicDistiller::new(council, (*tokenizer).clone());
    let epochs = 5; 
    let lr = 0.005; // Aumentado para romper la inercia del modelo inicial

    println!("\n🚀 Iniciando Ciclo de Crianza Equilibrada ({} épocas)...", epochs);
    
    'training_loop: for epoch in 0..epochs {
        let start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut count = 0;

        // Lectura por streaming para evitar saturar la RAM
        let file = File::open(dataset_path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            if !running.load(Ordering::SeqCst) {
                println!("    [!] Abortando entrenamiento por interrupción.");
                break 'training_loop;
            }

            let text = match line {
                Ok(l) => l.trim().to_string(),
                Err(_) => continue,
            };

            if text.len() < 10 { continue; }

            match distiller.distill_step(&mut student, &text, lr) {
                Ok(loss) => {
                    epoch_loss += loss;
                    count += 1;
                    
                    // Feedback más frecuente (cada 10 muestras)
                    if count % 10 == 0 {
                        println!("\n      [Step {}] Loss: {:.4} | T+{:?}", count, loss, global_start.elapsed());
                    }

                    // Guardado preventivo cada 100 muestras
                    if count % 100 == 0 {
                        let mut current_config = config.clone();
                        current_config.config.state = format!("distilling-e{}-s{}", epoch + 1, count);
                        _impl::io::loader::save_genomic_model(output_path, &student, &current_config, Some(&tokenizer))?;
                    }
                },
                Err(e) => println!("    [!] Error en muestra {}: {}", count, e),
            }
        }

        if count > 0 {
            let avg_loss = epoch_loss / count as f32;
            println!("✅ Época {}/{} completada | Loss: {:.4} | Tiempo: {:?} | Total: {:?}", 
                     epoch + 1, epochs, avg_loss, start.elapsed(), global_start.elapsed());
        }
        
        // Checkpoint final de época
        let mut final_config = config.clone();
        final_config.config.state = format!("completed-epoch-{}", epoch + 1);
        _impl::io::loader::save_genomic_model(output_path, &student, &final_config, Some(&tokenizer))?;
    }

    println!("\n[*] Guardado final de seguridad...");
    let mut exit_config = config.clone();
    exit_config.config.state = "graceful-exit".to_string();
    _impl::io::loader::save_genomic_model(output_path, &student, &exit_config, Some(&tokenizer))?;

    println!("\n✨ Destilación completada o interrumpida en {:?}.", global_start.elapsed());
    println!("[*] Modelo final guardado en: {}", output_path);
    Ok(())
}
