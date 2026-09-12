//! 💬 Terminal REPL Interactivo (gaje-cli chat)

use crate::core::tokenizer::GajeTokenizer;
use crate::nn::llm::GenomicLLM;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub struct ReplConfig {
    pub model_path: String,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub max_tokens: usize,
    pub system_prompt: String,
}

impl Default for ReplConfig {
    fn default() -> Self {
        let default_path = if Path::new("models/production/gaje_pico_135m.gaje").exists() {
            "models/production/gaje_pico_135m.gaje"
        } else if Path::new("models/production/gaje_pico_135m.flat").exists() {
            "models/production/gaje_pico_135m.flat"
        } else if Path::new("models/production/qwen2_5_0_5b_q2_0.gaje").exists() {
            "models/production/qwen2_5_0_5b_q2_0.gaje"
        } else if Path::new("models/production/qwen2_5_0_5b.gaje").exists() {
            "models/production/qwen2_5_0_5b.gaje"
        } else {
            "models/production/gaje_pico_135m.gaje"
        };
        Self {
            model_path: default_path.to_string(),
            temperature: 0.4,
            repetition_penalty: 1.15,
            max_tokens: 256,
            system_prompt: "Eres GAJE AI, un asistente genómico soberano, conciso y útil."
                .to_string(),
        }
    }
}

pub fn load_model_and_tokenizer(model_path: &str) -> Result<(GenomicLLM, GajeTokenizer), String> {
    let path = Path::new(model_path);
    if !path.exists() {
        return Err(format!(
            "El archivo del modelo no existe: {:?}\n💡 Tip: Descárgalo con `gaje-cli pull pico`",
            path
        ));
    }

    let reader = crate::io::flat_reader::GajeFlatFileReader::open(model_path)
        .map_err(|e| format!("Error abriendo modelo GAJE: {}", e))?;
    let model = reader
        .load_genomic()
        .map_err(|e| format!("Error cargando LLM: {}", e))?;
    let gtok = reader.get_embedded_gtok().ok_or_else(|| {
        "No se encontró tokenizador GTOK incrustado en el modelo.\n💡 El modelo debe contener el tokenizador nativo unificado.\n   Regenere el modelo con GTOK embebido."
            .to_string()
    })?;
    let tokenizer = GajeTokenizer::from_gtok(gtok);
    Ok((model, tokenizer))
}

pub fn run_repl(
    config: ReplConfig,
    running: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!(
        "\n🧬 ==============================================================================="
    );
    println!("💬 GAJE HELIX — Terminal REPL Interactivo (Inferencia Soberana Zero-Server)");
    println!("===============================================================================\n");
    println!("📦 Cargando organismo: {}...", config.model_path);

    let t0 = Instant::now();
    let (mut llm, tokenizer) = load_model_and_tokenizer(&config.model_path)?;
    let load_time = t0.elapsed();

    println!(
        "✅ Modelo listo en {:.2} ms (Memoria mmap zero-copy activa)",
        load_time.as_secs_f64() * 1000.0
    );

    let (memory_threshold, memory_uses_whitening, memory_mu, memory_whitening_missing) =
        crate::compute::island::configure_model_memory(Path::new(&config.model_path), llm.dim());
    let (mut memory_orch, memory_dir) = crate::compute::island::IslandOrchestrator::load_paired_for_model(
        Path::new(&config.model_path),
        llm.dim() as u32,
        memory_threshold,
    );
    let mut use_memory = false; // Modo "sin memoria" por defecto (Regla de Oro)

    let total_facts = memory_orch.documental.entries.len()
        + memory_orch.episodic.entries.len()
        + memory_orch.conversational.entries.len();
    if total_facts > 0 {
        println!(
            "🧠 Hipocampo Congénito: {} hechos activos en memoria asociativa (.gmem, τ*={:.2})",
            total_facts,
            memory_threshold
        );
    }
    println!();
    println!("Comandos disponibles:");
    println!("  /reset   - Limpia el historial de conversación y el KV-Cache");
    println!("  /stats   - Muestra las estadísticas del modelo");
    println!("  /memory  - Consulta o alterna la memoria hipocampal (/memory on | off)");
    println!("  /help    - Muestra este menú de ayuda");
    println!("  /exit    - Finaliza la sesión interactiva\n");
    println!("-------------------------------------------------------------------------------");

    let stdin = io::stdin();
    let mut history: Vec<(String, String)> = Vec::new();

    while running.load(Ordering::SeqCst) {
        print!("gaje ❯ ");
        io::stdout().flush()?;

        let mut user_input = String::new();
        if stdin.lock().read_line(&mut user_input)? == 0 {
            break; // EOF
        }

        let input_trimmed = user_input.trim();
        if input_trimmed.is_empty() {
            continue;
        }

        // Comandos especiales
        match input_trimmed {
            "/exit" | "/quit" | ":q" => {
                println!("👋 Finalizando sesión interactiva. ¡Hasta pronto!");
                break;
            }
            "/reset" | "/clear" => {
                history.clear();
                llm.clear_cache_core();
                println!("🧹 Historial y KV-Cache reiniciados con éxito.");
                continue;
            }
            "/help" => {
                println!("\nComandos disponibles:");
                println!("  /reset   - Limpia el historial de conversación");
                println!("  /stats   - Muestra métricas del modelo");
                println!("  /memory  - Estado de memoria (/memory on | off)");
                println!("  /exit    - Salir del REPL\n");
                continue;
            }
            "/memory" | "/memory status" => {
                println!("\n🧠 Estado de Memoria Hipocampal:");
                println!("   • Activa:              {}", if use_memory { "SÍ (on)" } else { "NO (off)" });
                println!("   • Directorio:          {:?}", memory_dir);
                let total = memory_orch.episodic.entries.len()
                    + memory_orch.documental.entries.len()
                    + memory_orch.conversational.entries.len();
                println!("   • Recuerdos cargados:  {}", total);
                println!("   • Umbral τ*:           {:.2}", memory_threshold);
                println!("   • Whitening:           {}", if memory_uses_whitening { "Activo (μ cargado)" } else { "Inactivo" });
                if memory_whitening_missing {
                    println!("   ⚠️ Aviso:              .mu.bin faltante/corrupto (Fallback Opción C)");
                }
                println!("   💡 Tip: Usa `/memory on` o `/memory off` para cambiar el estado.\n");
                continue;
            }
            "/memory on" => {
                use_memory = true;
                println!("🧠 Memoria Hipocampal ACTIVADA (use_memory: true).");
                continue;
            }
            "/memory off" => {
                use_memory = false;
                println!("🧠 Memoria Hipocampal DESACTIVADA (use_memory: false).");
                continue;
            }
            "/stats" => {
                println!("\n📊 Métricas del Organismo Activo:");
                println!("   • Archivo:             {}", config.model_path);
                println!("   • Turnos en Memoria:   {}", history.len());
                println!("   • Temperatura:         {}", config.temperature);
                println!("   • Repetition Penalty:  {}", config.repetition_penalty);
                println!("   • Memoria Hipocampal:  {}\n", if use_memory { "Activa" } else { "Inactiva" });
                continue;
            }
            _ => {}
        }

        let chat_history: Vec<crate::compute::island::ChatMessage> = history
            .iter()
            .flat_map(|(u, a)| {
                vec![
                    crate::compute::island::ChatMessage::new("user", u.clone()),
                    crate::compute::island::ChatMessage::new("assistant", a.clone()),
                ]
            })
            .collect();

        let template = crate::compute::island::detect_chat_template_from_tokenizer(&tokenizer);
        let mem_config = crate::compute::island::MemoryConfig {
            use_memory,
            threshold: memory_threshold,
            uses_whitening: memory_uses_whitening,
            mu_vector: memory_mu.clone(),
            whitening_missing: memory_whitening_missing,
            max_tokens_context: 128,
        };

        let (full_prompt, telemetry) = crate::compute::island::prepare_prompt_with_memory(
            input_trimmed,
            Some(&chat_history),
            &config.system_prompt,
            template,
            &llm,
            &tokenizer,
            Some(&memory_orch),
            &mem_config,
        );

        if use_memory {
            println!(
                "🧠 [Memoria: {} | Inyectados: {} | Latencia: {:.1} ms]",
                telemetry.state,
                telemetry.facts_injected,
                telemetry.latency_ms
            );
        }

        print!("\n🧬 GAJE: ");
        io::stdout().flush()?;

        let prompt_tokens_u32 = tokenizer
            .encode(&full_prompt, false)
            .map_err(|e| e.to_string())?;
        let prompt_tokens: Vec<usize> = prompt_tokens_u32.into_iter().map(|t| t as usize).collect();

        let gen_t0 = Instant::now();
        let eos_ids = vec![2, 0];
        let generated_tokens = match llm.generate_native_core(
            prompt_tokens,
            config.max_tokens,
            config.temperature,
            config.repetition_penalty,
            eos_ids,
        ) {
            Ok(tokens) => tokens,
            Err(e) => {
                println!("\n❌ Error durante la inferencia: {}", e);
                continue;
            }
        };

        let gen_u32: Vec<u32> = generated_tokens.into_iter().map(|t| t as u32).collect();
        let raw_reply = tokenizer.decode(&gen_u32, true).unwrap_or_default();

        let clean_reply = raw_reply
            .replace("<|im_end|>", "")
            .replace("<|im_start|>", "")
            .replace("<|endoftext|>", "")
            .trim()
            .to_string();

        println!("{}", clean_reply);

        let elapsed = gen_t0.elapsed().as_secs_f64();
        let tok_count = gen_u32.len();
        let tps = tok_count as f64 / elapsed.max(0.001);

        println!(
            "\n\x1b[90m[{:.1} tok/s · {} tokens · {:.2}s]\x1b[0m\n",
            tps, tok_count, elapsed
        );

        if use_memory {
            let turn_id = (history.len() + 1) as u64;
            let entry_text = format!("U: {} | A: {}", input_trimmed, clean_reply);
            let entry_vec = llm
                .embed_text(&entry_text, &tokenizer)
                .unwrap_or_else(|_| crate::compute::island::IslandOrchestrator::vector_from_text(&entry_text, llm.dim()));
            memory_orch.add_memory(
                crate::compute::island::IslandNiche::Conversational,
                turn_id,
                entry_vec,
                entry_text,
            );
        }

        history.push((input_trimmed.to_string(), clean_reply));
    }

    Ok(())
}
