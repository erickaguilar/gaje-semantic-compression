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
        Self {
            model_path: "models/production/gaje_pico_135m.flat".to_string(),
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

    let mut memory_orch = crate::compute::island::IslandOrchestrator::try_load_paired_memory(&config.model_path, llm.dim() as u32);
    if let Some(ref orch) = memory_orch {
        let total_facts = orch.documental.entries.len() + orch.episodic.entries.len() + orch.conversational.entries.len();
        println!("🧠 Hipocampo Congénito: {} hechos activos en memoria asociativa (.gmem)", total_facts);
    }
    println!();
    println!("Comandos disponibles:");
    println!("  /reset   - Limpia el historial de conversación y el KV-Cache");
    println!("  /stats   - Muestra las estadísticas de memoria y configuración");
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
                println!("  /exit    - Salir del REPL\n");
                continue;
            }
            "/stats" => {
                println!("\n📊 Métricas del Organismo Activo:");
                println!("   • Archivo:             {}", config.model_path);
                println!("   • Turnos en Memoria:   {}", history.len());
                println!("   • Temperatura:         {}", config.temperature);
                println!("   • Repetition Penalty:  {}\n", config.repetition_penalty);
                continue;
            }
            _ => {}
        }

        // Búsqueda en Hipocampo Congénito (.gmem)
        let mut context_prefix = String::new();
        if let Some(ref orch) = memory_orch {
            let q_vec = crate::compute::island::IslandOrchestrator::vector_from_text(input_trimmed, llm.dim());
            let matches = orch.retrieve_context(&q_vec, 2);
            let relevant: Vec<_> = matches.into_iter().filter(|m| m.similarity >= 0.50).collect();
            if !relevant.is_empty() {
                context_prefix = format!("[Conocimiento Hipocampal: {}]\n", relevant.iter().map(|m| m.text.as_str()).collect::<Vec<_>>().join(" | "));
            }
        }

        // Construir prompt con ChatML
        let mut full_prompt = format!("<|im_start|>system\n{}{}<|im_end|>\n", config.system_prompt, if context_prefix.is_empty() { "".to_string() } else { format!("\n{}", context_prefix) });
        for (u, a) in history.iter().rev().take(3).rev() {
            full_prompt.push_str(&format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>\n",
                u, a
            ));
        }
        full_prompt.push_str(&format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            input_trimmed
        ));

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

        if let Some(ref mut orch) = memory_orch {
            let turn_id = (history.len() + 1) as u64;
            let entry_text = format!("U: {} | A: {}", input_trimmed, clean_reply);
            let entry_vec = crate::compute::island::IslandOrchestrator::vector_from_text(&entry_text, llm.dim());
            orch.add_memory(crate::compute::island::IslandNiche::Conversational, turn_id, entry_vec, entry_text);
        }

        history.push((input_trimmed.to_string(), clean_reply));
    }

    Ok(())
}
