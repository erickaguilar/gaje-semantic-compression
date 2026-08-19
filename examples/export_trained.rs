//! Exporta el modelo estudiante con el cuerpo entrenado (Vía B) a **`.gaje.flat`
//! mmap GAJE`** (zero-copy, el mismo formato de producción que carga el Web UI),
//! para probarlo en el Web UI.
//!
//! Uso:
//!   cargo run --release --example export_trained -- <input.flat> <output.flat> <corpus> \
//!     [n_blk] [lr] [decay] [epochs] [train_lm_head]
//!
//! - `<corpus>`: fichero `.jsonl` (pares prompt/answer como en train_smollm2_1t.jsonl)
//!   o texto plano (se codifica entero).
//! - `n_blk`  (default 16): bloques entrenados desde el final.
//! - `lr`     (default 2e-4): lr base (bloques tardíos).
//! - `decay`  (default 0.8): factor de decaimiento por capa (layer-wise).
//! - `epochs` (default 1): pasadas sobre el corpus.
//! - `train_lm_head` (default 0): 1 para entrenar también el `lm_head` (NO recomendado:
//!   corrompe el vocabulario con corpus pequeños), 0 para congelarlo y entrenar solo el
//!   cuerpo.
//! - `max_tokens` (default 0 = todo): limita el número de tokens a entrenar.
//! - `progress_every` (default 200): imprime progreso cada N tokens.
//!
//! Ejemplo:
//!   cargo run --release --example export_trained -- \
//!     models/production/smollm2_4bit.gaje.flat \
//!     models/production/smollm2_4bit_trained.gaje.flat \
//!     data/distill/train_smollm2_1t.jsonl

use _impl::core::tokenizer::GajeTokenizer;
use _impl::io::config::ModelConfig;
use _impl::io::flat_reader::{load_genomic_auto, GajeFlatFileReader};
use _impl::io::flat_writer::save_genomic_flat;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("uso: export_trained <input.flat> <output.flat> <corpus> [n_blk] [lr] [decay] [epochs]");
        std::process::exit(2);
    }
    let input = &args[1];
    let output = &args[2];
    let corpus_path = &args[3];
    let n_blk: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(16);
    let lr: f32 = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(2e-4);
    let decay: f32 = args.get(6).and_then(|v| v.parse().ok()).unwrap_or(0.8);
    let epochs: usize = args.get(7).and_then(|v| v.parse().ok()).unwrap_or(1);
    let train_lm_head: bool = args
        .get(8)
        .and_then(|v| v.parse::<u8>().ok())
        .map(|v| v != 0)
        .unwrap_or(false);
    let max_tokens: usize = args.get(9).and_then(|v| v.parse().ok()).unwrap_or(0);
    let progress_every: usize = args.get(10).and_then(|v| v.parse().ok()).unwrap_or(200);
    let eval_only: bool = args
        .get(11)
        .and_then(|v| v.parse::<u8>().ok())
        .map(|v| v != 0)
        .unwrap_or(false);

    if eval_only {
        println!("→ Modo EVAL-ONLY: CE del modelo BASE sobre el corpus (sin entrenar)");
    } else {
        println!("→ Entrenando cuerpo ({n_blk} bloques, lr={lr}, decay={decay}, epochs={epochs}, train_lm_head={train_lm_head})");
    }

    let reader = GajeFlatFileReader::open(input).expect("abrir reader para config");
    let config: ModelConfig = serde_json::from_str(&reader.metadata_json)
        .unwrap_or_else(|e| panic!("parsear config desde metadata: {e}"));

    let mut model = load_genomic_auto(input).expect("cargar modelo");
    let tokenizer = GajeTokenizer::from_file("models/core/tokenizer.json").expect("tokenizer");

    let vocab = model.embeddings.out_features;
    let raw = std::fs::read_to_string(corpus_path).expect("leer corpus");

    // Secuencias POR EJEMPLO (no un stream concatenado): cada pareja prompt+answer es una
    // secuencia independiente con cache reseteado. Lección del diagnóstico C: un stream
    // concatenado sin separadores reproduce el problema patológico (base CE ~4.5); cada
    // ejemplo delimitado mantiene la correlación CE <-> generación.
    let mut sequences: Vec<Vec<usize>> = Vec::new();
    if corpus_path.ends_with(".jsonl") {
        for line in raw.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                let p = v["prompt"].as_str().unwrap_or("");
                let a = v["answer"].as_str().unwrap_or("");
                let ids = tokenizer.encode(&format!("{p}{a}"), false).expect("encode");
                if ids.len() >= 2 {
                    sequences.push(ids.iter().map(|id| (*id as usize).min(vocab - 1)).collect());
                }
            }
        }
    } else {
        let ids = tokenizer.encode(&raw, false).expect("encode txt");
        if ids.len() >= 2 {
            sequences.push(ids.iter().map(|id| (*id as usize).min(vocab - 1)).collect());
        }
    }
    let n_tok: usize = sequences.iter().map(|s| s.len()).sum();
    let n_seq = sequences.len();
    println!("→ {n_seq} secuencias, {n_tok} tokens del corpus, vocab={vocab}");
    if max_tokens > 0 && n_tok > max_tokens {
        let mut acc = 0;
        sequences.retain(|s| {
            if acc >= max_tokens {
                return false;
            }
            acc += s.len();
            acc <= max_tokens
        });
        println!("→ limitado a {max_tokens} tokens para medición de rendimiento");
    }

    let mut train_loss = 0.0f32;
    if eval_only {
        let mut total = 0.0f32;
        let mut n = 0usize;
        for seq in &sequences {
            let ce = model.eval_ce_core(seq).expect("eval CE base (forward-only)");
            total += ce * seq.len().saturating_sub(1) as f32;
            n += seq.len().saturating_sub(1);
        }
        let ce = total / n.max(1) as f32;
        println!("BASE CE = {ce:.4}   PPL = {:.2}", ce.exp());
        std::process::exit(0);
    }
    let mut tokens_done = 0usize;
    let t0 = std::time::Instant::now();
    for ep in 0..epochs {
        let mut total = 0.0f32;
        let mut n = 0usize;
        for seq in &sequences {
            let l = model
                .train_sequence_cached_layerwise_core(
                    seq.clone(),
                    lr,
                    n_blk,
                    1.0,
                    decay,
                    train_lm_head,
                    None,
                )
                .expect("entrenar cuerpo");
            total += l * seq.len().saturating_sub(1) as f32;
            n += seq.len().saturating_sub(1);
            tokens_done += seq.len().saturating_sub(1);
            if tokens_done % progress_every == 0 {
                println!(
                    "  progress {tokens_done}/{n_tok} tokens, {:.1}s ({:.0} tok/min)",
                    t0.elapsed().as_secs_f32(),
                    (tokens_done as f32) / (t0.elapsed().as_secs_f32() / 60.0).max(1e-6)
                );
            }
        }
        train_loss = total / n.max(1) as f32;
        println!("  epoch {}/{}: train CE = {train_loss:.4}", ep + 1, epochs);
    }
    model.clear_cache_core();

    println!("→ Guardando {output}");
    save_genomic_flat(output, &model, &config, Some(&tokenizer)).expect("guardar flat mmap");

    // Verificación: recargar y comprobar que el forward es finito.
    let mut reloaded = load_genomic_auto(output).expect("recargar export");
    let first_token = sequences
        .first()
        .and_then(|s| s.first())
        .copied()
        .unwrap_or(0);
    let (logits, _) = reloaded
        .forward_with_hidden_core(first_token, true)
        .expect("forward de verificación");
    let finite = logits.iter().all(|v| v.is_finite());
    println!("✔ export OK (forward finito: {finite}, último train CE {train_loss:.4})");
    if !finite {
        std::process::exit(1);
    }
}