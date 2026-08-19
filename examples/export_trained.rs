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
    let mut stream: Vec<usize> = Vec::new();
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
                for id in ids {
                    stream.push((id as usize).min(vocab - 1));
                }
            }
        }
    } else {
        let ids = tokenizer.encode(&raw, false).expect("encode txt");
        for id in ids {
            stream.push((id as usize).min(vocab - 1));
        }
    }
    let n_tok = stream.len();
    println!("→ {n_tok} tokens del corpus, vocab={vocab}");
    if max_tokens > 0 && stream.len() > max_tokens {
        stream.truncate(max_tokens);
        println!("→ limitado a {max_tokens} tokens para medición de rendimiento");
    }

    let mut train_loss = 0.0f32;
    if eval_only {
        let ce = model
            .eval_ce_core(&stream)
            .expect("eval CE base (forward-only)");
        println!("BASE CE = {ce:.4}   PPL = {:.2}", ce.exp());
        std::process::exit(0);
    }
    for ep in 0..epochs {
        let l = model
            .train_sequence_cached_layerwise_core(
                stream.clone(),
                lr,
                n_blk,
                1.0,
                decay,
                train_lm_head,
                Some(progress_every),
            )
            .expect("entrenar cuerpo");
        train_loss = l;
        println!("  epoch {}/{}: train CE = {l:.4}", ep + 1, epochs);
    }
    model.clear_cache_core();

    println!("→ Guardando {output}");
    save_genomic_flat(output, &model, &config, Some(&tokenizer)).expect("guardar flat mmap");

    // Verificación: recargar y comprobar que el forward es finito.
    let mut reloaded = load_genomic_auto(output).expect("recargar export");
    let (logits, _) = reloaded
        .forward_with_hidden_core(stream[0], true)
        .expect("forward de verificación");
    let finite = logits.iter().all(|v| v.is_finite());
    println!("✔ export OK (forward finito: {finite}, último train CE {train_loss:.4})");
    if !finite {
        std::process::exit(1);
    }
}