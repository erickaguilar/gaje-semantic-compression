mod index;
mod kernels;
mod nn;
mod utils;
mod archive;
mod loader;
mod db;

use crate::loader::{NativeLoader, ModelConfig};
use tokenizers::Tokenizer;
use std::env;
use std::time::Instant;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: gaje-chat <model_path.gaje> <tokenizer_path.json>");
        return Ok(());
    }

    let model_path = &args[1];
    let tokenizer_path = &args[2];

    println!("🧬 GAJE Native Runtime (Zero-Python)");
    
    println!("[*] Loading tokenizer: {}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?;

    println!("[*] Loading model: {}", model_path);
    let start = Instant::now();
    let loader = NativeLoader::new(model_path)?;
    
    let mut model = loader.load_llm()?;
    println!("[*] Model loaded in {:?}", start.elapsed());

    loop {
        print!("\n👤 User: ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let prompt = input.trim();
        if prompt.is_empty() { continue; }
        if prompt == "exit" || prompt == "quit" { break; }

        let encoding = tokenizer.encode(prompt, false).map_err(|e| e.to_string())?;
        let tokens = encoding.get_ids();

        print!("🤖 GAJE: ");
        io::stdout().flush()?;

        model.clear_cache().unwrap();
        let mut current_tokens = tokens.to_vec();
        
        if current_tokens.is_empty() {
            println!("(Empty tokens)");
            continue;
        }

        // Initial forward for prompt
        let mut logits = Vec::new();
        for &tid in &current_tokens {
            logits = model.forward(tid as usize, false).unwrap();
        }

        if logits.is_empty() {
            println!("(Error: Empty logits from model)");
            continue;
        }

        for _ in 0..50 {
            // Simple greedy sampling with NaN safety
            let next_token = logits.iter()
                .enumerate()
                .filter(|(_, &a)| !a.is_nan())
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i);

            let next_token = match next_token {
                Some(t) => t,
                None => {
                    println!("(Error: Failed to sample next token - all NaNs or empty)");
                    break;
                }
            };

            if next_token == 0 { break; } // Assuming 0 is EOS or pad
            
            let decoded = tokenizer.decode(&[next_token as u32], true).map_err(|e| e.to_string())?;
            print!("{}", decoded);
            io::stdout().flush()?;

            current_tokens.push(next_token as u32);
            logits = model.forward(next_token, false).unwrap();
        }
        println!();
    }

    Ok(())
}
