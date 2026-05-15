mod index;
mod kernels;
mod nn;
mod utils;
mod archive;
mod loader;

use crate::loader::{NativeLoader, ModelConfig};
use tokenizers::Tokenizer;
use std::env;
use std::time::Instant;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: gaje-chat <model_path.gaje> <tokenizer_path.json>");
        return Ok(());
    }

    let model_path = &args[1];
    let tokenizer_path = &args[2];

    println!("🧬 GAJE Native Runtime (Zero-Python)");
    
    println!("[*] Loading tokenizer: {}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;

    println!("[*] Loading model: {}", model_path);
    let start = Instant::now();
    let loader = NativeLoader::new(model_path)?;
    
    // Default config for SmolLM-135M
    let config = ModelConfig {
        name: "SmolLM-135M".to_string(),
        n_embd: 576,
        n_head: 9,
        n_head_kv: 3,
        n_blocks: 30,
        eps: 1e-5,
        rope_base: 10000.0,
        vocab_size: 49152,
        block_size: 128, // GAJE default block size
    };

    let mut model = loader.load_llm(&config);
    println!("[*] Model loaded in {:?}", start.elapsed());

    loop {
        print!("\n👤 User: ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let prompt = input.trim();
        if prompt.is_empty() { continue; }
        if prompt == "exit" || prompt == "quit" { break; }

        let encoding = tokenizer.encode(prompt, false)?;
        let tokens = encoding.get_ids();

        print!("🤖 GAJE: ");
        io::stdout().flush()?;

        model.clear_cache()?;
        let mut current_tokens = tokens.to_vec();
        
        // Initial forward for prompt
        let mut logits = Vec::new();
        for &tid in &current_tokens {
            logits = model.forward(tid as usize, false)?;
        }

        for _ in 0..50 {
            // Simple greedy sampling for now
            let next_token = logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            if next_token == 0 { break; } // Assuming 0 is EOS or pad
            
            let decoded = tokenizer.decode(&[next_token as u32], true)?;
            print!("{}", decoded);
            io::stdout().flush()?;

            current_tokens.push(next_token as u32);
            logits = model.forward(next_token, false)?;
        }
        println!();
    }

    Ok(())
}
