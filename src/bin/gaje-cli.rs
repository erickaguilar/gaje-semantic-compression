use _impl::loader::NativeLoader;
use _impl::nn::RustGenomicLLM;
use _impl::kernels;
use std::env;
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
    
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--model" && i + 1 < args.len() {
            model_path = args[i+1].clone();
            i += 2;
        } else if args[i] == "--prompt" && i + 1 < args.len() {
            prompt_arg = Some(args[i+1].clone());
            i += 2;
        } else if model_path.is_empty() {
            model_path = args[i].clone();
            i += 1;
        } else {
            i += 1;
        }
    }

    if model_path.is_empty() {
        println!("Usage: gaje-cli <model_path.gaje> [--prompt \"your prompt\"]");
        return Ok(());
    }

    println!("🧬 GAJE Native Runtime (v0.6.3)");
    
    println!("[*] Loading model from: {}", model_path);
    let start = Instant::now();
    let loader = NativeLoader::new(&model_path)?;
    
    println!("[*] Extracting tokenizer...");
    let tokenizer = loader.load_tokenizer().map_err(|e| e.to_string())?;

    let mut model = loader.load_llm()?;
    println!("[*] Model & Tokenizer loaded in {:?}", start.elapsed());

    if let Some(prompt) = prompt_arg {
        generate(&mut model, &tokenizer, &prompt, 50)?;
    } else {
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
    
    if current_tokens.is_empty() {
        return Ok(());
    }

    // Initial forward for prompt
    let mut logits = Vec::new();
    for &tid in &current_tokens {
        logits = model.forward(tid as usize, false).unwrap();
    }

    for _ in 0..max_tokens {
        // Simple greedy sampling with NaN safety
        let next_token = logits.iter()
            .enumerate()
            .filter(|(_, &a)| !a.is_nan())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);

        let next_token = match next_token {
            Some(t) => t,
            None => break,
        };

        if next_token == 0 { break; } // Assuming 0 is EOS or pad
        
        let decoded = tokenizer.decode(&[next_token as u32], true).map_err(|e| e.to_string())?;
        print!("{}", decoded);
        io::stdout().flush()?;

        current_tokens.push(next_token as u32);
        logits = model.forward(next_token, false).unwrap();
    }
    
    Ok(())
}
