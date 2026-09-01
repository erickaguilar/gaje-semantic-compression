use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (mut model, tok) = _impl::nn::repl::load_model_and_tokenizer("models/production/gaje_pico_135m.flat")?;
    let prompts: Vec<String> = (0..50).map(|i| format!("The meaning of life {} is", i)).collect();
    let mut total_tok = 0; let mut total_time = 0.0; let t0 = Instant::now();
    for (idx, prompt) in prompts.iter().enumerate() {
        let toks = tok.encode(prompt, false)?;
        let toks_usize: Vec<usize> = toks.into_iter().map(|t| t as usize).collect();
        let s = Instant::now(); let gen = model.generate_native_core(toks_usize, 100, 0.7, 1.15, vec![2])?;
        let elapsed = s.elapsed().as_secs_f32(); total_tok += gen.len(); total_time += elapsed;
        if idx+1 == 25 {
            println!("Checkpoint 25/50: tok {} time {:.1}s tok/s {:.1}", total_tok, total_time, total_tok as f32/total_time);
            let mem = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
            for line in mem.lines().take(3) { println!("  {}", line); }
        }
        if (idx+1)%10==0 { println!("  {}/50 tok {} tok/s {:.1}", idx+1, total_tok, total_tok as f32/total_time); }
    }
    println!("Done 50/50: total_tok {} total_time {:.1}s tok/s {:.1} wall {:.1}s", total_tok, total_time, total_tok as f32/total_time, t0.elapsed().as_secs_f32());
    Ok(())
}
