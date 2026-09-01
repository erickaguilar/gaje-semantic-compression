use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (mut model, tok) = _impl::nn::repl::load_model_and_tokenizer("models/production/gaje_pro_3b.flat")?;
    let mut total_tok = 0; let mut total_time = 0.0; let t0 = Instant::now();
    for i in 0..300 {
        let prompt = format!("Explain topic {} in detail with examples", i);
        let toks = tok.encode(&prompt, false)?;
        let toks_usize: Vec<usize> = toks.into_iter().map(|t| t as usize).collect();
        let s = Instant::now();
        let gen = model.generate_native_core(toks_usize, 120, 0.7, 1.15, vec![2])?;
        let elapsed = s.elapsed().as_secs_f32();
        total_tok += gen.len(); total_time += elapsed;
        if (i+1)%50==0 {
            println!("{}/300 tok {} tok/s {:.1} wall {:.0}s free {}", i+1, total_tok, total_tok as f32/total_time, t0.elapsed().as_secs_f32(), std::fs::read_to_string("/proc/meminfo").unwrap().lines().next().unwrap_or(""));
        }
    }
    println!("Done 300/300: tok {} time {:.1}s tok/s {:.1} wall {:.1}s", total_tok, total_time, total_tok as f32/total_time, t0.elapsed().as_secs_f32());
    // Save to corpus
    Ok(())
}
