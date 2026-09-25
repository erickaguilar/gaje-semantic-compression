use _impl::nn::repl::load_model_and_tokenizer;
use std::path::Path;

#[test]
fn test_chunking_and_sliding_window_ablation() {
    let p = "models/production/qwen2_5_1_5b.gaje";
    if !Path::new(p).exists() {
        println!("Modelo no encontrado: {}", p);
        return;
    }

    println!("\n=======================================================");
    println!("🔬 EXPERIMENTO DE CHUNKING Y VENTANA DESLIZANTE DE KV CACHE");
    println!("=======================================================\n");

    let (mut llm, tokenizer) = load_model_and_tokenizer(p).expect("cargar modelo");
    let eos_ids: Vec<usize> = tokenizer
        .get_stop_tokens()
        .into_iter()
        .map(|t| t as usize)
        .collect();

    let user_msg = "¿Cuál es la capital de Argentina?";
    let sys_long = "El sistema solar comprende una variedad de planetas rocosos y gaseosos que orbitan alrededor de una estrella central clasificada como enana amarilla de tipo espectral G2V. La física celeste y las leyes de Kepler gobiernan sus trayectorias orbitales con precisión matemática comprobable mediante observaciones astronómicas continuas.";
    let sys_memory = "Eres un asistente útil.\n\nInformación de contexto:\n- Buenos Aires es la capital de Argentina.";

    let prompt_170 = format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        sys_long, user_msg
    );
    let prompt_72 = format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        sys_memory, user_msg
    );

    let tokens_170: Vec<usize> = tokenizer.encode(&prompt_170, false).unwrap().into_iter().map(|t| t as usize).collect();
    let tokens_72: Vec<usize> = tokenizer.encode(&prompt_72, false).unwrap().into_iter().map(|t| t as usize).collect();

    println!("Total tokens prompt largo:  {}", tokens_170.len());
    println!("Total tokens prompt memoria: {}", tokens_72.len());

    // Helper para generar texto dado un estado
    let mut generate_from_current_state = |llm: &mut _impl::nn::llm::GenomicLLM, last_token: usize| -> String {
        let max_tokens = 30;
        let mut last_logits = llm.forward_core(last_token, false).unwrap();
        let mut generated = Vec::new();

        for _ in 0..max_tokens {
            if last_logits.is_empty() { break; }
            let mut logits = last_logits.clone();
            if generated.is_empty() {
                for &eos_id in &eos_ids {
                    if eos_id < logits.len() { logits[eos_id] = -1e9; }
                }
            }
            let next_tok = _impl::compute::sampling::sample_min_p(&logits, 0.0, 0.05).unwrap_or(0);
            generated.push(next_tok);
            if eos_ids.contains(&next_tok) { break; }
            last_logits = llm.forward_core(next_tok, false).unwrap();
        }

        let gen_u32: Vec<u32> = generated.into_iter().map(|t| t as u32).collect();
        let raw = tokenizer.decode(&gen_u32, true).unwrap_or_default();
        raw.split("<|im_end|>").next().unwrap_or(&raw).trim().to_string()
    };

    // -------------------------------------------------------------
    // CASO A: Baseline Estándar N=170 (Full Cache, sin ventana)
    // -------------------------------------------------------------
    llm.clear_cache_core();
    for i in 0..tokens_170.len() - 1 {
        llm.forward_blocks_only(tokens_170[i]).unwrap();
    }
    let reply_a = generate_from_current_state(&mut llm, tokens_170[tokens_170.len() - 1]);
    println!("\n-------------------------------------------------------");
    println!("📋 Caso A: Baseline Estándar N=170 (Full Cache)");
    println!("   • Respuesta: {:?}", reply_a);
    println!("   • Tiene 'Argentiña': {}", reply_a.contains("Argentiña"));
    println!("   • Tiene 'capitale':  {}", reply_a.contains("capitale"));
    println!("   • Tiene 'e s':       {}", reply_a.contains("e s"));

    // -------------------------------------------------------------
    // CASO B: Sliding Window W=32 en prefill sobre N=170
    // (Truncando k_cache y v_cache a los últimos 32 tokens)
    // -------------------------------------------------------------
    llm.clear_cache_core();
    let window_w = 32;
    for i in 0..tokens_170.len() - 1 {
        llm.forward_blocks_only(tokens_170[i]).unwrap();
        for blk in &mut llm.blocks {
            if blk.attn.k_cache.len() > window_w {
                let overflow = blk.attn.k_cache.len() - window_w;
                blk.attn.k_cache.drain(0..overflow);
                blk.attn.v_cache.drain(0..overflow);
            }
        }
    }
    let reply_b = generate_from_current_state(&mut llm, tokens_170[tokens_170.len() - 1]);
    println!("\n-------------------------------------------------------");
    println!("📋 Caso B: Sliding Window W=32 sobre N=170");
    println!("   • Respuesta: {:?}", reply_b);
    println!("   • Tiene 'Argentiña': {}", reply_b.contains("Argentiña"));
    println!("   • Tiene 'capitale':  {}", reply_b.contains("capitale"));
    println!("   • Tiene 'e s':       {}", reply_b.contains("e s"));

    // -------------------------------------------------------------
    // CASO C: Attention Sinks (4 primeros tokens + últimos 28 tokens) sobre N=170
    // -------------------------------------------------------------
    llm.clear_cache_core();
    let num_sinks = 4;
    let local_w = 28;
    for i in 0..tokens_170.len() - 1 {
        llm.forward_blocks_only(tokens_170[i]).unwrap();
        for blk in &mut llm.blocks {
            let total = blk.attn.k_cache.len();
            if total > num_sinks + local_w {
                let drop_start = num_sinks;
                let drop_count = total - (num_sinks + local_w);
                blk.attn.k_cache.drain(drop_start..drop_start + drop_count);
                blk.attn.v_cache.drain(drop_start..drop_start + drop_count);
            }
        }
    }
    let reply_c = generate_from_current_state(&mut llm, tokens_170[tokens_170.len() - 1]);
    println!("\n-------------------------------------------------------");
    println!("📋 Caso C: Attention Sinks (4 sinks + 28 local) sobre N=170");
    println!("   • Respuesta: {:?}", reply_c);
    println!("   • Tiene 'Argentiña': {}", reply_c.contains("Argentiña"));
    println!("   • Tiene 'capitale':  {}", reply_c.contains("capitale"));
    println!("   • Tiene 'e s':       {}", reply_c.contains("e s"));

    // -------------------------------------------------------------
    // CASO D: Memoria RAG N=72 con Baseline Full Cache
    // -------------------------------------------------------------
    llm.clear_cache_core();
    for i in 0..tokens_72.len() - 1 {
        llm.forward_blocks_only(tokens_72[i]).unwrap();
    }
    let reply_d = generate_from_current_state(&mut llm, tokens_72[tokens_72.len() - 1]);
    println!("\n-------------------------------------------------------");
    println!("📋 Caso D: RAG Memoria N=72 (Full Cache Baseline)");
    println!("   • Respuesta: {:?}", reply_d);
    println!("   • Tiene 'capitale':  {}", reply_d.contains("capitale"));
    println!("   • Tiene 'capital':   {}", reply_d.contains("capital"));
    println!("   • Tiene 'e s':       {}", reply_d.contains("e s"));

    // -------------------------------------------------------------
    // CASO E: Memoria RAG N=72 con Sliding Window W=32
    // -------------------------------------------------------------
    llm.clear_cache_core();
    for i in 0..tokens_72.len() - 1 {
        llm.forward_blocks_only(tokens_72[i]).unwrap();
        for blk in &mut llm.blocks {
            if blk.attn.k_cache.len() > window_w {
                let overflow = blk.attn.k_cache.len() - window_w;
                blk.attn.k_cache.drain(0..overflow);
                blk.attn.v_cache.drain(0..overflow);
            }
        }
    }
    let reply_e = generate_from_current_state(&mut llm, tokens_72[tokens_72.len() - 1]);
    println!("\n-------------------------------------------------------");
    println!("📋 Caso E: RAG Memoria N=72 con Sliding Window W=32");
    println!("   • Respuesta: {:?}", reply_e);
    println!("   • Tiene 'capitale':  {}", reply_e.contains("capitale"));
    println!("   • Tiene 'capital':   {}", reply_e.contains("capital"));
    println!("   • Tiene 'e s':       {}", reply_e.contains("e s"));
    println!("-------------------------------------------------------\n");
}
