use _impl::nn::repl::load_model_and_tokenizer;
use std::path::Path;

#[test]
fn test_exact_drift_frontier() {
    let p = "models/production/qwen2_5_1_5b.gaje";
    if !Path::new(p).exists() {
        println!("Modelo no encontrado: {}", p);
        return;
    }

    println!("\n=======================================================");
    println!("🔬 MEDICIÓN DE LA FRONTERA EXACTA DE DERIVA MORFOLÓGICA");
    println!("=======================================================\n");

    let (mut llm, tokenizer) = load_model_and_tokenizer(p).expect("cargar modelo");
    let eos_ids: Vec<usize> = tokenizer
        .get_stop_tokens()
        .into_iter()
        .map(|t| t as usize)
        .collect();

    // Encontrar IDs de tokens relevantes
    let tok_capital_space = tokenizer.encode(" capital", false).unwrap();
    let tok_capitale_space = tokenizer.encode(" capitale", false).unwrap();
    println!("🔍 Token IDs:");
    println!("   • ' capital':  {:?}", tok_capital_space);
    println!("   • ' capitale': {:?}", tok_capitale_space);

    let id_capital = tok_capital_space[0] as usize;
    let id_capitale = tok_capitale_space[0] as usize;

    let user_msg = "¿Cuál es la capital de Argentina?";

    // System prompt base de 22 tokens (que junto con user_msg da N=47)
    let sys_full = "Eres un asistente útil y conciso.";
    
    // Generar diferentes system prompts para barrer N desde 25 hasta 47
    let sys_variants = vec![
        (0, ""),
        (1, "Ok"),
        (2, "Hola."),
        (3, "Responde."),
        (4, "Asistente útil."),
        (5, "Eres un asistente."),
        (6, "Eres un asistente útil."),
        (7, "Eres un asistente muy útil."),
        (8, "Eres un asistente útil y claro."),
        (9, "Eres un asistente útil y conciso."),
        (10, "Eres un asistente de inteligencia artificial útil."),
    ];

    println!("\n------------------------------------------------------------------------------------------------");
    println!(" {:<4} | {:<22} | {:<12} | {:<12} | {:<10} | {:<30}", 
        "N", "System Prompt", "Logit(capital)", "Logit(capit.)", "Diff (C-Ce)", "Respuesta Generada");
    println!("------------------------------------------------------------------------------------------------");

    let mut crossover_n: Option<usize> = None;

    for (_, sys) in sys_variants {
        let chat_prompt = if sys.is_empty() {
            format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", user_msg)
        } else {
            format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", sys, user_msg)
        };

        let prompt_tokens: Vec<usize> = tokenizer
            .encode(&chat_prompt, false)
            .unwrap()
            .into_iter()
            .map(|t| t as usize)
            .collect();

        let n = prompt_tokens.len();

        // 1. Prefill
        llm.clear_cache_core();
        for i in 0..n - 1 {
            llm.forward_blocks_only(prompt_tokens[i]).unwrap();
        }

        // 2. Evaluar primer token de generación tras el prompt
        let first_logits = llm.forward_core(prompt_tokens[n - 1], false).unwrap();
        let first_tok = _impl::compute::sampling::sample_min_p(&first_logits, 0.0, 0.05).unwrap_or(0);

        // 3. Forzar forward del primer token generado (que suele ser "La") para inspeccionar logits del segundo token ("capital" vs "capitale")
        let second_logits = llm.forward_core(first_tok, false).unwrap();
        let l_capital = second_logits.get(id_capital).copied().unwrap_or(0.0);
        let l_capitale = second_logits.get(id_capitale).copied().unwrap_or(0.0);
        let diff = l_capital - l_capitale;

        // 4. Generar el resto de la respuesta
        let mut gen = vec![first_tok];
        let mut curr_logits = second_logits;
        for _ in 0..15 {
            let next_tok = _impl::compute::sampling::sample_min_p(&curr_logits, 0.0, 0.05).unwrap_or(0);
            gen.push(next_tok);
            if eos_ids.contains(&next_tok) { break; }
            curr_logits = llm.forward_core(next_tok, false).unwrap();
        }

        let gen_u32: Vec<u32> = gen.into_iter().map(|t| t as u32).collect();
        let raw = tokenizer.decode(&gen_u32, true).unwrap_or_default();
        let clean = raw.split("<|im_end|>").next().unwrap_or(&raw).trim().to_string();

        let sys_display = if sys.len() > 20 { format!("{}...", &sys[0..17]) } else { sys.to_string() };

        println!(" {:<4} | {:<22} | {:<12.4} | {:<12.4} | {:<+10.4} | {:?}", 
            n, sys_display, l_capital, l_capitale, diff, clean);

        if diff < 0.0 && crossover_n.is_none() {
            crossover_n = Some(n);
        }
    }
    println!("------------------------------------------------------------------------------------------------\n");

    if let Some(cn) = crossover_n {
        println!("🎯 PUNTO EXACTO DE CRUCE (CROSSOVER): N* = {} tokens", cn);
        println!("   Para N < {}, el logit de 'capital' domina sobre 'capitale'.", cn);
        println!("   Para N >= {}, el logit de 'capitale' supera a 'capital' induciendo la deriva morfológica.", cn);
    } else {
        println!("ℹ️ No se detectó cruce en este rango.");
    }
}
