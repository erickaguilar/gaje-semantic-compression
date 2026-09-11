#[cfg(test)]
mod tests {
    use _impl::io::flat_reader::GajeFlatFileReader;
    use _impl::io::header::Q4_0Block;
    use _impl::nn::linear::database::WeightDatabase;
    use _impl::nn::linear::GenomicLinear;
    use _impl::nn::llm::GenomicLLM;
    use half::f16;
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;
    use std::time::Instant;

    #[derive(Serialize, Deserialize, Debug)]
    struct PromptResult {
        id: String,
        prompt_text: String,
        prompt_tokens: Vec<usize>,
        tokenization_roundtrip_ok: bool,
        generated_tokens: Vec<usize>,
        generated_text: String,
        stop_reason: String,
        token_1_agreement_with_q4_0: bool,
        top_5_agreement: f32,
        distinct_1: f32,
        mean_top1_prob: f32,
        mean_entropy: f32,
        is_degenerate: bool,
        generation_time_ms: u128,
        determinism_ok: bool,
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct LongPromptMilestone {
        token_pos: usize,
        sc_body_l10: f32,
        sc_exit_l23: f32,
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct LongPromptResult {
        id: String,
        prompt_text: String,
        tokenization_roundtrip_ok: bool,
        generated_tokens: Vec<usize>,
        generated_text: String,
        stop_reason: String,
        sc_reference: String,
        sc_candidate: String,
        milestones: Vec<LongPromptMilestone>,
        mean_sc_l10: f32,
        mean_sc_l23: f32,
        distinct_1_last_50: f32,
        is_degenerate_last_50: bool,
        semantic_focus_hits: usize,
        semantic_focus_ok: bool,
        generation_time_ms: u128,
        determinism_ok: bool,
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct ConfigReport {
        config: String,
        sc_reference: String,
        sc_candidate: String,
        penalty: f32,
        temperature: f32,
        short_prompts: Vec<PromptResult>,
        long_prompts: Vec<LongPromptResult>,
    }

    #[inline(always)]
    fn norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[inline(always)]
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[inline(always)]
    fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
        let na = norm(a);
        let nb = norm(b);
        if na > 1e-12 && nb > 1e-12 {
            (dot(a, b) / (na * nb)).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    fn quantize_linear_to_3bit(linear_q4: &GenomicLinear) -> GenomicLinear {
        let mut new_linear = linear_q4.clone();
        if let WeightDatabase::GenomicQ4_0(db) = &linear_q4.weight_db {
            let mut new_blocks = Vec::with_capacity(db.len());
            for block in db.iter() {
                let mut w = [0.0f32; 32];
                let mut w_min = f32::INFINITY;
                let mut w_max = -f32::INFINITY;
                for i in 0..32 {
                    let val = block.dequantize_weight(i);
                    w[i] = val;
                    if val < w_min {
                        w_min = val;
                    }
                    if val > w_max {
                        w_max = val;
                    }
                }
                let diff = w_max - w_min;
                let mut new_block = Q4_0Block {
                    scale: f16::from_f32(0.0),
                    min: f16::from_f32(w_min),
                    qs: [0; 16],
                };
                if diff > 1e-8 {
                    let scale_f32 = diff / 7.0;
                    let scale_f16 = f16::from_f32(scale_f32);
                    let min_f16 = f16::from_f32(w_min);
                    let effective_scale = scale_f16.to_f32();
                    let effective_min = min_f16.to_f32();

                    new_block.scale = scale_f16;
                    new_block.min = min_f16;
                    for i in 0..32 {
                        let q = ((w[i] - effective_min) / effective_scale)
                            .round()
                            .clamp(0.0, 7.0) as u8;
                        new_block.set_q_value(i, q);
                    }
                }
                new_blocks.push(new_block);
            }
            new_linear.weight_db = WeightDatabase::GenomicQ4_0(Arc::new(new_blocks));
        }
        new_linear
    }

    fn build_chatml_tokens(
        tokenizer: &_impl::core::gtok::GtokNativeTokenizer,
        user_text: &str,
    ) -> Vec<usize> {
        let mut tokens = vec![151644, 872, 198]; // <|im_start|>user\n
        tokens.extend(tokenizer.encode(user_text).into_iter().map(|t| t as usize));
        tokens.extend(vec![151645, 198, 151644, 77091, 198]); // <|im_end|>\n<|im_start|>assistant\n
        tokens
    }

    struct GenOutput {
        tokens: Vec<usize>,
        stop_reason: String,
        mean_top1_prob: f32,
        mean_entropy: f32,
        time_ms: u128,
        // [generated_token_idx] -> (h_layer_10, h_layer_23)
        hidden_states: Vec<(Vec<f32>, Vec<f32>)>,
    }

    fn generate_detailed(
        llm: &mut GenomicLLM,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        repetition_penalty: f32,
        eos_ids: &[usize],
        capture_hidden: bool,
    ) -> GenOutput {
        let t0 = Instant::now();
        llm.clear_cache_core();

        for &t in &prompt_tokens[..prompt_tokens.len() - 1] {
            let _ = llm.forward_blocks_only(t);
        }

        let mut last_logits = llm.forward_core(prompt_tokens[prompt_tokens.len() - 1], false).unwrap_or_default();
        let mut generated = Vec::new();
        let mut stop_reason = "max_tokens".to_string();
        let mut top1_probs = Vec::new();
        let mut entropies = Vec::new();
        let mut hidden_states = Vec::new();

        for _ in 0..max_new_tokens {
            if last_logits.is_empty() {
                break;
            }

            // Repetition penalty
            let mut logits = last_logits.clone();
            if repetition_penalty > 1.0 {
                let mut seen = std::collections::HashSet::new();
                for &tok in &generated {
                    seen.insert(tok);
                }
                for &e in eos_ids {
                    seen.remove(&e);
                }
                for &tok in &seen {
                    if tok < logits.len() {
                        if logits[tok] < 0.0 {
                            logits[tok] *= repetition_penalty;
                        } else {
                            logits[tok] /= repetition_penalty;
                        }
                    }
                }
            }

            // Confianza y entropía
            let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0f32;
            for &l in &logits {
                sum_exp += (l - max_l).exp();
            }
            let inv_sum = 1.0 / sum_exp.max(1e-12);

            let mut argmax_idx = 0usize;
            let mut argmax_p = 0.0f32;
            let mut entropy = 0.0f32;

            for (idx, &l) in logits.iter().enumerate() {
                let p = (l - max_l).exp() * inv_sum;
                if p > argmax_p {
                    argmax_p = p;
                    argmax_idx = idx;
                }
                if p > 1e-12 {
                    entropy -= p * p.ln();
                }
            }

            top1_probs.push(argmax_p);
            entropies.push(entropy);

            let next_tok = argmax_idx; // Greedy T=0.0
            generated.push(next_tok);

            if eos_ids.contains(&next_tok) {
                stop_reason = "eos".to_string();
                break;
            }

            // Loop detector
            let mut repeated = false;
            for w in 2..=32 {
                if generated.len() >= w * 3 {
                    let l_end = generated.len();
                    let chunk3 = &generated[l_end - w..l_end];
                    let chunk2 = &generated[l_end - 2 * w..l_end - w];
                    let chunk1 = &generated[l_end - 3 * w..l_end - 2 * w];
                    if chunk3 == chunk2 && chunk2 == chunk1 {
                        repeated = true;
                        break;
                    }
                }
            }
            if repeated {
                stop_reason = "loop".to_string();
                break;
            }

            // Forward next token
            if capture_hidden {
                let mut h = llm.get_token_embedding(next_tok).unwrap();
                let mut h10 = Vec::new();
                let mut h23 = Vec::new();
                for l in 0..24 {
                    h = llm.blocks[l].forward_core(h, prompt_tokens.len() + generated.len() - 1).unwrap();
                    if l == 10 {
                        h10 = h.clone();
                    } else if l == 23 {
                        h23 = h.clone();
                    }
                }
                hidden_states.push((h10, h23));
                let h_norm = unsafe {
                    _impl::compute::kernels::rms_norm_offset(&h, &llm.output_norm, llm.eps, 0.0)
                };
                last_logits = llm.lm_head.forward_core(h_norm, None, false).unwrap();
            } else {
                last_logits = llm.forward_core(next_tok, false).unwrap_or_default();
            }
        }

        let elapsed = t0.elapsed().as_millis();
        let mean_p = if top1_probs.is_empty() { 0.0 } else { top1_probs.iter().sum::<f32>() / top1_probs.len() as f32 };
        let mean_ent = if entropies.is_empty() { 0.0 } else { entropies.iter().sum::<f32>() / entropies.len() as f32 };

        GenOutput {
            tokens: generated,
            stop_reason,
            mean_top1_prob: mean_p,
            mean_entropy: mean_ent,
            time_ms: elapsed,
            hidden_states,
        }
    }

    #[test]
    fn test_bilingual_sustained_generation_harness() {
        let path4 = "models/production/qwen2_5_0_5b.gaje";
        let path2 = "models/qwen2_5_0_5b_q2_0.flat";

        println!("\n==================================================================================");
        println!("🧬 GAJE HELIX — ARNÉS OFICIAL DE GENERACIÓN BILINGÜE Y COHERENCIA SOSTENIDA");
        println!("==================================================================================");

        let reader4 = GajeFlatFileReader::open(path4).expect("Open Q4");
        let reader2 = GajeFlatFileReader::open(path2).expect("Open Q2");
        let tokenizer = reader4.get_embedded_gtok().expect("Embedded GTOK");

        let eos_ids = vec![151645, 151643];

        // 1. Definición de Prompts Cortos
        let short_prompts = [
            ("EN-1", "What is the capital of France?"),
            ("EN-2", "The Earth orbits around the"),
            ("EN-3", "Water is composed of hydrogen and"),
            ("ZH-1", "太阳系中最大的行星是哪一颗？"),
            ("ZH-2", "法国的首都是哪座城市？"),
            ("ZH-3", "光合作用是植物利用阳光、水和二氧化碳制造"),
        ];

        // 2. Definición del Par Bilingüe Largo
        let long_en = (
            "LONG-EN",
            "The Solar System is the gravitationally bound system of the Sun and the objects that orbit it. The four inner system planets are Mercury, Venus, Earth, and Mars, which are terrestrial planets composed primarily of rock and metal. Describe in detail the physical characteristics of these inner planets, their atmospheric composition, surface temperatures, and how their orbital periods follow Kepler's laws of planetary motion.",
            vec!["Mercury", "Venus", "Earth", "Mars", "atmosphere", "temperature", "orbital", "planet"],
        );

        let long_zh = (
            "LONG-ZH",
            "太阳系是由太阳以及所有受其引力约束的天体构成的系统。太阳系的四颗类地行星是水星、金星、地球和火星，它们主要由岩石和金属组成。请详细阐述这四颗内行星的物理特征、大气成分、表面温度，以及它们的公转周期如何遵循开普勒行星运动定律。",
            vec!["水星", "金星", "地球", "火星", "大气", "温度", "轨道", "行星", "引力", "开普勒"],
        );

        // PASO 1: VERIFICACIÓN DE TOKENIZACIÓN ROUNDTRIP
        println!("\n🔍 [PASO 1/5] Verificando Tokenización Roundtrip (encode -> decode)...");
        for (id, prompt) in &short_prompts {
            let encoded = tokenizer.encode(prompt);
            let decoded = tokenizer.decode(&encoded);
            let ok = decoded.trim() == prompt.trim();
            println!("  • [{}] Roundtrip OK: {} ('{}')", id, ok, prompt);
            assert!(ok, "Fallo de roundtrip en prompt {}", id);
        }
        let en_enc = tokenizer.encode(long_en.1);
        let en_dec = tokenizer.decode(&en_enc);
        assert_eq!(en_dec.trim(), long_en.1.trim(), "Fallo roundtrip LONG-EN");
        println!("  • [LONG-EN] Roundtrip OK: true");

        let zh_enc = tokenizer.encode(long_zh.1);
        let zh_dec = tokenizer.decode(&zh_enc);
        assert_eq!(zh_dec.trim(), long_zh.1.trim(), "Fallo roundtrip LONG-ZH");
        println!("  • [LONG-ZH] Roundtrip OK: true");

        // Configuraciones a contrastar
        let config_names = [
            "q4_0_control",
            "pure_q2_0",
            "var_b_attn_q4_ffn_q2",
            "var_a_attn_q4_ffn_3bit",
        ];

        let mut reports: Vec<ConfigReport> = Vec::new();

        // Estructuras de referencia Q4_0
        let mut q4_short_tokens: Vec<Vec<usize>> = Vec::new();
        let mut q4_long_en_output: Option<GenOutput> = None;
        let mut q4_long_zh_output: Option<GenOutput> = None;

        let m_ctrl_base = reader4.load_genomic().expect("Load Q4 base");

        for cfg_name in &config_names {
            println!("\n⚙️  Configurando modelo: {} ...", cfg_name);
            let mut model = match *cfg_name {
                "q4_0_control" => reader4.load_genomic().unwrap(),
                "pure_q2_0" => reader2.load_genomic().unwrap(),
                "var_b_attn_q4_ffn_q2" => {
                    let mut m = reader2.load_genomic().unwrap();
                    for l in 0..24 {
                        m.blocks[l].q_gen = m_ctrl_base.blocks[l].q_gen.clone();
                        m.blocks[l].k_gen = m_ctrl_base.blocks[l].k_gen.clone();
                        m.blocks[l].v_gen = m_ctrl_base.blocks[l].v_gen.clone();
                        m.blocks[l].w_o = m_ctrl_base.blocks[l].w_o.clone();
                    }
                    m
                }
                "var_a_attn_q4_ffn_3bit" => {
                    let mut m = reader2.load_genomic().unwrap();
                    for l in 0..24 {
                        m.blocks[l].q_gen = m_ctrl_base.blocks[l].q_gen.clone();
                        m.blocks[l].k_gen = m_ctrl_base.blocks[l].k_gen.clone();
                        m.blocks[l].v_gen = m_ctrl_base.blocks[l].v_gen.clone();
                        m.blocks[l].w_o = m_ctrl_base.blocks[l].w_o.clone();

                        m.blocks[l].gate_gen = quantize_linear_to_3bit(&m_ctrl_base.blocks[l].gate_gen);
                        m.blocks[l].up_gen = quantize_linear_to_3bit(&m_ctrl_base.blocks[l].up_gen);
                        m.blocks[l].w_down = quantize_linear_to_3bit(&m_ctrl_base.blocks[l].w_down);
                    }
                    m
                }
                _ => unreachable!(),
            };

            // Ejecución de Prompts Cortos (35 tokens)
            println!("   ▶ Ejecutando 6 prompts cortos (max 35 tokens, penalty=1.10)...");
            let mut prompt_results = Vec::new();

            for (p_idx, (id, prompt)) in short_prompts.iter().enumerate() {
                let p_tokens = build_chatml_tokens(&tokenizer, prompt);
                let out = generate_detailed(&mut model, &p_tokens, 35, 1.10, &eos_ids, false);

                let u32_tok: Vec<u32> = out.tokens.iter().map(|&t| t as u32).collect();
                let text = tokenizer.decode(&u32_tok);

                // Chequeo de determinismo (corrida 2)
                let out2 = generate_detailed(&mut model, &p_tokens, 35, 1.10, &eos_ids, false);
                let determinism_ok = out.tokens == out2.tokens;

                // Agreement con Q4_0
                let (t1_agree, top5_agree) = if *cfg_name == "q4_0_control" {
                    q4_short_tokens.push(out.tokens.clone());
                    (true, 1.0f32)
                } else {
                    let q4_t = &q4_short_tokens[p_idx];
                    let t1 = !out.tokens.is_empty() && !q4_t.is_empty() && out.tokens[0] == q4_t[0];
                    let mut matches = 0;
                    let comp_len = 5.min(out.tokens.len()).min(q4_t.len());
                    for i in 0..comp_len {
                        if out.tokens[i] == q4_t[i] {
                            matches += 1;
                        }
                    }
                    let top5 = if comp_len > 0 { matches as f32 / comp_len as f32 } else { 0.0 };
                    (t1, top5)
                };

                // Distinct-1
                let distinct_1 = if out.tokens.is_empty() {
                    0.0
                } else {
                    let set: std::collections::HashSet<_> = out.tokens.iter().collect();
                    set.len() as f32 / out.tokens.len() as f32
                };
                let is_degenerate = distinct_1 < 0.45 || out.stop_reason == "loop";

                prompt_results.push(PromptResult {
                    id: id.to_string(),
                    prompt_text: prompt.to_string(),
                    prompt_tokens: p_tokens,
                    tokenization_roundtrip_ok: true,
                    generated_tokens: out.tokens,
                    generated_text: text,
                    stop_reason: out.stop_reason,
                    token_1_agreement_with_q4_0: t1_agree,
                    top_5_agreement: top5_agree,
                    distinct_1,
                    mean_top1_prob: out.mean_top1_prob,
                    mean_entropy: out.mean_entropy,
                    is_degenerate,
                    generation_time_ms: out.time_ms,
                    determinism_ok,
                });
            }

            // Ejecución de Prompts Largos (200 tokens con captura de hidden states)
            println!("   ▶ Ejecutando Par Bilingüe Largo (200 tokens)...");
            let mut long_results = Vec::new();

            let long_cases = [
                (long_en.0, long_en.1, &long_en.2),
                (long_zh.0, long_zh.1, &long_zh.2),
            ];

            for (l_id, l_prompt, keywords) in &long_cases {
                let p_tokens = build_chatml_tokens(&tokenizer, l_prompt);
                let out = generate_detailed(&mut model, &p_tokens, 200, 1.10, &eos_ids, true);

                let u32_tok: Vec<u32> = out.tokens.iter().map(|&t| t as u32).collect();
                let text = tokenizer.decode(&u32_tok);

                // Determinismo
                let out2 = generate_detailed(&mut model, &p_tokens, 200, 1.10, &eos_ids, false);
                let determinism_ok = out.tokens == out2.tokens;

                // Salvar referencias de Q4_0
                let (sc_ref_str, q4_hidden) = if *cfg_name == "q4_0_control" {
                    if *l_id == "LONG-EN" {
                        q4_long_en_output = Some(GenOutput {
                            tokens: out.tokens.clone(),
                            stop_reason: out.stop_reason.clone(),
                            mean_top1_prob: out.mean_top1_prob,
                            mean_entropy: out.mean_entropy,
                            time_ms: out.time_ms,
                            hidden_states: out.hidden_states.clone(),
                        });
                        ("q4_0_control", &out.hidden_states)
                    } else {
                        q4_long_zh_output = Some(GenOutput {
                            tokens: out.tokens.clone(),
                            stop_reason: out.stop_reason.clone(),
                            mean_top1_prob: out.mean_top1_prob,
                            mean_entropy: out.mean_entropy,
                            time_ms: out.time_ms,
                            hidden_states: out.hidden_states.clone(),
                        });
                        ("q4_0_control", &out.hidden_states)
                    }
                } else if *l_id == "LONG-EN" {
                    ("q4_0_control", &q4_long_en_output.as_ref().unwrap().hidden_states)
                } else {
                    ("q4_0_control", &q4_long_zh_output.as_ref().unwrap().hidden_states)
                };

                // Muestreo granular en [1, 5, 10, 25, 50, 100, 150, 200]
                let milestone_positions = [1, 5, 10, 25, 50, 100, 150, 200];
                let mut milestones = Vec::new();
                let mut sc_l10_vals = Vec::new();
                let mut sc_l23_vals = Vec::new();

                for &m_pos in &milestone_positions {
                    let idx = m_pos - 1;
                    if idx < out.hidden_states.len() && idx < q4_hidden.len() {
                        let sc10 = cos_sim(&q4_hidden[idx].0, &out.hidden_states[idx].0);
                        let sc23 = cos_sim(&q4_hidden[idx].1, &out.hidden_states[idx].1);
                        sc_l10_vals.push(sc10);
                        sc_l23_vals.push(sc23);
                        milestones.push(LongPromptMilestone {
                            token_pos: m_pos,
                            sc_body_l10: sc10,
                            sc_exit_l23: sc23,
                        });
                    }
                }

                let mean_sc10 = if sc_l10_vals.is_empty() { 0.0 } else { sc_l10_vals.iter().sum::<f32>() / sc_l10_vals.len() as f32 };
                let mean_sc23 = if sc_l23_vals.is_empty() { 0.0 } else { sc_l23_vals.iter().sum::<f32>() / sc_l23_vals.len() as f32 };

                // Distinct-1 en los últimos 50 tokens
                let last_50 = if out.tokens.len() > 50 {
                    &out.tokens[out.tokens.len() - 50..]
                } else {
                    &out.tokens[..]
                };
                let distinct_1_last_50 = if last_50.is_empty() {
                    0.0
                } else {
                    let s: std::collections::HashSet<_> = last_50.iter().collect();
                    s.len() as f32 / last_50.len() as f32
                };
                let is_deg_last_50 = distinct_1_last_50 < 0.45 || out.stop_reason == "loop";

                // Foco Semántico (conteo de keywords en el texto de los últimos 50 tokens)
                let last_50_u32: Vec<u32> = last_50.iter().map(|&t| t as u32).collect();
                let last_50_text = tokenizer.decode(&last_50_u32);

                let mut hits = 0;
                for &kw in *keywords {
                    if last_50_text.contains(kw) {
                        hits += 1;
                    }
                }
                let semantic_focus_ok = hits >= 3;

                long_results.push(LongPromptResult {
                    id: l_id.to_string(),
                    prompt_text: l_prompt.to_string(),
                    tokenization_roundtrip_ok: true,
                    generated_tokens: out.tokens,
                    generated_text: text,
                    stop_reason: out.stop_reason,
                    sc_reference: sc_ref_str.to_string(),
                    sc_candidate: cfg_name.to_string(),
                    milestones,
                    mean_sc_l10: mean_sc10,
                    mean_sc_l23: mean_sc23,
                    distinct_1_last_50,
                    is_degenerate_last_50: is_deg_last_50,
                    semantic_focus_hits: hits,
                    semantic_focus_ok,
                    generation_time_ms: out.time_ms,
                    determinism_ok,
                });
            }

            reports.push(ConfigReport {
                config: cfg_name.to_string(),
                sc_reference: "q4_0_control".to_string(),
                sc_candidate: cfg_name.to_string(),
                penalty: 1.10,
                temperature: 0.0,
                short_prompts: prompt_results,
                long_prompts: long_results,
            });
        }

        // GUARDAR REPORTE JSON
        let json_str = serde_json::to_string_pretty(&reports).unwrap();
        let out_path = "docs/research/bilingual_sustained_generation_matrix.json";
        std::fs::write(out_path, &json_str).expect("Write JSON matrix");
        println!("\n💾 Reporte JSON guardado en: {}", out_path);

        // =========================================================================
        // RESUMEN CONSOLIDADO EN TERMINAL
        // =========================================================================
        println!("\n==================================================================================");
        println!("📊 RESUMEN 1: PROMPTS CORTOS (AGREEMENT Y DEGENERACIÓN VS Q4_0)");
        println!("==================================================================================");
        println!("{:<25} | {:<12} | {:<12} | {:<12} | {:<12}", "Configuración", "Token1 Agree", "Top5 Agree", "Distinct-1", "Degeneradas");
        println!("{}", "-".repeat(85));
        for rep in &reports {
            let t1_count = rep.short_prompts.iter().filter(|p| p.token_1_agreement_with_q4_0).count();
            let top5_avg = rep.short_prompts.iter().map(|p| p.top_5_agreement).sum::<f32>() / rep.short_prompts.len() as f32;
            let dist1_avg = rep.short_prompts.iter().map(|p| p.distinct_1).sum::<f32>() / rep.short_prompts.len() as f32;
            let deg_count = rep.short_prompts.iter().filter(|p| p.is_degenerate).count();
            println!("{:<25} | {:<12} | {:<12.4} | {:<12.4} | {:<12}",
                rep.config,
                format!("{}/{}", t1_count, rep.short_prompts.len()),
                top5_avg,
                dist1_avg,
                format!("{}/{}", deg_count, rep.short_prompts.len())
            );
        }

        println!("\n==================================================================================");
        println!("📊 RESUMEN 2: DERIVA GRANULAR EN PROMPTS LARGOS (200 TOKENS)");
        println!("==================================================================================");
        println!("Configuración: Var A (Attn Q4 + FFN 3-bit) vs Q4_0 Reference");
        let var_a_rep = &reports[3];
        for l_res in &var_a_rep.long_prompts {
            println!("\n  [{}] {} (Foco Semántico Hits: {}/3 -> {})", l_res.id, l_res.stop_reason, l_res.semantic_focus_hits, l_res.semantic_focus_ok);
            print!("    Posición : ");
            for m in &l_res.milestones { print!("{:>7} ", m.token_pos); }
            println!();
            print!("    Sc L10   : ");
            for m in &l_res.milestones { print!("{:>7.4} ", m.sc_body_l10); }
            println!();
            print!("    Sc L23   : ");
            for m in &l_res.milestones { print!("{:>7.4} ", m.sc_exit_l23); }
            println!();
        }

        println!("\n==================================================================================");
        println!("📊 RESUMEN 3: VEREDICTO FORMAL DE ARBITRAJE");
        println!("==================================================================================");
        let var_a = &reports[3];
        let t1_ratio = var_a.short_prompts.iter().filter(|p| p.token_1_agreement_with_q4_0).count() as f32 / var_a.short_prompts.len() as f32;
        let deg_short = var_a.short_prompts.iter().any(|p| p.is_degenerate);
        let deg_long = var_a.long_prompts.iter().any(|p| p.is_degenerate_last_50);
        let det_ok = var_a.short_prompts.iter().all(|p| p.determinism_ok) && var_a.long_prompts.iter().all(|p| p.determinism_ok);
        let focus_ok = var_a.long_prompts.iter().all(|p| p.semantic_focus_ok);
        let sc_ok = var_a.long_prompts.iter().all(|p| p.milestones.iter().all(|m| m.sc_body_l10 >= 0.85 && m.sc_exit_l23 >= 0.85));

        println!("• Token-1 Agreement (Cortos)  : {:.1}% (Meta: >= 80%)", t1_ratio * 100.0);
        println!("• Determinismo (T=0.0)         : {}", if det_ok { "✅ CUMPLE (100% Determinista)" } else { "❌ FALLO" });
        println!("• Foco Semántico en Cola      : {}", if focus_ok { "✅ CUMPLE (Keywords presentes)" } else { "❌ FALLO" });
        println!("• Ausencia de Degeneración     : {}", if !deg_short && !deg_long { "✅ CUMPLE (Cero bucles)" } else { "❌ FALLO" });
        println!("• Estabilidad Angular S_c      : {}", if sc_ok { "✅ CUMPLE (S_c >= 0.85 sostenido)" } else { "❌ FALLO" });

        if t1_ratio >= 0.80 && det_ok && focus_ok && !deg_short && !deg_long && sc_ok {
            println!("\n🏆 VEREDICTO FINAL: APROBADO TOTAL -> IMPLEMENTAR Q3_0Block.");
        } else if t1_ratio >= 0.60 && !deg_short && !deg_long {
            println!("\n⚠️ VEREDICTO FINAL: REVISIÓN -> Requiere ajuste fino de penalización antes de implementar.");
        } else {
            println!("\n❌ VEREDICTO FINAL: RECHAZADO -> Q4_0 es el piso real.");
        }
        println!("==================================================================================\n");
    }
}
