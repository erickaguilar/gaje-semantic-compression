#[cfg(test)]
mod tests {
    use _impl::io::flat_reader::GajeFlatFileReader;

    #[test]
    fn test_tokenizer_chatml_probe() {
        let path4 = "models/production/qwen2_5_0_5b.gaje";
        let reader4 = GajeFlatFileReader::open(path4).expect("Open Q4");
        let tokenizer = reader4.get_embedded_gtok().expect("Embedded GTOK");

        let prompt_raw = "La capital de Francia es";
        let prompt_chatml = "<|im_start|>user\nLa capital de Francia es<|im_end|>\n<|im_start|>assistant\n";

        let tokens_raw = tokenizer.encode(prompt_raw);
        let tokens_chatml = tokenizer.encode(prompt_chatml);

        println!("\n=== TOKENIZER VERIFICATION PROBE ===");
        println!("Prompt Raw: '{}'", prompt_raw);
        println!("Token IDs Raw: {:?}", tokens_raw);
        for &t in &tokens_raw {
            println!("  ID {}: '{}'", t, tokenizer.decode(&[t]));
        }

        println!("\nPrompt ChatML: '{}'", prompt_chatml);
        println!("Token IDs ChatML: {:?}", tokens_chatml);
        for &t in &tokens_chatml {
            println!("  ID {}: '{}'", t, tokenizer.decode(&[t]));
        }

        // Check special tokens
        let im_start_id = tokenizer.encode("<|im_start|>");
        let im_end_id = tokenizer.encode("<|im_end|>");
        println!("\nSpecial token '<|im_start|>': {:?}", im_start_id);
        println!("Special token '<|im_end|>': {:?}", im_end_id);
        let mut m_ctrl = reader4.load_genomic().expect("Load Q4");

        // Caso 3: Inglés canónico
        let mut en_tokens: Vec<usize> = vec![151644, 872, 198];
        en_tokens.extend(tokenizer.encode("What is the capital of France?").into_iter().map(|t| t as usize));
        en_tokens.extend(vec![151645, 198, 151644, 77091, 198]);

        println!("\n--- GENERACIÓN EN INGLÉS (Q4_0) ---");
        println!("Prompt: 'What is the capital of France?'");
        let gen3 = m_ctrl.generate_native_core(en_tokens, 20, 0.0, 1.1, vec![151645, 151643]).unwrap();
        let gen3_u32: Vec<u32> = gen3.iter().map(|&t| t as u32).collect();
        println!("Output: '{}'", tokenizer.decode(&gen3_u32));
    }

    #[test]
    fn test_qwen_1_5b_logits_probe() {
        let path = "models/production/qwen2_5_1_5b.gaje";
        if !std::path::Path::new(path).exists() {
            println!("Model not found: {}", path);
            return;
        }
        let reader = GajeFlatFileReader::open(path).expect("Open 1.5B");
        let tokenizer = reader.get_embedded_gtok().expect("Embedded GTOK");
        let mut model = reader.load_genomic().expect("Load 1.5B");

        println!("\n=== 1.5B FORWARD PASS DIAGNOSTIC PROBE ===");
        println!("Model dim: {}", model.dim());
        println!("Blocks: {}", model.blocks.len());
        println!("Head dim: {}", model.blocks[0].attn.head_dim);
        println!("N head: {}", model.blocks[0].attn.n_head);
        println!("N head kv: {}", model.blocks[0].attn.n_head_kv);
        println!("RoPE base: {}", model.blocks[0].attn.rope_base);
        println!("RoPE style: {}", model.blocks[0].attn.rope_style);

        let token_id = 9707; // "Hello"
        println!("\nTesting single token forward: ID {} ('Hello')", token_id);

        let emb = model.get_token_embedding(token_id).expect("get_token_embedding");
        let nan_emb = emb.iter().filter(|v| v.is_nan()).count();
        println!("Embedding: len={}, NaNs={}, first 5={:?}", emb.len(), nan_emb, &emb[..5]);

        let mut h = emb;
        for (i, block) in model.blocks.iter_mut().enumerate() {
            match block.forward_core(h.clone(), 0) {
                Ok(next_h) => {
                    let nan_h = next_h.iter().filter(|v| v.is_nan()).count();
                    let inf_h = next_h.iter().filter(|v| v.is_infinite()).count();
                    if nan_h > 0 || inf_h > 0 || i < 3 || i == 27 {
                        println!("Layer {:2}: NaNs={}, Infs={}, min={:.3}, max={:.3}, first 3={:?}",
                            i, nan_h, inf_h,
                            next_h.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                            next_h.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                            &next_h[..3]
                        );
                    }
                    if nan_h > 0 {
                        println!("🚨 NaN FIRST DETECTED AT LAYER {}!", i);
                        break;
                    }
                    h = next_h;
                }
                Err(e) => {
                    println!("🚨 Layer {} returned Err: {}", i, e);
                    break;
                }
            }
        }

        let h_norm = unsafe { _impl::compute::kernels::rms_norm(&h, &model.output_norm, model.eps) };
        println!("\nh_norm: len={}, min={:.3}, max={:.3}, first 5={:?}",
            h_norm.len(),
            h_norm.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            h_norm.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
            &h_norm[..5]
        );

        let n_blocks = model.lm_head.in_features / model.lm_head.block_size;
        println!("LM Head: in={}, out={}, block_size={}, stride={}, n_blocks={}",
            model.lm_head.in_features, model.lm_head.out_features, model.lm_head.block_size, model.lm_head.stride, n_blocks
        );
        println!("LM Head DB bytes: {}, centroids len: {}", model.lm_head.database_ref().len(), model.lm_head.centroids.len());

        let row0 = model.lm_head.compute_single_row(0, &h_norm, &[1.0; 4], n_blocks);
        let row100 = model.lm_head.compute_single_row(100, &h_norm, &[1.0; 4], n_blocks);
        println!("Direct from disk: compute_single_row(0): {:.6}", row0);
        println!("Direct from disk: compute_single_row(100): {:.6}", row100);

        let logits = model.forward_core(token_id, true).expect("forward_core");
        let nan_logits = logits.iter().filter(|v| v.is_nan()).count();
        let inf_logits = logits.iter().filter(|v| v.is_infinite()).count();
        let min_l = logits.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        println!("\nLogits: len={}, NaNs={}, Infs={}, min={:.3}, max={:.3}", logits.len(), nan_logits, inf_logits, min_l, max_l);
        println!("Logits [0..5]: {:?}", &logits[..5]);

        let max_idx = logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).map(|(idx, _)| idx);
        println!("Argmax token ID: {:?}", max_idx);
        if let Some(idx) = max_idx {
            println!("Decoded token: '{}'", tokenizer.decode(&[idx as u32]));
        }

        let mut en_tokens: Vec<usize> = vec![151644, 872, 198];
        en_tokens.extend(tokenizer.encode("Hello").into_iter().map(|t| t as usize));
        en_tokens.extend(vec![151645, 198, 151644, 77091, 198]);
        let gen = model.generate_native_core(en_tokens, 10, 0.0, 1.1, vec![151645, 151643]).unwrap();
        let gen_u32: Vec<u32> = gen.iter().map(|&t| t as u32).collect();
        println!("Generated output for 'Hello': '{}'", tokenizer.decode(&gen_u32));
    }

    #[test]
    fn test_qwen_1_5b_typo_probe() {
        let path = "models/production/qwen2_5_1_5b.gaje";
        let reader = GajeFlatFileReader::open(path).expect("Open 1.5B");
        let tokenizer = reader.get_embedded_gtok().expect("Embedded GTOK");
        let mut model = reader.load_genomic().expect("Load 1.5B");
        let stop_tokens = vec![151645, 151643];

        println!("\n==================================================================");
        println!("🧪 PASO 1: TEST DE PROMPT RESTRINGIDO ('Answer with only the name/number')");
        println!("==================================================================");

        let restricted_cases = vec![
            (13, "Ganymede", "<|im_start|>system\nKnowledge: Ganymede is the largest moon of Jupiter.<|im_end|>\n<|im_start|>user\nWhat is the largest moon of Jupiter? Answer with only the name.<|im_end|>\n<|im_start|>assistant\n"),
            (14, "Venus", "<|im_start|>system\nKnowledge: Venus is the second planet from the Sun.<|im_end|>\n<|im_start|>user\nWhat is the second planet from the Sun? Answer with only the name.<|im_end|>\n<|im_start|>assistant\n"),
            (15, "Venus", "<|im_start|>system\nKnowledge: Venus is the planet commonly known as the Morning Star.<|im_end|>\n<|im_start|>user\nWhich planet is known as the Morning Star? Answer with only the name.<|im_end|>\n<|im_start|>assistant\n"),
            (19, "118", "<|im_start|>system\nKnowledge: There are 118 confirmed elements in the periodic table.<|im_end|>\n<|im_start|>user\nHow many elements are in the periodic table? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n"),
        ];

        for (id, target, prompt) in &restricted_cases {
            let input_ids: Vec<usize> = tokenizer.encode(prompt).into_iter().map(|t| t as usize).collect();
            let output_ids = model.generate_native_core(input_ids, 15, 0.0, 1.15, stop_tokens.clone()).unwrap();
            let output_u32: Vec<u32> = output_ids.iter().map(|&t| t as u32).collect();
            let response = tokenizer.decode(&output_u32);
            println!("[ID {:02}] Target: {:10} | Raw Output: '{}'", id, target, response.trim().replace('\n', " "));
        }

        println!("\n==================================================================");
        println!("🔍 PASO 2: LOGIT RANKING PROBES EN STEP 0 (TRAS 'Answer: ')");
        println!("==================================================================");

        let logit_cases = vec![
            (19, "ID 19 (Periodic Table Elements)", "<|im_start|>system\nKnowledge: There are 118 confirmed elements in the periodic table.<|im_end|>\n<|im_start|>user\nHow many elements are in the periodic table?<|im_end|>\n<|im_start|>assistant\nAnswer: ", vec!["118", " 118", "108", " 108", "There", " There"]),
            (13, "ID 13 (Largest moon of Jupiter)", "<|im_start|>system\nKnowledge: Ganymede is the largest moon of Jupiter.<|im_end|>\n<|im_start|>user\nWhat is the largest moon of Jupiter?<|im_end|>\n<|im_start|>assistant\nAnswer: ", vec!["Ganymede", " Ganymede", "Ganymiel", " Ganymiel", "The", " The"]),
            (14, "ID 14 (Second planet from Sun)", "<|im_start|>system\nKnowledge: Venus is the second planet from the Sun.<|im_end|>\n<|im_start|>user\nWhat is the second planet from the Sun?<|im_end|>\n<|im_start|>assistant\nAnswer: ", vec!["Venus", " Venus", "Venuis", " Venuis", "The", " The"]),
        ];

        for (_id, label, prompt, probe_words) in &logit_cases {
            println!("\n--- {} ---", label);
            let input_ids: Vec<usize> = tokenizer.encode(prompt).into_iter().map(|t| t as usize).collect();

            model.clear_cache_core();
            for i in 0..input_ids.len() - 1 {
                model.forward_blocks_only(input_ids[i]).unwrap();
            }
            let logits = model.forward_core(input_ids[input_ids.len() - 1], false).unwrap();

            let mut indexed_logits: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
            indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            println!("Top 15 tokens at Step 0:");
            for i in 0..15 {
                let (tid, logit) = indexed_logits[i];
                let decoded = tokenizer.decode(&[tid as u32]);
                println!("  #{:2}: ID={:6} (logit={:7.3}) -> '{}'", i + 1, tid, logit, decoded);
            }

            println!("Probing specific tokens:");
            for word in probe_words {
                let tokens = tokenizer.encode(word);
                for &t in &tokens {
                    let logit = logits[t as usize];
                    let rank = indexed_logits.iter().position(|&(idx, _)| idx == t as usize).map(|r| r + 1).unwrap_or(99999);
                    let dec = tokenizer.decode(&[t]);
                    println!("  Word '{}' -> Token ID {} ('{}') | Logit: {:7.3} | Rank: #{}", word, t, dec, logit, rank);
                }
            }
        }
    }

    fn print_vm(label: &str) {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let vm_data = status.lines().find(|l| l.starts_with("VmData:")).unwrap_or("");
        let vm_rss = status.lines().find(|l| l.starts_with("VmRSS:")).unwrap_or("");
        println!("[{}] {} | {}", label, vm_data.trim(), vm_rss.trim());
    }

    #[test]
    fn test_qwen_1_5b_rss_baseline() {
        let path = "models/production/qwen2_5_1_5b.gaje";
        if !std::path::Path::new(path).exists() {
            println!("Model not found: {}", path);
            return;
        }
        print_vm("START");
        let reader = GajeFlatFileReader::open(path).expect("Open 1.5B");
        print_vm("AFTER OPEN (MMAP)");

        println!("Number of tensors in tensor_map: {}", reader.tensor_map.len());
        if let Some(entry) = reader.tensor_map.get("token_embd") {
            println!("token_embd: bit_depth={}, dna_len={}, anc_len={}, c_len={}, bias_len={}",
                entry.bit_depth, entry.dna_len, entry.anc_len, entry.c_len, entry.bias_len);
        }
        if let Some(entry) = reader.tensor_map.get("lm_head") {
            println!("lm_head: bit_depth={}, dna_off={}, dna_len={}, anc_len={}, c_len={}, bias_len={}",
                entry.bit_depth, entry.dna_off, entry.dna_len, entry.anc_len, entry.c_len, entry.bias_len);
        }
        if let Some(entry) = reader.tensor_map.get("token_embd") {
            println!("token_embd: dna_off={}", entry.dna_off);
        }

        let model = reader.load_genomic().expect("Load 1.5B");
        print_vm("AFTER LOAD_GENOMIC");

        println!("\n=== BASELINE RSS MEASUREMENT (1.5B LOADED) ===");
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in status.lines() {
            if line.starts_with("VmRSS:") || line.starts_with("VmHWM:") || line.starts_with("VmData:") || line.starts_with("VmSize:") {
                println!("{}", line);
            }
        }
        println!("Model loaded successfully: {} transformer blocks", model.blocks.len());
    }

    #[test]
    fn test_qwen_1_5b_bpe_fragmentation_benchmark() {
        let path = "models/production/qwen2_5_1_5b.gaje";
        let reader = GajeFlatFileReader::open(path).expect("Open 1.5B");
        let tokenizer = reader.get_embedded_gtok().expect("Embedded GTOK");
        let mut model = reader.load_genomic().expect("Load 1.5B");
        let stop_tokens = vec![151645, 151643];

        println!("\n==================================================================");
        println!("🧬 BENCHMARK DE ENTIDADES BPE FRAGMENTADAS (10 CASOS) — QWEN 2.5 1.5B");
        println!("Objetivo: Determinar si el typo de 'Ganymede' es un caso aislado o un patrón sistemático");
        println!("==================================================================\n");

        let benchmark_cases = vec![
            (1, "Ganymede", "Ganymede is the largest moon of Jupiter.", "What is the largest moon of Jupiter?"),
            (2, "Enceladus", "Enceladus is the ice-covered moon of Saturn with water geysers.", "Which moon of Saturn has water geysers?"),
            (3, "Betelgeuse", "Betelgeuse is a red supergiant star located in the constellation Orion.", "What red supergiant star is located in the constellation Orion?"),
            (4, "Oumuamua", "Oumuamua was the first known interstellar object to visit our solar system.", "What was the first known interstellar object to visit our solar system?"),
            (5, "Deoxyribose", "Deoxyribose is the five-carbon sugar molecule found in DNA nucleotides.", "What five-carbon sugar molecule is found in DNA nucleotides?"),
            (6, "Mitochondria", "Mitochondria are the organelles known as the cellular powerhouses.", "What organelles are known as the cellular powerhouses?"),
            (7, "Phenolphthalein", "Phenolphthalein is a chemical compound commonly used as an acid-base pH indicator.", "What chemical compound is commonly used as an acid-base pH indicator?"),
            (8, "Quetzalcoatl", "Quetzalcoatl was the famous feathered serpent deity in Aztec mythology.", "Who was the feathered serpent deity in Aztec mythology?"),
            (9, "Tegucigalpa", "Tegucigalpa is the capital city of Honduras.", "What is the capital city of Honduras?"),
            (10, "Rhinoceros", "The rhinoceros is a large herbivorous mammal known for its keratin horns.", "What large herbivorous mammal is known for its keratin horns?"),
        ];

        let mut extract_hits = 0;
        let mut synth_hits = 0;

        for (id, entity, fact, question) in &benchmark_cases {
            // Inspección BPE de la entidad
            let entity_tokens = tokenizer.encode(entity);
            let token_strs: Vec<String> = entity_tokens.iter().map(|&t| format!("'{}' (#{})", tokenizer.decode(&[t]), t)).collect();
            println!("\n[{:02}/10] ENTIDAD: \"{}\" | Subpalabras BPE ({} tokens): [{}]", id, entity, entity_tokens.len(), token_strs.join(", "));
            println!("  Hecho: \"{}\"", fact);

            // 1. MODO EXTRACCIÓN ("Answer with only the name.")
            let extract_prompt = format!(
                "<|im_start|>system\nKnowledge: {}<|im_end|>\n<|im_start|>user\n{} Answer with only the name.<|im_end|>\n<|im_start|>assistant\n",
                fact, question
            );
            let extract_ids: Vec<usize> = tokenizer.encode(&extract_prompt).into_iter().map(|t| t as usize).collect();
            let extract_out = model.generate_native_core(extract_ids, 12, 0.0, 1.15, stop_tokens.clone()).unwrap();
            let extract_u32: Vec<u32> = extract_out.iter().map(|&t| t as u32).collect();
            let extract_text = tokenizer.decode(&extract_u32);
            let extract_clean = extract_text.trim().replace('\n', " ");
            let is_extract_hit = extract_clean.to_lowercase().contains(&entity.to_lowercase());
            if is_extract_hit { extract_hits += 1; }
            let ext_emoji = if is_extract_hit { "✅ HIT " } else { "❌ MISS" };
            println!("  [Extracción] {} -> \"{}\"", ext_emoji, extract_clean);

            // 2. MODO SÍNTESIS ("Answer: ")
            let synth_prompt = format!(
                "<|im_start|>system\nKnowledge: {}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\nAnswer: ",
                fact, question
            );
            let synth_ids: Vec<usize> = tokenizer.encode(&synth_prompt).into_iter().map(|t| t as usize).collect();
            let synth_out = model.generate_native_core(synth_ids, 20, 0.0, 1.15, stop_tokens.clone()).unwrap();
            let synth_u32: Vec<u32> = synth_out.iter().map(|&t| t as u32).collect();
            let synth_text = tokenizer.decode(&synth_u32);
            let synth_clean = synth_text.trim().replace('\n', " ");
            let is_synth_hit = synth_clean.to_lowercase().contains(&entity.to_lowercase());
            if is_synth_hit { synth_hits += 1; }
            let syn_emoji = if is_synth_hit { "✅ HIT " } else { "❌ MISS" };
            println!("  [Síntesis]   {} -> \"{}\"", syn_emoji, synth_clean);
        }

        println!("\n==================================================================");
        println!("📊 RESUMEN FINAL DEL BENCHMARK DE BPE FRAGMENTADO (10 CASOS)");
        println!("==================================================================");
        println!("Modo Extracción Concisa: {} / 10 ({:.1}%)", extract_hits, (extract_hits as f32 / 10.0) * 100.0);
        println!("Modo Síntesis Abierta:   {} / 10 ({:.1}%)", synth_hits, (synth_hits as f32 / 10.0) * 100.0);
        println!("==================================================================\n");
    }

    #[test]
    fn test_qwen_1_5b_q8head_sanity_check() {
        let path = "models/production/qwen2_5_1_5b.gaje";
        if !std::path::Path::new(path).exists() {
            println!("Model not found: {}", path);
            return;
        }

        print_vm("START");
        let t0 = std::time::Instant::now();
        let reader = GajeFlatFileReader::open(path).expect("Open 1.5B Q8Head");
        print_vm("AFTER MMAP OPEN");

        let tokenizer = reader.get_embedded_gtok().expect("Embedded GTOK");
        let mut model = reader.load_genomic().expect("Load 1.5B Q8Head");
        let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
        print_vm("AFTER LOAD_GENOMIC");

        println!("\n=== SANITY CHECK 1.5B HÍBRIDO (LM_HEAD Q8_0) ===");
        println!("Load Time: {:.2} ms", load_ms);
        println!("lm_head bit_depth: {}", model.lm_head.bit_depth());
        println!("lm_head centroids count: {}", model.lm_head.centroids.len());

        let stop_tokens = vec![151645, 151643];
        let sanity_cases = vec![
            (14, "Venus", "Venus is the second planet from the Sun.", "What is the second planet from the Sun?"),
            (15, "Venus", "Venus is the planet commonly known as the Morning Star.", "Which planet is known as the Morning Star?"),
            (19, "118", "There are 118 confirmed elements in the periodic table.", "How many elements are in the periodic table?"),
            (13, "Ganymede", "Ganymede is the largest moon of Jupiter.", "What is the largest moon of Jupiter?"),
        ];

        println!("\n--- MODO EXTRACCIÓN CONCISA ---");
        for (id, target, fact, question) in &sanity_cases {
            let prompt = format!(
                "<|im_start|>system\nKnowledge: {}<|im_end|>\n<|im_start|>user\n{} Answer with only the name or number.<|im_end|>\n<|im_start|>assistant\n",
                fact, question
            );
            let input_ids: Vec<usize> = tokenizer.encode(&prompt).into_iter().map(|t| t as usize).collect();
            let out_ids = model.generate_native_core(input_ids, 15, 0.0, 1.15, stop_tokens.clone()).unwrap();
            let u32_ids: Vec<u32> = out_ids.iter().map(|&t| t as u32).collect();
            let response = tokenizer.decode(&u32_ids);
            let clean = response.trim().replace('\n', " ");
            let hit = clean.to_lowercase().contains(&target.to_lowercase());
            let tag = if hit { "✅ HIT " } else { "❌ MISS" };
            println!("[ID {:02}] Target: {:10} | {} | Output: '{}'", id, target, tag, clean);
        }

        println!("\n--- MODO SÍNTESIS ABIERTA ('Answer: ') ---");
        for (id, target, fact, question) in &sanity_cases {
            let prompt = format!(
                "<|im_start|>system\nKnowledge: {}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\nAnswer: ",
                fact, question
            );
            let input_ids: Vec<usize> = tokenizer.encode(&prompt).into_iter().map(|t| t as usize).collect();
            let out_ids = model.generate_native_core(input_ids, 20, 0.0, 1.15, stop_tokens.clone()).unwrap();
            let u32_ids: Vec<u32> = out_ids.iter().map(|&t| t as u32).collect();
            let response = tokenizer.decode(&u32_ids);
            let clean = response.trim().replace('\n', " ");
            let hit = clean.to_lowercase().contains(&target.to_lowercase());
            let tag = if hit { "✅ HIT " } else { "❌ MISS" };
            println!("[ID {:02}] Target: {:10} | {} | Output: '{}'", id, target, tag, clean);
        }
    }
}


