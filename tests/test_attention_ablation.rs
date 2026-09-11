#[cfg(test)]
mod tests {
    use _impl::io::flat_reader::GajeFlatFileReader;

    const PROMPT: &str = "Demuestra por que la raiz cuadrada de 2 es un numero irracional usando el metodo de reduccion al absurdo paso a paso. Comienza asumiendo que existen dos enteros coprimos p y q tales que su cociente al cuadrado es 2. Luego desarrolla la paridad de p y q hasta llegar a una contradiccion logica formal. Explica detalladamente por que si p al cuadrado es par entonces p debe ser par, y como esto implica que q tambien debe ser par contradiciendo la premisa coprima inicial.";

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

    #[inline(always)]
    fn median(slice: &[f32]) -> f32 {
        if slice.is_empty() {
            return 0.0;
        }
        let mut sorted = slice.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    struct ModelRun {
        name: String,
        // [token_idx][layer_idx]
        layer_sc: Vec<Vec<f32>>,
    }

    #[test]
    fn test_attention_ablation_matrix() {
        let path4 = "models/production/qwen2_5_0_5b.gaje";
        let path2 = "models/qwen2_5_0_5b_q2_0.flat";

        println!("\n==================================================================================");
        println!("🧬 GAJE HELIX — MATRIZ DE ABLACIÓN ASIMÉTRICA: ATENCIÓN (V1, V2) Y CAPAS TERMINALES");
        println!("==================================================================================");

        let reader4 = GajeFlatFileReader::open(path4).expect("Open Q4");
        let reader2 = GajeFlatFileReader::open(path2).expect("Open Q2");

        let tokenizer = reader4.get_embedded_gtok().expect("Embedded GTOK");
        let tokens_u32 = tokenizer.encode(PROMPT);
        let token_ids: Vec<usize> = tokens_u32.into_iter().map(|t| t as usize).collect();
        let n_tokens = token_ids.len();

        println!("📝 Corpus de prueba: {} tokens autoregresivos continuos.", n_tokens);

        // 1. Cargar Control Q4 y guardar sus activaciones por token y capa
        println!("\n[1/7] Ejecutando Control Q4_0 (Base de Verdad)...");
        let mut m_ctrl = reader4.load_genomic().expect("Load Q4");
        m_ctrl.clear_cache_core();

        let mut ctrl_activations: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_tokens);
        for (pos, &tid) in token_ids.iter().enumerate() {
            let mut h = m_ctrl.get_token_embedding(tid).unwrap();
            let mut layers_h = Vec::with_capacity(24);
            for l in 0..24 {
                h = m_ctrl.blocks[l].forward_core(h, pos).unwrap();
                layers_h.push(h.clone());
            }
            ctrl_activations.push(layers_h);
        }

        // Definición de las variantes a evaluar:
        // Todas parten de una copia de Q2_0 y reciben trasplantes quirúrgicos desde m_ctrl
        let variants: Vec<(&str, Box<dyn Fn(&mut _impl::nn::llm::GenomicLLM, &_impl::nn::llm::GenomicLLM)>)> = vec![
            (
                "Pure Q2_0",
                Box::new(|_cand, _ctrl| {
                    // Nada que modificar, puro Q2
                }),
            ),
            (
                "V1: Wq+Wk Q4 (Rest Q2)",
                Box::new(|cand, ctrl| {
                    for l in 0..24 {
                        cand.blocks[l].q_gen = ctrl.blocks[l].q_gen.clone();
                        cand.blocks[l].k_gen = ctrl.blocks[l].k_gen.clone();
                    }
                }),
            ),
            (
                "V2: Attn Q4 (Wq,k,v,o) (FFN Q2)",
                Box::new(|cand, ctrl| {
                    for l in 0..24 {
                        cand.blocks[l].q_gen = ctrl.blocks[l].q_gen.clone();
                        cand.blocks[l].k_gen = ctrl.blocks[l].k_gen.clone();
                        cand.blocks[l].v_gen = ctrl.blocks[l].v_gen.clone();
                        cand.blocks[l].w_o = ctrl.blocks[l].w_o.clone();
                    }
                }),
            ),
            (
                "Terminal Only (0-20 Q2, 21-23 Q4)",
                Box::new(|cand, ctrl| {
                    for l in 21..24 {
                        cand.blocks[l] = ctrl.blocks[l].clone();
                    }
                }),
            ),
            (
                "V1 + Terminal (0-20 V1, 21-23 Q4)",
                Box::new(|cand, ctrl| {
                    for l in 0..21 {
                        cand.blocks[l].q_gen = ctrl.blocks[l].q_gen.clone();
                        cand.blocks[l].k_gen = ctrl.blocks[l].k_gen.clone();
                    }
                    for l in 21..24 {
                        cand.blocks[l] = ctrl.blocks[l].clone();
                    }
                }),
            ),
            (
                "V2 + Terminal (0-20 V2, 21-23 Q4)",
                Box::new(|cand, ctrl| {
                    for l in 0..21 {
                        cand.blocks[l].q_gen = ctrl.blocks[l].q_gen.clone();
                        cand.blocks[l].k_gen = ctrl.blocks[l].k_gen.clone();
                        cand.blocks[l].v_gen = ctrl.blocks[l].v_gen.clone();
                        cand.blocks[l].w_o = ctrl.blocks[l].w_o.clone();
                    }
                    for l in 21..24 {
                        cand.blocks[l] = ctrl.blocks[l].clone();
                    }
                }),
            ),
        ];

        let mut results: Vec<ModelRun> = Vec::new();

        for (var_idx, (name, setup_fn)) in variants.iter().enumerate() {
            println!("\n[{}/7] Evaluando variante: {} ...", var_idx + 2, name);
            let mut cand = reader2.load_genomic().expect("Load Q2 candidate");
            setup_fn(&mut cand, &m_ctrl);
            cand.clear_cache_core();

            let mut layer_sc: Vec<Vec<f32>> = vec![Vec::with_capacity(n_tokens); 24];

            for (pos, &tid) in token_ids.iter().enumerate() {
                let mut h = cand.get_token_embedding(tid).unwrap();
                for l in 0..24 {
                    h = cand.blocks[l].forward_core(h, pos).unwrap();
                    let sc = cos_sim(&ctrl_activations[pos][l], &h);
                    layer_sc[l].push(sc);
                }
            }

            results.push(ModelRun {
                name: name.to_string(),
                layer_sc,
            });
        }

        // =========================================================================
        // REPORTE DE RESULTADOS
        // =========================================================================

        let key_layers = [2, 6, 10, 16, 20, 21, 22, 23];
        let windows: &[(&str, std::ops::Range<usize>)] = &[
            ("Pos 0 (i.i.d.)", 0..1),
            ("Pos 1..4 (Onset)", 1..5.min(n_tokens)),
            ("Pos 5..19 (Short)", 5.min(n_tokens)..20.min(n_tokens)),
            ("Pos 20..49 (Mid)", 20.min(n_tokens)..50.min(n_tokens)),
            ("Pos 50+ (Deep)", 50.min(n_tokens)..n_tokens),
        ];

        println!("\n==================================================================================");
        println!("📊 RESUMEN 1: SIMILARIDAD COSENO EN CUERPO (CAPA 10) POR VENTANA TEMPORAL");
        println!("==================================================================================");
        print!("{:<32}", "Variante");
        for (w_name, _) in windows {
            print!(" | {:<16}", w_name);
        }
        println!(" | {:<10}", "Global Med");
        println!("{}", "-".repeat(120));

        for r in &results {
            print!("{:<32}", r.name);
            let l10 = &r.layer_sc[10];
            for (_, range) in windows {
                if range.start < range.end {
                    let sub = &l10[range.clone()];
                    let m = median(sub);
                    print!(" | {:<16.4}", m);
                } else {
                    print!(" | {:<16}", "N/A");
                }
            }
            println!(" | {:<10.4}", median(l10));
        }

        println!("\n==================================================================================");
        println!("📊 RESUMEN 2: SIMILARIDAD COSENO EN SALIDA FINAL (CAPA 23) POR VENTANA TEMPORAL");
        println!("==================================================================================");
        print!("{:<32}", "Variante");
        for (w_name, _) in windows {
            print!(" | {:<16}", w_name);
        }
        println!(" | {:<10}", "Global Med");
        println!("{}", "-".repeat(120));

        for r in &results {
            print!("{:<32}", r.name);
            let l23 = &r.layer_sc[23];
            for (_, range) in windows {
                if range.start < range.end {
                    let sub = &l23[range.clone()];
                    let m = median(sub);
                    print!(" | {:<16.4}", m);
                } else {
                    print!(" | {:<16}", "N/A");
                }
            }
            println!(" | {:<10.4}", median(l23));
        }

        println!("\n==================================================================================");
        println!("📊 RESUMEN 3: PERFIL DETALLADO POR CAPA (MEDIANA EN CONTEXTO POS >= 5)");
        println!("==================================================================================");
        print!("{:<32}", "Variante");
        for &l in &key_layers {
            print!(" | L{:<4}", l);
        }
        println!();
        println!("{}", "-".repeat(110));

        for r in &results {
            print!("{:<32}", r.name);
            for &l in &key_layers {
                let sub = if n_tokens > 5 {
                    &r.layer_sc[l][5..n_tokens]
                } else {
                    &r.layer_sc[l][..]
                };
                print!(" | {:<6.4}", median(sub));
            }
            println!();
        }

        println!("\n==================================================================================");
        println!("📊 RESUMEN 4: OVERHEAD DE PARÁMETROS Y BIT-DEPTH EFECTIVO ESTIMADO");
        println!("==================================================================================");
        println!("• Pure Q2_0:                        2.000 bits/weight  (+0.0% overhead)");
        println!("• V1 (Wq, Wk en Q4_0):              2.124 bits/weight  (+6.2% overhead en atención)");
        println!("• V2 (Attn Wq,k,v,o en Q4_0):       2.246 bits/weight  (+12.3% overhead en atención)");
        println!("• Terminal Only (Capas 21-23 Q4):   2.250 bits/weight  (+12.5% overhead en 3 bloques)");
        println!("• V1 + Terminal:                    2.344 bits/weight  (+17.2% overhead)");
        println!("• V2 + Terminal:                    2.441 bits/weight  (+22.0% overhead)");
        println!("==================================================================================\n");
    }
}
