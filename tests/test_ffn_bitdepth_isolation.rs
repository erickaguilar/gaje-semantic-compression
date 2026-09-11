#[cfg(test)]
mod tests {
    use _impl::io::flat_reader::GajeFlatFileReader;
    use _impl::io::header::Q4_0Block;
    use _impl::nn::linear::database::WeightDatabase;
    use _impl::nn::linear::GenomicLinear;
    use half::f16;
    use std::sync::Arc;

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

    /// Cuantiza una capa lineal Q4_0 a 3 bits reales por bloque de 32 (8 niveles equiespaciados con escala + min local)
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

    struct ModelRun {
        name: String,
        layer_sc: Vec<Vec<f32>>,
    }

    #[test]
    fn test_ffn_bitdepth_and_isolation_matrix() {
        let path4 = "models/production/qwen2_5_0_5b.gaje";
        let path2 = "models/qwen2_5_0_5b_q2_0.flat";

        println!("\n==================================================================================");
        println!("🧬 GAJE HELIX — MATRIZ DE AISLAMIENTO: FFN 3-BIT VS ATENCIÓN VS FFN Q4");
        println!("==================================================================================");

        let reader4 = GajeFlatFileReader::open(path4).expect("Open Q4");
        let reader2 = GajeFlatFileReader::open(path2).expect("Open Q2");

        let tokenizer = reader4.get_embedded_gtok().expect("Embedded GTOK");
        let tokens_u32 = tokenizer.encode(PROMPT);
        let token_ids: Vec<usize> = tokens_u32.into_iter().map(|t| t as usize).collect();
        let n_tokens = token_ids.len();

        println!("📝 Corpus de prueba: {} tokens autoregresivos continuos.", n_tokens);

        // 1. Control Q4_0 (Base de Verdad Techo)
        println!("\n[1/5] Ejecutando Control C: Q4_0 Puro (Techo de Verdad)...");
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

        // Variantes a contrastar:
        // 1. Pure Q2_0 (Piso inferior)
        // 2. Variante B: Attn Q4 + FFN Q2 (Baseline V2)
        // 3. Variante D: Attn Q2 + FFN Q4 (Aislamiento: ¿Es la atención Q2 o el FFN el culpable?)
        // 4. Variante A: Attn Q4 + FFN 3-bit (Test de umbral de producto)
        let variants: Vec<(&str, Box<dyn Fn(&mut _impl::nn::llm::GenomicLLM, &_impl::nn::llm::GenomicLLM)>)> = vec![
            (
                "Pure Q2_0 (Piso)",
                Box::new(|_cand, _ctrl| {
                    // Q2 puro
                }),
            ),
            (
                "Var B: Attn Q4 + FFN Q2 (V2)",
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
                "Var D: Attn Q2 + FFN Q4 (Aislamiento)",
                Box::new(|cand, ctrl| {
                    // Conserva Attn en Q2, pero pone FFN en Q4
                    for l in 0..24 {
                        cand.blocks[l].gate_gen = ctrl.blocks[l].gate_gen.clone();
                        cand.blocks[l].up_gen = ctrl.blocks[l].up_gen.clone();
                        cand.blocks[l].w_down = ctrl.blocks[l].w_down.clone();
                    }
                }),
            ),
            (
                "Var A: Attn Q4 + FFN 3-bit (Umbral)",
                Box::new(|cand, ctrl| {
                    for l in 0..24 {
                        // Atención en Q4
                        cand.blocks[l].q_gen = ctrl.blocks[l].q_gen.clone();
                        cand.blocks[l].k_gen = ctrl.blocks[l].k_gen.clone();
                        cand.blocks[l].v_gen = ctrl.blocks[l].v_gen.clone();
                        cand.blocks[l].w_o = ctrl.blocks[l].w_o.clone();

                        // FFN cuantizado a 3-bit (8 niveles por bloque de 32)
                        cand.blocks[l].gate_gen = quantize_linear_to_3bit(&ctrl.blocks[l].gate_gen);
                        cand.blocks[l].up_gen = quantize_linear_to_3bit(&ctrl.blocks[l].up_gen);
                        cand.blocks[l].w_down = quantize_linear_to_3bit(&ctrl.blocks[l].w_down);
                    }
                }),
            ),
        ];

        let mut results: Vec<ModelRun> = Vec::new();

        for (var_idx, (name, setup_fn)) in variants.iter().enumerate() {
            println!("\n[{}/5] Evaluando: {} ...", var_idx + 2, name);
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
        // REPORTE DE RESULTADOS Y VEREDICTO DE UMBRAL
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
        print!("{:<38}", "Configuración");
        for (w_name, _) in windows {
            print!(" | {:<16}", w_name);
        }
        println!(" | {:<10}", "Global Med");
        println!("{}", "-".repeat(130));

        for r in &results {
            print!("{:<38}", r.name);
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
        print!("{:<38}", "Configuración");
        for (w_name, _) in windows {
            print!(" | {:<16}", w_name);
        }
        println!(" | {:<10}", "Global Med");
        println!("{}", "-".repeat(130));

        for r in &results {
            print!("{:<38}", r.name);
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
        println!("📊 RESUMEN 3: PERFIL COMPLETO POR CAPA (MEDIANA EN CONTEXTO POS >= 5)");
        println!("==================================================================================");
        print!("{:<38}", "Configuración");
        for &l in &key_layers {
            print!(" | L{:<4}", l);
        }
        println!();
        println!("{}", "-".repeat(110));

        for r in &results {
            print!("{:<38}", r.name);
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
        println!("📊 RESUMEN 4: CONCLUSIÓN DE AISLAMIENTO Y VIABILIDAD DE 3-BIT");
        println!("==================================================================================");
        let sc_med_a_l10 = median(&results[3].layer_sc[10]);
        let sc_med_d_l10 = median(&results[2].layer_sc[10]);
        println!("• Var D (Attn Q2 + FFN Q4): Mediana L10 = {:.4}", sc_med_d_l10);
        println!("• Var A (Attn Q4 + FFN 3b): Mediana L10 = {:.4}", sc_med_a_l10);
        if sc_med_a_l10 >= 0.85 {
            println!("✅ RESULTADO: FFN a 3-bit es VIABLE (Sc >= 0.85). Justifica el desarrollo de formato Q3_0.");
        } else if sc_med_a_l10 >= 0.60 {
            println!("⚠️ RESULTADO: FFN a 3-bit en zona gris (0.60 <= Sc < 0.85). Podría requerir Q3_K no uniforme.");
        } else {
            println!("❌ RESULTADO: FFN a 3-bit INSUFICIENTE (Sc < 0.60). Demuestra empíricamente que Q4_0 es el piso real.");
        }
        println!("==================================================================================\n");
    }
}
