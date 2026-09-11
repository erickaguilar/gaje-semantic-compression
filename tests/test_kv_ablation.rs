#[cfg(test)]
mod tests {
    use _impl::io::flat_reader::GajeFlatFileReader;

    const PROMPT: &str = "Demuestra por que la raiz cuadrada de 2 es un numero irracional usando el metodo de reduccion al absurdo paso a paso. Comienza asumiendo que existen dos enteros coprimos p y q tales que su cociente al cuadrado es 2. Luego desarrolla la paridad de p y q hasta llegar a una contradiccion logica formal.";

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

    #[test]
    fn test_kv_and_sandwich_ablation() {
        let path4 = "models/production/qwen2_5_0_5b.gaje";
        let path2 = "models/qwen2_5_0_5b_q2_0.flat";

        let reader4 = GajeFlatFileReader::open(path4).unwrap();
        let reader2 = GajeFlatFileReader::open(path2).unwrap();

        let tokenizer = reader4.get_embedded_gtok().unwrap();
        let tokens_u32 = tokenizer.encode(PROMPT);
        let token_ids: Vec<usize> = tokens_u32.into_iter().map(|t| t as usize).collect();
        let n_tokens = token_ids.len();

        println!("\n=== ABLACIÓN DE KV-CACHE Y TOPOLOGÍA SÁNDWICH (Prompt = {} tokens) ===", n_tokens);

        // Modelos a contrastar:
        // 1. Control Q4
        let mut m_q4 = reader4.load_genomic().unwrap();
        // 2. Q2 Puro
        let mut m_q2 = reader2.load_genomic().unwrap();
        // 3. B1: Entrada Q4 (Capas 0-1 en Q4, 2-23 en Q2)
        let mut m_b1 = reader2.load_genomic().unwrap();
        m_b1.blocks[0] = m_q4.blocks[0].clone();
        m_b1.blocks[1] = m_q4.blocks[1].clone();
        // 4. B2: Salida Q4 (Capas 0-20 en Q2, 21-23 en Q4)
        let mut m_b2 = reader2.load_genomic().unwrap();
        m_b2.blocks[21] = m_q4.blocks[21].clone();
        m_b2.blocks[22] = m_q4.blocks[22].clone();
        m_b2.blocks[23] = m_q4.blocks[23].clone();
        // 5. Sándwich 4-2-4 (Capas 0-1 en Q4, 2-20 en Q2, 21-23 en Q4)
        let mut m_sandwich = reader2.load_genomic().unwrap();
        m_sandwich.blocks[0] = m_q4.blocks[0].clone();
        m_sandwich.blocks[1] = m_q4.blocks[1].clone();
        m_sandwich.blocks[21] = m_q4.blocks[21].clone();
        m_sandwich.blocks[22] = m_q4.blocks[22].clone();
        m_sandwich.blocks[23] = m_q4.blocks[23].clone();

        m_q4.clear_cache_core();
        m_q2.clear_cache_core();
        m_b1.clear_cache_core();
        m_b2.clear_cache_core();
        m_sandwich.clear_cache_core();

        // Almacenamos por posición: Sc en Capa 10 (cuerpo) y Capa 23 (salida)
        let mut sc_q2_c10 = Vec::new();
        let mut sc_q2_c23 = Vec::new();

        let mut sc_b1_c10 = Vec::new();
        let mut sc_b1_c23 = Vec::new();

        let mut sc_b2_c10 = Vec::new();
        let mut sc_b2_c23 = Vec::new();

        let mut sc_sw_c10 = Vec::new();
        let mut sc_sw_c23 = Vec::new();

        for (pos, &tid) in token_ids.iter().enumerate() {
            let mut h4 = m_q4.get_token_embedding(tid).unwrap();
            let mut h2 = m_q2.get_token_embedding(tid).unwrap();
            let mut hb1 = m_b1.get_token_embedding(tid).unwrap();
            let mut hb2 = m_b2.get_token_embedding(tid).unwrap();
            let mut hsw = m_sandwich.get_token_embedding(tid).unwrap();

            let mut h4_c10 = Vec::new();
            let mut h2_c10 = Vec::new();
            let mut hb1_c10 = Vec::new();
            let mut hb2_c10 = Vec::new();
            let mut hsw_c10 = Vec::new();

            for l in 0..24 {
                h4 = m_q4.blocks[l].forward_core(h4, pos).unwrap();
                h2 = m_q2.blocks[l].forward_core(h2, pos).unwrap();
                hb1 = m_b1.blocks[l].forward_core(hb1, pos).unwrap();
                hb2 = m_b2.blocks[l].forward_core(hb2, pos).unwrap();
                hsw = m_sandwich.blocks[l].forward_core(hsw, pos).unwrap();

                if l == 10 {
                    h4_c10 = h4.clone();
                    h2_c10 = h2.clone();
                    hb1_c10 = hb1.clone();
                    hb2_c10 = hb2.clone();
                    hsw_c10 = hsw.clone();
                }
            }

            sc_q2_c10.push(cos_sim(&h4_c10, &h2_c10));
            sc_q2_c23.push(cos_sim(&h4, &h2));

            sc_b1_c10.push(cos_sim(&h4_c10, &hb1_c10));
            sc_b1_c23.push(cos_sim(&h4, &hb1));

            sc_b2_c10.push(cos_sim(&h4_c10, &hb2_c10));
            sc_b2_c23.push(cos_sim(&h4, &hb2));

            sc_sw_c10.push(cos_sim(&h4_c10, &hsw_c10));
            sc_sw_c23.push(cos_sim(&h4, &hsw));
        }

        let mean = |slice: &[f32]| -> f32 { slice.iter().sum::<f32>() / slice.len().max(1) as f32 };

        println!("\n=================================================================================================================");
        println!("RESULTADOS POR VENTANA DE POSICIÓN DE TOKEN EN CONTEXTO AUTOREGRESIVO");
        println!("=================================================================================================================");
        println!("Rango Tokens  | Q2 Puro (C10 / C23) | B1: Ent Q4 (C10 / C23) | B2: Sal Q4 (C10 / C23) | Sándwich 4-2-4 (C10 / C23)");
        println!("-----------------------------------------------------------------------------------------------------------------");

        let windows = [(0, 1), (1, 5), (5, 20), (20, 50), (50, n_tokens)];
        for (start, end) in windows {
            let e = end.min(n_tokens);
            if start >= e { continue; }
            let len = e - start;
            println!("Tokens {:2}..{:2}  |  {:.4} / {:.4}   |   {:.4} / {:.4}    |   {:.4} / {:.4}    |   {:.4} / {:.4}",
                start, e - 1,
                mean(&sc_q2_c10[start..e]), mean(&sc_q2_c23[start..e]),
                mean(&sc_b1_c10[start..e]), mean(&sc_b1_c23[start..e]),
                mean(&sc_b2_c10[start..e]), mean(&sc_b2_c23[start..e]),
                mean(&sc_sw_c10[start..e]), mean(&sc_sw_c23[start..e]),
            );
        }

        println!("-----------------------------------------------------------------------------------------------------------------");
        println!("PROMEDIO TOTAL|  {:.4} / {:.4}   |   {:.4} / {:.4}    |   {:.4} / {:.4}    |   {:.4} / {:.4}",
            mean(&sc_q2_c10), mean(&sc_q2_c23),
            mean(&sc_b1_c10), mean(&sc_b1_c23),
            mean(&sc_b2_c10), mean(&sc_b2_c23),
            mean(&sc_sw_c10), mean(&sc_sw_c23),
        );
        println!("=================================================================================================================\n");
    }
}
