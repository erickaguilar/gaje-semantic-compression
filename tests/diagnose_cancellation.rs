#[cfg(test)]
mod tests {
    use _impl::io::flat_reader::GajeFlatFileReader;
    use std::fs::File;
    use std::io::Write;

    const PROMPTS: &[&str] = &[
        "Demuestra por que la raiz cuadrada de 2 es un numero irracional usando el metodo de reduccion al absurdo paso a paso. Comienza asumiendo que existen dos enteros coprimos p y q tales que su cociente al cuadrado es 2.",
        "Explica el principio de accion minima de Hamilton en mecanica clasica y como se relaciona con las ecuaciones de Euler-Lagrange para una particula en un potencial armonico unidimensional.",
        "Implement a fast lock-free circular queue in C++20 using std::atomic with acquire-release memory orderings and explain how cacheline contention is avoided with hardware padding.",
        "Describe la funcion de la ADN polimerasa delta y epsilon en la replicacion celular de eucariotas, enfatizando la correccion de pruebas 3 prima a 5 prima y la sintesis coordinada de la hebra rezagada.",
        "Analiza el experimento mental de la habitacion china de John Searle y evalua si la objecion del sistema o la del robot refutan validamente la ausencia de intencionalidad intrinseca en modelos sintacticos.",
        "Detalla la microarquitectura de un procesador superescalar moderno: prediccion de saltos TAGE, estaciones de reserva, ejecucion fuera de orden (OoO), ejecucion especulativa y buffer de reordenamiento (ROB).",
        "Calcula la complejidad computacional en FLOPS y memoria en bytes de la atencion FlashAttention-2 comparada con la atencion estandar O(N^2) con KV-cache paginado para secuencias largas.",
        "Sintetiza las tensiones sociopoliticas entre Atenas y Esparta que desembocaron en la Guerra del Peloponeso segun la cronica de Tucidides, enfocandote en la dinamica de la trampa de Tucidides.",
        "Explica la aproximacion de Born-Oppenheimer en la ecuacion de Schrodinger molecular y describe como los estados electronicos adiabaticos se desacoplan del movimiento nuclear por la relacion de masas.",
        "Explica como el consenso Raft previene la particion de cerebro dividido (split-brain) durante la eleccion de lider cuando existe una particion de red asimetrica entre tres nodos.",
        "Define formalmente el equilibrio de Nash perfecto en subjuegos para un juego dinamico finito y proporciona un ejemplo con un juego de ultimatum con dos rondas de negociacion secuencial.",
        "Describe el ciclo de transduccion de senales del receptor acoplado a proteina G (GPCR) activado por adrenalina, desde la activacion de adenilato ciclasa hasta la fosforilacion de PKA.",
        "Explica la construccion de intercambio de claves Diffie-Hellman en curvas elípticas (ECDH) sobre Curve25519 y como mitigar ataques de intermediario usando firmas digitales Ed25519.",
        "Describe los procesos geodinamicos que ocurren en una zona de subduccion tipo Mariana versus una tipo chilena, incluyendo el flujo de fluidos, serpentinizacion del manto y magmatismo de arco.",
        "Como permite la cuantizacion vectorial cuaternaria en 2-bits (A, C, G, T) preservar las representaciones semanticas en un espacio metrico proyectivo sin perder la simetria de fase compleja?",
        "Analiza la evolucion del concepto de entropia desde la termodinamica clasica de Clausius hasta la mecanica estadistica de Boltzmann y la teoria de informacion de Shannon.",
    ];

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
    fn angle_rad(cos_val: f32) -> f32 {
        cos_val.clamp(-1.0, 1.0).acos()
    }

    #[inline(always)]
    fn percentile(sorted: &[f32], p: f32) -> f32 {
        if sorted.is_empty() { return 0.0; }
        let idx = ((sorted.len() - 1) as f32 * p).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    #[test]
    fn test_diagnose_cancellation_and_surgery() {
        let path4 = "models/production/qwen2_5_0_5b.gaje";
        let path2 = "models/qwen2_5_0_5b_q2_0.flat";

        println!("\n==================================================================================");
        println!("🧬 GAJE HELIX — Diagnostico Global de Cancelacion y Cirugia de Capas");
        println!("==================================================================================");

        let reader4 = GajeFlatFileReader::open(path4).expect("Open Q4");
        let reader2 = GajeFlatFileReader::open(path2).expect("Open Q2");

        let tokenizer = reader4.get_embedded_gtok().expect("GTOK Tokenizer");

        let mut m_ctrl = reader4.load_genomic().expect("Load Q4");
        let mut m_q2 = reader2.load_genomic().expect("Load Q2");

        // Preparar modelos quirúrgicos en memoria
        let mut m_surg21 = reader2.load_genomic().expect("Load Surgery 21");
        m_surg21.blocks[21] = m_ctrl.blocks[21].clone();

        let mut m_surg21_23 = reader2.load_genomic().expect("Load Surgery 21-23");
        m_surg21_23.blocks[21] = m_ctrl.blocks[21].clone();
        m_surg21_23.blocks[22] = m_ctrl.blocks[22].clone();
        m_surg21_23.blocks[23] = m_ctrl.blocks[23].clone();

        println!("✅ Modelos preparados en memoria:");
        println!("   • m_ctrl:        Control Puro (Q4_0, 24 capas)");
        println!("   • m_q2:          Candidato Puro (Q2_0, 24 capas)");
        println!("   • m_surg21:      Cirugia Capa 21 (Q2_0 con Capa 21 en Q4_0)");
        println!("   • m_surg21_23:   Cirugia Capas 21-23 (Q2_0 con Capas 21, 22, 23 en Q4_0)");

        let target_tokens = 500; // Muestra estadistica robusta (>30x grados de libertad)
        println!("\n⏳ Procesando corpus multi-prompt (Meta: {} tokens)...", target_tokens);

        let n_layers = 24;

        // Metricas recolectadas por capa: [layer][token]
        let mut cos_hin_d4 = vec![Vec::new(); n_layers];
        let mut cos_hin_d2 = vec![Vec::new(); n_layers];
        let mut theta_d4 = vec![Vec::new(); n_layers];
        let mut theta_d2 = vec![Vec::new(); n_layers];
        let mut ratio_norm_d4 = vec![Vec::new(); n_layers];
        let mut ratio_norm_d2 = vec![Vec::new(); n_layers];
        let mut cos_hout_q4_q2 = vec![Vec::new(); n_layers];

        // Metricas cirugia por capa (21, 22, 23):
        let mut cos_surg21 = vec![Vec::new(); n_layers];
        let mut cos_surg21_23 = vec![Vec::new(); n_layers];

        let mut total_tokens = 0usize;
        let mut prompt_idx = 0usize;

        while total_tokens < target_tokens && prompt_idx < PROMPTS.len() {
            let text = PROMPTS[prompt_idx];
            let token_u32 = tokenizer.encode(text);
            let token_ids: Vec<usize> = token_u32.into_iter().map(|t| t as usize).collect();
            let n_tok = token_ids.len();

            m_ctrl.clear_cache_core();
            m_q2.clear_cache_core();
            m_surg21.clear_cache_core();
            m_surg21_23.clear_cache_core();

            for &tid in &token_ids {
                let pos = m_ctrl.blocks[0].attn.k_cache_len();

                let mut h4 = m_ctrl.get_token_embedding(tid).unwrap();
                let mut h2 = m_q2.get_token_embedding(tid).unwrap();
                let mut hs21 = m_surg21.get_token_embedding(tid).unwrap();
                let mut hs23 = m_surg21_23.get_token_embedding(tid).unwrap();

                for l in 0..n_layers {
                    let h4_in = h4.clone();
                    let h2_in = h2.clone();

                    h4 = m_ctrl.blocks[l].forward_core(h4, pos).unwrap();
                    h2 = m_q2.blocks[l].forward_core(h2, pos).unwrap();
                    hs21 = m_surg21.blocks[l].forward_core(hs21, pos).unwrap();
                    hs23 = m_surg21_23.blocks[l].forward_core(hs23, pos).unwrap();

                    let d4: Vec<f32> = h4.iter().zip(&h4_in).map(|(o, i)| o - i).collect();
                    let d2: Vec<f32> = h2.iter().zip(&h2_in).map(|(o, i)| o - i).collect();

                    let c_in_d4 = cos_sim(&h4_in, &d4);
                    let c_in_d2 = cos_sim(&h2_in, &d2);
                    let th4 = angle_rad(c_in_d4);
                    let th2 = angle_rad(c_in_d2);

                    let rn4 = norm(&d4) / norm(&h4_in).max(1e-12);
                    let rn2 = norm(&d2) / norm(&h2_in).max(1e-12);

                    let c_out = cos_sim(&h4, &h2);

                    cos_hin_d4[l].push(c_in_d4);
                    cos_hin_d2[l].push(c_in_d2);
                    theta_d4[l].push(th4);
                    theta_d2[l].push(th2);
                    ratio_norm_d4[l].push(rn4);
                    ratio_norm_d2[l].push(rn2);
                    cos_hout_q4_q2[l].push(c_out);

                    cos_surg21[l].push(cos_sim(&h4, &hs21));
                    cos_surg21_23[l].push(cos_sim(&h4, &hs23));
                }
            }

            total_tokens += n_tok;
            println!("   • Prompt {:02}: {:3} tokens | Total acumulado: {:4}/{}", prompt_idx + 1, n_tok, total_tokens, target_tokens);
            prompt_idx += 1;
        }

        let n_samples = cos_hout_q4_q2[0].len();
        println!("\n✅ Analisis completado sobre N = {} tokens i.i.d.\n", n_samples);

        // =========================================================================
        // TABLA 1: Perfil de Cancelacion a lo largo de las 24 Capas
        // =========================================================================
        println!("==================================================================================================");
        println!("TABLA 1: Perfil Global de Dinamica de Cancelacion por Capa (N = {})", n_samples);
        println!("==================================================================================================");
        println!("Capa | cos(h,Δ) Q4       | cos(h,Δ) Q2       | θ_Δ Q4 (deg) | θ_Δ Q2 (deg) | ||Δ||/||h|| Q4/Q2 | S_c(h_out Q4, Q2)");
        println!("--------------------------------------------------------------------------------------------------");

        for l in 0..n_layers {
            let mean_c4 = cos_hin_d4[l].iter().sum::<f32>() / n_samples as f32;
            let mean_c2 = cos_hin_d2[l].iter().sum::<f32>() / n_samples as f32;
            let mean_th4_deg = (theta_d4[l].iter().sum::<f32>() / n_samples as f32) * (180.0 / std::f32::consts::PI);
            let mean_th2_deg = (theta_d2[l].iter().sum::<f32>() / n_samples as f32) * (180.0 / std::f32::consts::PI);
            let mean_rn4 = ratio_norm_d4[l].iter().sum::<f32>() / n_samples as f32;
            let mean_rn2 = ratio_norm_d2[l].iter().sum::<f32>() / n_samples as f32;

            let mean_sc = cos_hout_q4_q2[l].iter().sum::<f32>() / n_samples as f32;
            let std_sc = (cos_hout_q4_q2[l].iter().map(|x| (x - mean_sc).powi(2)).sum::<f32>() / n_samples as f32).sqrt();

            let flag = if l == 2 {
                " <-- Bifurcacion de ganancia"
            } else if l == 21 {
                " <-- Cancelacion Portadora"
            } else {
                ""
            };

            println!("{:02}   | {:7.4}           | {:7.4}           | {:6.2}°       | {:6.2}°       | {:5.2} / {:5.2}     | {:6.4} ± {:6.4}{}",
                l, mean_c4, mean_c2, mean_th4_deg, mean_th2_deg, mean_rn4, mean_rn2, mean_sc, std_sc, flag);
        }

        // =========================================================================
        // TABLA 2: Distribucion Detallada en Capas Terminales (21, 22, 23)
        // =========================================================================
        println!("\n==================================================================================================");
        println!("TABLA 2: Percentiles y Tasa de Colapso en Capas Terminales (Q2_0 Puro)");
        println!("==================================================================================================");
        println!("Capa | Media  | Std    | Min     | p5      | p25     | Mediana | p75     | p95     | Max     | Tasa Sc<0.3 | Inversion (Sc<0)");
        println!("--------------------------------------------------------------------------------------------------");

        for &l in &[20, 21, 22, 23] {
            let mut vals = cos_hout_q4_q2[l].clone();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let mean = vals.iter().sum::<f32>() / vals.len() as f32;
            let std = (vals.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / vals.len() as f32).sqrt();
            let min = vals[0];
            let max = *vals.last().unwrap();
            let p5 = percentile(&vals, 0.05);
            let p25 = percentile(&vals, 0.25);
            let med = percentile(&vals, 0.50);
            let p75 = percentile(&vals, 0.75);
            let p95 = percentile(&vals, 0.95);

            let collapse_count = vals.iter().filter(|&&x| x < 0.30).count();
            let collapse_rate = (collapse_count as f32 / vals.len() as f32) * 100.0;

            let inv_count = vals.iter().filter(|&&x| x < 0.0).count();
            let inv_rate = (inv_count as f32 / vals.len() as f32) * 100.0;

            println!("{:02}   | {:6.4} | {:6.4} | {:7.4} | {:7.4} | {:7.4} | {:7.4} | {:7.4} | {:7.4} | {:7.4} | {:5.1}%      | {:5.1}%",
                l, mean, std, min, p5, p25, med, p75, p95, max, collapse_rate, inv_rate);
        }

        // =========================================================================
        // TABLA 3: Matriz de Correlacion Inter-Capa de Colapso
        // =========================================================================
        println!("\n==================================================================================================");
        println!("TABLA 3: Matriz de Correlacion Inter-Capa (Colapso y Similitud Coseno)");
        println!("==================================================================================================");

        let layers_corr = [21, 22, 23];
        println!("Matriz de Correlacion de Pearson r(Sc_i, Sc_j):");
        print!("     ");
        for &l in &layers_corr { print!("   Capa {:02}", l); }
        println!();

        let mean_scs: Vec<f32> = layers_corr.iter().map(|&l| cos_hout_q4_q2[l].iter().sum::<f32>() / n_samples as f32).collect();
        let std_scs: Vec<f32> = layers_corr.iter().enumerate().map(|(idx, &l)| {
            (cos_hout_q4_q2[l].iter().map(|x| (x - mean_scs[idx]).powi(2)).sum::<f32>() / n_samples as f32).sqrt()
        }).collect();

        for (i, &l1) in layers_corr.iter().enumerate() {
            print!("Capa {:02}:", l1);
            for (j, &l2) in layers_corr.iter().enumerate() {
                let cov: f32 = cos_hout_q4_q2[l1].iter().zip(&cos_hout_q4_q2[l2])
                    .map(|(&x, &y)| (x - mean_scs[i]) * (y - mean_scs[j]))
                    .sum::<f32>() / n_samples as f32;
                let r = cov / (std_scs[i] * std_scs[j]).max(1e-12);
                print!("     {:6.3}", r);
            }
            println!();
        }

        // =========================================================================
        // TABLA 4: Test de Cirugia — Recuperacion Medida vs Control
        // =========================================================================
        println!("\n==================================================================================================");
        println!("TABLA 4: Resultados de la Intervencion Quirurgica vs Control Q4_0");
        println!("==================================================================================================");
        println!("Capa | S_c Q2_0 Puro      | S_c Cirugia Capa 21  | S_c Cirugia Capas 21-23 | Ganancia Cirugia 21 vs 21-23");
        println!("--------------------------------------------------------------------------------------------------");

        for &l in &[20, 21, 22, 23] {
            let sc_q2_mean = cos_hout_q4_q2[l].iter().sum::<f32>() / n_samples as f32;
            let sc_q2_std = (cos_hout_q4_q2[l].iter().map(|x| (x - sc_q2_mean).powi(2)).sum::<f32>() / n_samples as f32).sqrt();

            let sc_s21_mean = cos_surg21[l].iter().sum::<f32>() / n_samples as f32;
            let sc_s21_std = (cos_surg21[l].iter().map(|x| (x - sc_s21_mean).powi(2)).sum::<f32>() / n_samples as f32).sqrt();

            let sc_s23_mean = cos_surg21_23[l].iter().sum::<f32>() / n_samples as f32;
            let sc_s23_std = (cos_surg21_23[l].iter().map(|x| (x - sc_s23_mean).powi(2)).sum::<f32>() / n_samples as f32).sqrt();

            println!("{:02}   | {:6.4} ± {:6.4}   | {:6.4} ± {:6.4}    | {:6.4} ± {:6.4}       | Δ_21 = {:+6.4} | Δ_23 = {:+6.4}",
                l, sc_q2_mean, sc_q2_std, sc_s21_mean, sc_s21_std, sc_s23_mean, sc_s23_std,
                sc_s21_mean - sc_q2_mean, sc_s23_mean - sc_q2_mean);
        }

        println!("==================================================================================================\n");
    }
}
