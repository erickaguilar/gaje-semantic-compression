#[cfg(test)]
mod tests {
    use _impl::io::flat_reader::GajeFlatFileReader;

    #[test]
    fn test_cross_substitution_layer21() {
        let path4 = "models/production/qwen2_5_0_5b.gaje";
        let path2 = "models/qwen2_5_0_5b_q2_0.flat";

        let reader4 = GajeFlatFileReader::open(path4).expect("Open Q4");
        let reader2 = GajeFlatFileReader::open(path2).expect("Open Q2");

        let mut m4 = reader4.load_genomic().expect("Load Q4");
        let mut m2 = reader2.load_genomic().expect("Load Q2");

        // Evaluamos sobre 10 tokens distintos para asegurar que no sea un artefacto de token
        let test_tokens = [100, 500, 1000, 1500, 2048, 5000, 10000, 25000, 50000, 100000];

        let n = |v: &[f32]| -> f32 { v.iter().map(|x| x * x).sum::<f32>().sqrt() };
        let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
        let cos_sim = |a: &[f32], b: &[f32]| -> f32 {
            let na = n(a);
            let nb = n(b);
            if na > 1e-12 && nb > 1e-12 { dot(a, b) / (na * nb) } else { 0.0 }
        };

        println!("\n=== TEST DE SUSTITUCIÓN CRUZADA EN CAPA 21 (N=10 TOKENS) ===");
        println!("Token | cos(h_in) | cos(h4,d4) | cos(h2,d2) | cos(h_out Q4/Q2) | cos(h_in_Q2 + d_Q4, h_out_Q4) | cos(h_in_Q4 + d_Q2, h_out_Q4)");
        println!("---------------------------------------------------------------------------------------------------------------------------------");

        for &token_id in &test_tokens {
            m4.clear_cache_core();
            m2.clear_cache_core();

            let mut h4 = m4.get_token_embedding(token_id).unwrap();
            let mut h2 = m2.get_token_embedding(token_id).unwrap();

            for i in 0..21 {
                h4 = m4.blocks[i].forward_core(h4, 0).unwrap();
                h2 = m2.blocks[i].forward_core(h2, 0).unwrap();
            }

            let h4_out = m4.blocks[21].forward_core(h4.clone(), 0).unwrap();
            let h2_out = m2.blocks[21].forward_core(h2.clone(), 0).unwrap();

            let d4: Vec<f32> = h4_out.iter().zip(&h4).map(|(o, i)| o - i).collect();
            let d2: Vec<f32> = h2_out.iter().zip(&h2).map(|(o, i)| o - i).collect();

            // Sustitución 1: h_in_Q2 con Delta_Q4 (re-escalada a la norma de h2 o cruda)
            // Caso A: Delta_Q4 re-escalada por ratio ||h2|| / ||h4||
            let scale_factor = n(&h2) / n(&h4);
            let d4_scaled: Vec<f32> = d4.iter().map(|&x| x * scale_factor).collect();
            let hybrid_hin2_d4: Vec<f32> = h2.iter().zip(&d4_scaled).map(|(i, d)| i + d).collect();

            // Sustitución 2: h_in_Q4 con Delta_Q2 (re-escalada por ratio ||h4|| / ||h2||)
            let scale_factor_inv = n(&h4) / n(&h2);
            let d2_scaled: Vec<f32> = d2.iter().map(|&x| x * scale_factor_inv).collect();
            let hybrid_hin4_d2: Vec<f32> = h4.iter().zip(&d2_scaled).map(|(i, d)| i + d).collect();

            let c_hin = cos_sim(&h4, &h2);
            let c_d4 = cos_sim(&h4, &d4);
            let c_d2 = cos_sim(&h2, &d2);
            let c_orig = cos_sim(&h4_out, &h2_out);
            let c_hyb1 = cos_sim(&hybrid_hin2_d4, &h4_out);
            let c_hyb2 = cos_sim(&hybrid_hin4_d2, &h4_out);

            println!("{:5} | {:9.4} | {:10.4} | {:10.4} | {:16.4} | {:29.4} | {:29.4}",
                token_id, c_hin, c_d4, c_d2, c_orig, c_hyb1, c_hyb2);
        }
    }
}
