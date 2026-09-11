#[cfg(test)]
mod tests {
    use _impl::io::flat_reader::GajeFlatFileReader;

    #[test]
    fn test_compare_extraction_methods() {
        let path4 = "models/production/qwen2_5_0_5b.gaje";
        let path2 = "models/qwen2_5_0_5b_q2_0.flat";

        let reader4 = GajeFlatFileReader::open(path4).unwrap();
        let reader2 = GajeFlatFileReader::open(path2).unwrap();

        let mut m4_seq = reader4.load_genomic().unwrap();
        let mut m2_seq = reader2.load_genomic().unwrap();

        let token_ids = vec![1912, 15128, 288, 2172, 3193, 81, 2804, 95459, 1187, 15122];

        // Metodo 1: extract_sequence_activations (usado en extract_activations.py)
        let acts4 = m4_seq.extract_sequence_activations(&token_ids, true).unwrap();
        let acts2 = m2_seq.extract_sequence_activations(&token_ids, true).unwrap();

        let n = |v: &[f32]| -> f32 { v.iter().map(|x| x * x).sum::<f32>().sqrt() };
        let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };

        println!("\n=== METODO 1: extract_sequence_activations ===");
        for t in 0..token_ids.len() {
            let c = &acts4[t][10];
            let q = &acts2[t][10];
            let cos = dot(c, q) / (n(c) * n(q));
            println!("Token {:2} (Layer 10): cos = {:.4}", t, cos);
        }

        // Metodo 2: El loop de diagnose_cancellation.rs
        let mut m4_loop = reader4.load_genomic().unwrap();
        let mut m2_loop = reader2.load_genomic().unwrap();
        m4_loop.clear_cache_core();
        m2_loop.clear_cache_core();

        println!("\n=== METODO 2: Loop manual en diagnose_cancellation.rs ===");
        for (t, &tid) in token_ids.iter().enumerate() {
            let pos = m4_loop.blocks[0].attn.k_cache_len();
            let mut h4 = m4_loop.get_token_embedding(tid).unwrap();
            let mut h2 = m2_loop.get_token_embedding(tid).unwrap();

            for l in 0..24 {
                h4 = m4_loop.blocks[l].forward_core(h4, pos).unwrap();
                h2 = m2_loop.blocks[l].forward_core(h2, pos).unwrap();
                if l == 10 {
                    let cos = dot(&h4, &h2) / (n(&h4) * n(&h2));
                    println!("Token {:2} (Layer 10): cos = {:.4}", t, cos);
                }
            }
        }
    }
}
