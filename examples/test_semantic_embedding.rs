use _impl::io::gmem::cosine_similarity;
use _impl::nn::repl::load_model_and_tokenizer;
use std::time::Instant;

fn main() {
    println!("================================================================");
    println!("🧪 TEST AISLADO DE embed_text (Mean Pooling de Token Embeddings)");
    println!("================================================================\n");

    let models_to_test = [
        ("models/born/max.gaje", "Q2_0 Born"),
        ("models/production/qwen2_5_0_5b.gaje", "Q4_0 Hybrid"),
        ("models/production/gaje_pico_135m.gaje", "Legacy FP32"),
    ];

    for (path, label) in &models_to_test {
        if !std::path::Path::new(path).exists() {
            println!("⚠️ Saltando {} (no existe)", path);
            continue;
        }

        println!("----------------------------------------------------------------");
        println!("📦 Evaluando Modelo: {} ({})", path, label);
        println!("----------------------------------------------------------------");

        let t0 = Instant::now();
        let (llm, tokenizer) = match load_model_and_tokenizer(path) {
            Ok(pair) => pair,
            Err(e) => {
                println!("❌ Error cargando {}: {}", path, e);
                continue;
            }
        };
        println!("  ⏱️ Modelo cargado en {:.2} ms (dim={})", t0.elapsed().as_secs_f64() * 1000.0, llm.dim());

        // 1. Verificación Sub-paso 2.2: get_token_embedding no contiene NaNs ni Infs
        let sample_token = 100usize;
        match llm.get_token_embedding(sample_token) {
            Ok(emb) => {
                let has_nan = emb.iter().any(|v| v.is_nan() || v.is_infinite());
                println!("  🔍 Verificación get_token_embedding(100): len={}, has_nan/inf={}", emb.len(), has_nan);
                assert!(!has_nan, "get_token_embedding no debe contener NaNs");
            }
            Err(e) => {
                println!("  ❌ get_token_embedding falló: {}", e);
            }
        }

        // 2. Verificación Sub-paso 2.1: embed_text sobre pares de texto
        let p1_fact = "La capital de Francia es París.";
        let p1_query = "¿Cuál es la capital de Francia?";

        let p2_fact = "El proyecto GAJE comprime redes neuronales mediante algoritmos genéticos.";
        let p2_query = "¿Cómo comprime GAJE los modelos de lenguaje?";

        let p_unrelated = "Receta para cocinar pastel de chocolate con fresas y crema dulce.";

        let t_emb0 = Instant::now();
        let v1_fact = llm.embed_text(p1_fact, &tokenizer).expect("embed v1_fact");
        let v1_query = llm.embed_text(p1_query, &tokenizer).expect("embed v1_query");
        let v2_fact = llm.embed_text(p2_fact, &tokenizer).expect("embed v2_fact");
        let v2_query = llm.embed_text(p2_query, &tokenizer).expect("embed v2_query");
        let v_unrel = llm.embed_text(p_unrelated, &tokenizer).expect("embed unrel");
        let emb_time_us = t_emb0.elapsed().as_micros() as f64 / 5.0;

        let sim_pair1 = cosine_similarity(&v1_fact, &v1_query);
        let sim_pair2 = cosine_similarity(&v2_fact, &v2_query);
        let sim_unrel1 = cosine_similarity(&v1_fact, &v_unrel);
        let sim_unrel2 = cosine_similarity(&v2_fact, &v_unrel);
        let sim_cross = cosine_similarity(&v1_fact, &v2_fact);

        println!("  ⚡ Latencia promedio embed_text (Mean Pooling): {:.2} µs ({:.4} ms)", emb_time_us, emb_time_us / 1000.0);
        println!("  📊 Similitudes Coseno Obtenidas (Mean Pooling puro):");
        println!("     • Par Relacionado 1 ('Francia/París'):          CosSim = {:.4}", sim_pair1);
        println!("     • Par Relacionado 2 ('GAJE/compresión'):        CosSim = {:.4}", sim_pair2);
        println!("     • Cruzado No Relacionado ('Francia' vs 'GAJE'): CosSim = {:.4}", sim_cross);
        println!("     • No Relacionado ('Francia' vs 'Pastel'):       CosSim = {:.4}", sim_unrel1);
        println!("     • No Relacionado ('GAJE' vs 'Pastel'):          CosSim = {:.4}", sim_unrel2);

        let gap1 = sim_pair1 - sim_unrel1;
        let gap2 = sim_pair2 - sim_unrel2;
        println!("  🎯 Separación Semántica (Delta Sim):");
        println!("     • Gap 1: +{:.4}", gap1);
        println!("     • Gap 2: +{:.4}\n", gap2);

        // 3. Test de Weighted Pooling (downweighting de palabras vacías)
        let is_stopword = |w: &str| -> bool {
            matches!(w, "de" | "la" | "el" | "en" | "es" | "y" | "a" | "un" | "una" | "unos" | "unas" | "con" | "por" | "para" | "que" | "del" | "los" | "las" | "the" | "is" | "of" | "and" | "in" | "to" | "," | "." | ";" | ":" | "¿" | "?" | "!" | "¡")
        };

        let embed_weighted = |text: &str| -> Vec<f32> {
            let tokens = tokenizer.encode(text, false).unwrap();
            let stops = tokenizer.get_stop_tokens();
            let dim = llm.dim();
            let mut sum_vec = vec![0.0f32; dim];
            let mut total_w = 0.0f32;
            for &t_u32 in &tokens {
                if stops.contains(&t_u32) { continue; }
                let t_str = tokenizer.decode(&[t_u32], true).unwrap_or_default().trim().to_lowercase();
                let w = if is_stopword(&t_str) { 0.15f32 } else { 1.0f32 };
                if let Ok(emb) = llm.get_token_embedding(t_u32 as usize) {
                    if emb.len() == dim && !emb.iter().any(|v| v.is_nan() || v.is_infinite()) {
                        for i in 0..dim { sum_vec[i] += emb[i] * w; }
                        total_w += w;
                    }
                }
            }
            if total_w > 0.0 {
                let norm = sum_vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                for v in sum_vec.iter_mut() { *v /= norm; }
            }
            sum_vec
        };

        let w1_fact = embed_weighted(p1_fact);
        let w1_query = embed_weighted(p1_query);
        let w2_fact = embed_weighted(p2_fact);
        let w2_query = embed_weighted(p2_query);
        let w_unrel = embed_weighted(p_unrelated);

        let w_sim1 = cosine_similarity(&w1_fact, &w1_query);
        let w_sim2 = cosine_similarity(&w2_fact, &w2_query);
        let w_unrel1 = cosine_similarity(&w1_fact, &w_unrel);
        let w_unrel2 = cosine_similarity(&w2_fact, &w_unrel);
        let w_cross = cosine_similarity(&w1_fact, &w2_fact);

        println!("  💎 Similitudes Coseno con Weighted Pooling (Down-weighting stopwords):");
        println!("     • Par Relacionado 1 ('Francia/París'):          CosSim = {:.4}", w_sim1);
        println!("     • Par Relacionado 2 ('GAJE/compresión'):        CosSim = {:.4}", w_sim2);
        println!("     • Cruzado No Relacionado ('Francia' vs 'GAJE'): CosSim = {:.4}", w_cross);
        println!("     • No Relacionado ('Francia' vs 'Pastel'):       CosSim = {:.4}", w_unrel1);
        println!("     • No Relacionado ('GAJE' vs 'Pastel'):          CosSim = {:.4}", w_unrel2);
        println!("  🎯 Separación Semántica Mejorada:");
        println!("     • Gap 1: +{:.4} (antes +{:.4})", w_sim1 - w_unrel1, gap1);
        println!("     • Gap 2: +{:.4} (antes +{:.4})\n", w_sim2 - w_unrel2, gap2);
    }

    println!("================================================================");
    println!("✅ TEST AISLADO FINALIZADO EXITOSAMENTE");
    println!("================================================================");
}
