use _impl::compute::island::{IslandNiche, IslandOrchestrator};
use _impl::nn::repl::load_model_and_tokenizer;

#[test]
fn test_entropy_gating_and_traps() {
    let (llm, tokenizer) = load_model_and_tokenizer("models/born/max.gaje").expect("cargar max.gaje");
    let dim = llm.dim() as u32;

    println!("\n=========================================================================================");
    println!("🛡️ TEST DE VERIFICACIÓN DE GATING DE ENTROPÍA Y CASOS TRAMPA (max.gaje)");
    println!("=========================================================================================\n");

    let mut orch = IslandOrchestrator::new(dim);
    orch.min_similarity = 0.42;
    orch.documental_min_sim = 0.50;
    orch.entropy_gap_threshold = 0.12;
    orch.kwta_ratio = 0.90;

    // 1. Poblamos el nicho documental con hechos fácticos
    let facts = [
        (1, "La capital de Francia es París."),
        (2, "El agua hierve a 100°C a nivel del mar."),
        (3, "GAJE comprime modelos con cuantización Q4_0."),
        (4, "Marte es el cuarto planeta del sistema solar."),
        (5, "El ADN contiene información genética."),
        (6, "Rust es un lenguaje de programación de sistemas."),
        (7, "El Everest es la montaña más alta del mundo."),
        (8, "La Revolución Francesa comenzó en 1789."),
        (9, "Madrid es la capital de España."),
        (18, "La Segunda Guerra Mundial terminó en 1945."),
    ];

    for (id, text) in &facts {
        let vec = llm.embed_text(text, &tokenizer).unwrap();
        orch.add_memory(IslandNiche::Documental, *id, vec, text.to_string());
    }

    println!("📚 Base documental cargada con {} hechos.\n", facts.len());

    // 2. Consulta Positiva Legítima (P18)
    let q_p18 = "¿Cuándo acabó la Segunda Guerra Mundial?";
    let v_p18 = llm.embed_text(q_p18, &tokenizer).unwrap();
    let matches_p18 = orch.retrieve_context(&v_p18, 3);
    let prompt_p18 = orch.build_augmented_prompt_from_matches(q_p18, &matches_p18, 128);

    println!("-----------------------------------------------------------------------------------------");
    println!("🔍 Caso Positivo P18: '{}'", q_p18);
    for (i, m) in matches_p18.iter().enumerate() {
        println!("   [{}] Sim={:.4} | Id={} | Texto='{}'", i + 1, m.similarity, m.id, m.text);
    }
    let p18_injected = prompt_p18.contains("Segunda Guerra Mundial");
    println!("   ➡️ ¿Contexto inyectado en prompt?: {} (Esperado: SÍ)\n", p18_injected);

    // 3. Consulta Trampa Léxica (N18)
    let q_n18 = "¿Qué causó la Primera Guerra Mundial?";
    let v_n18 = llm.embed_text(q_n18, &tokenizer).unwrap();
    let matches_n18 = orch.retrieve_context(&v_n18, 3);
    let prompt_n18 = orch.build_augmented_prompt_from_matches(q_n18, &matches_n18, 128);

    println!("-----------------------------------------------------------------------------------------");
    println!("⚠️ Caso Trampa N18: '{}'", q_n18);
    for (i, m) in matches_n18.iter().enumerate() {
        println!("   [{}] Sim={:.4} | Id={} | Texto='{}'", i + 1, m.similarity, m.id, m.text);
    }
    let n18_injected = prompt_n18.contains("Segunda Guerra Mundial");
    println!("   ➡️ ¿Hecho espurio inyectado?: {} (Esperado: NO)\n", n18_injected);

    // 4. Consulta Trampa Geográfica (N9)
    let q_n9 = "¿Cuál es el río más largo?";
    let v_n9 = llm.embed_text(q_n9, &tokenizer).unwrap();
    let matches_n9 = orch.retrieve_context(&v_n9, 3);
    let prompt_n9 = orch.build_augmented_prompt_from_matches(q_n9, &matches_n9, 128);

    println!("-----------------------------------------------------------------------------------------");
    println!("⚠️ Caso Trampa N9: '{}'", q_n9);
    for (i, m) in matches_n9.iter().enumerate() {
        println!("   [{}] Sim={:.4} | Id={} | Texto='{}'", i + 1, m.similarity, m.id, m.text);
    }
    let n9_injected = prompt_n9.contains("Everest");
    println!("   ➡️ ¿Hecho espurio inyectado?: {} (Esperado: NO)\n", n9_injected);
}

#[test]
fn test_entropy_gating_qwen_whitening() {
    let (llm, tokenizer) = load_model_and_tokenizer("models/production/qwen2_5_0_5b_q2_0.gaje").expect("cargar qwen");
    let dim = llm.dim() as u32;

    println!("\n=========================================================================================");
    println!("🛡️ TEST DE VERIFICACIÓN DE GATING Y TRAMPAS (qwen2_5_0_5b_q2_0 con Whitening)");
    println!("=========================================================================================\n");

    // Calculamos mu
    let mut sum_vec = vec![0.0f32; dim as usize];
    let mut count = 0usize;
    if let Ok(file) = std::fs::File::open("data/latam_corpus.jsonl") {
        use std::io::BufRead;
        let reader = std::io::BufReader::new(file);
        for line in reader.lines().flatten() {
            if let Ok(v) = llm.embed_text(&line, &tokenizer) {
                for i in 0..dim as usize { sum_vec[i] += v[i]; }
                count += 1;
                if count >= 100 { break; }
            }
        }
    }
    for val in sum_vec.iter_mut() { *val /= count as f32; }

    let embed_white = |text: &str| -> Vec<f32> {
        let raw = llm.embed_text(text, &tokenizer).unwrap();
        let mut centered = vec![0.0f32; dim as usize];
        let mut norm_sq = 0.0f32;
        for i in 0..dim as usize {
            let diff = raw[i] - sum_vec[i];
            centered[i] = diff;
            norm_sq += diff * diff;
        }
        let norm = norm_sq.sqrt().max(1e-8);
        for val in centered.iter_mut() { *val /= norm; }
        centered
    };

    let mut orch = IslandOrchestrator::new(dim);
    orch.min_similarity = 0.33;
    orch.documental_min_sim = 0.35;
    orch.entropy_gap_threshold = 0.08;
    orch.kwta_ratio = 0.90;

    let facts = [
        (1, "La capital de Francia es París."),
        (2, "El agua hierve a 100°C a nivel del mar."),
        (3, "GAJE comprime modelos con cuantización Q4_0."),
        (4, "Marte es el cuarto planeta del sistema solar."),
        (5, "El ADN contiene información genética."),
        (6, "Rust es un lenguaje de programación de sistemas."),
        (7, "El Everest es la montaña más alta del mundo."),
        (8, "La Revolución Francesa comenzó en 1789."),
        (9, "Madrid es la capital de España."),
        (18, "La Segunda Guerra Mundial terminó en 1945."),
    ];

    for (id, text) in &facts {
        let vec = embed_white(text);
        orch.add_memory(IslandNiche::Documental, *id, vec, text.to_string());
    }

    // 1. P18
    let q_p18 = "¿Cuándo acabó la Segunda Guerra Mundial?";
    let v_p18 = embed_white(q_p18);
    let matches_p18 = orch.retrieve_context(&v_p18, 3);
    let prompt_p18 = orch.build_augmented_prompt_from_matches(q_p18, &matches_p18, 128);

    println!("-----------------------------------------------------------------------------------------");
    println!("🔍 Caso Positivo P18: '{}'", q_p18);
    for (i, m) in matches_p18.iter().enumerate() {
        println!("   [{}] Sim={:.4} | Id={} | Texto='{}'", i + 1, m.similarity, m.id, m.text);
    }
    let p18_injected = prompt_p18.contains("Segunda Guerra Mundial");
    println!("   ➡️ ¿Contexto inyectado en prompt?: {}\n", p18_injected);

    // 2. N18
    let q_n18 = "¿Qué causó la Primera Guerra Mundial?";
    let v_n18 = embed_white(q_n18);
    let matches_n18 = orch.retrieve_context(&v_n18, 3);
    let prompt_n18 = orch.build_augmented_prompt_from_matches(q_n18, &matches_n18, 128);

    println!("-----------------------------------------------------------------------------------------");
    println!("⚠️ Caso Trampa N18: '{}'", q_n18);
    for (i, m) in matches_n18.iter().enumerate() {
        println!("   [{}] Sim={:.4} | Id={} | Texto='{}'", i + 1, m.similarity, m.id, m.text);
    }
    let n18_injected = prompt_n18.contains("Segunda Guerra Mundial");
    println!("   ➡️ ¿Hecho espurio inyectado?: {}\n", n18_injected);
}

#[test]
fn test_entropy_gating_max_whitening() {
    let (llm, tokenizer) = load_model_and_tokenizer("models/born/max.gaje").expect("cargar max");
    let dim = llm.dim() as u32;

    println!("\n=========================================================================================");
    println!("🛡️ TEST DE VERIFICACIÓN DE GATING Y TRAMPAS (max.gaje con Whitening)");
    println!("=========================================================================================\n");

    let mut sum_vec = vec![0.0f32; dim as usize];
    let mut count = 0usize;
    if let Ok(file) = std::fs::File::open("data/latam_corpus.jsonl") {
        use std::io::BufRead;
        let reader = std::io::BufReader::new(file);
        for line in reader.lines().flatten() {
            if let Ok(v) = llm.embed_text(&line, &tokenizer) {
                for i in 0..dim as usize { sum_vec[i] += v[i]; }
                count += 1;
                if count >= 100 { break; }
            }
        }
    }
    for val in sum_vec.iter_mut() { *val /= count as f32; }

    let embed_white = |text: &str| -> Vec<f32> {
        let raw = llm.embed_text(text, &tokenizer).unwrap();
        let mut centered = vec![0.0f32; dim as usize];
        let mut norm_sq = 0.0f32;
        for i in 0..dim as usize {
            let diff = raw[i] - sum_vec[i];
            centered[i] = diff;
            norm_sq += diff * diff;
        }
        let norm = norm_sq.sqrt().max(1e-8);
        for val in centered.iter_mut() { *val /= norm; }
        centered
    };

    let mut orch = IslandOrchestrator::new(dim);
    orch.min_similarity = 0.40;
    orch.documental_min_sim = 0.42;
    orch.entropy_gap_threshold = 0.08;
    orch.kwta_ratio = 0.90;

    let facts = [
        (1, "La capital de Francia es París."),
        (2, "El agua hierve a 100°C a nivel del mar."),
        (3, "GAJE comprime modelos con cuantización Q4_0."),
        (4, "Marte es el cuarto planeta del sistema solar."),
        (5, "El ADN contiene información genética."),
        (6, "Rust es un lenguaje de programación de sistemas."),
        (7, "El Everest es la montaña más alta del mundo."),
        (8, "La Revolución Francesa comenzó en 1789."),
        (9, "Madrid es la capital de España."),
        (18, "La Segunda Guerra Mundial terminó en 1945."),
    ];

    for (id, text) in &facts {
        let vec = embed_white(text);
        orch.add_memory(IslandNiche::Documental, *id, vec, text.to_string());
    }

    // 1. P18
    let q_p18 = "¿Cuándo acabó la Segunda Guerra Mundial?";
    let v_p18 = embed_white(q_p18);
    let matches_p18 = orch.retrieve_context(&v_p18, 3);
    let prompt_p18 = orch.build_augmented_prompt_from_matches(q_p18, &matches_p18, 128);

    println!("-----------------------------------------------------------------------------------------");
    println!("🔍 Caso Positivo P18: '{}'", q_p18);
    for (i, m) in matches_p18.iter().enumerate() {
        println!("   [{}] Sim={:.4} | Id={} | Texto='{}'", i + 1, m.similarity, m.id, m.text);
    }
    let p18_injected = prompt_p18.contains("Segunda Guerra Mundial");
    println!("   ➡️ ¿Contexto inyectado en prompt?: {}\n", p18_injected);

    // 2. N18
    let q_n18 = "¿Qué causó la Primera Guerra Mundial?";
    let v_n18 = embed_white(q_n18);
    let matches_n18 = orch.retrieve_context(&v_n18, 3);
    let prompt_n18 = orch.build_augmented_prompt_from_matches(q_n18, &matches_n18, 128);

    println!("-----------------------------------------------------------------------------------------");
    println!("⚠️ Caso Trampa N18: '{}'", q_n18);
    for (i, m) in matches_n18.iter().enumerate() {
        println!("   [{}] Sim={:.4} | Id={} | Texto='{}'", i + 1, m.similarity, m.id, m.text);
    }
    let n18_injected = prompt_n18.contains("Segunda Guerra Mundial");
    println!("   ➡️ ¿Hecho espurio inyectado?: {}\n", n18_injected);
}
