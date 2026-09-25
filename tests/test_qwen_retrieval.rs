use _impl::compute::island::{load_mu_vector, resolve_mu_vector_path};
use _impl::io::gmem::cosine_similarity;
use _impl::nn::llm::GenomicLLM;
use _impl::nn::repl::load_model_and_tokenizer;
use std::path::Path;

fn embed_whitened(
    llm: &GenomicLLM,
    tokenizer: &_impl::core::tokenizer::GajeTokenizer,
    text: &str,
    mu: &[f32],
) -> Vec<f32> {
    let raw = llm.embed_text(text, tokenizer).unwrap();
    let dim = llm.dim();
    let mut centered = vec![0.0f32; dim];
    let mut norm_sq = 0.0f32;

    for i in 0..dim {
        let diff = raw[i] - mu[i];
        centered[i] = diff;
        norm_sq += diff * diff;
    }
    let norm = norm_sq.sqrt().max(1e-8);
    for val in centered.iter_mut() {
        *val /= norm;
    }
    centered
}

#[test]
fn test_qwen_retrieval_discrimination() {
    let model_path = Path::new("models/production/qwen2_5_1_5b.gaje");
    if !model_path.exists() {
        println!("⚠️ Modelo no encontrado: {:?}", model_path);
        return;
    }

    let (llm, tokenizer) = load_model_and_tokenizer(model_path.to_str().unwrap()).expect("cargar Qwen 1.5B");
    let dim = llm.dim();
    assert_eq!(dim, 1536, "Dimensión esperada 1536");

    let mu_path = resolve_mu_vector_path(model_path).expect("debe resolver qwen2_5_1_5b.mu.bin");
    let mu = load_mu_vector(&mu_path, dim).expect("cargar .mu.bin");

    let query = "¿Cuál es la capital de Argentina?";
    let fact_buenos_aires = "La capital de Argentina es Buenos Aires.";
    let fact_canberra = "Canberra es la capital de Australia.";
    let fact_sena = "El río Sena atraviesa París.";

    // 1. Raw embeddings (sin centrado)
    let q_raw = llm.embed_text(query, &tokenizer).unwrap();
    let ba_raw = llm.embed_text(fact_buenos_aires, &tokenizer).unwrap();
    let can_raw = llm.embed_text(fact_canberra, &tokenizer).unwrap();
    let sena_raw = llm.embed_text(fact_sena, &tokenizer).unwrap();

    let sim_ba_raw = cosine_similarity(&q_raw, &ba_raw);
    let sim_can_raw = cosine_similarity(&q_raw, &can_raw);
    let sim_sena_raw = cosine_similarity(&q_raw, &sena_raw);

    // 2. Whitened embeddings (con centrado μ + normalización L2)
    let q_white = embed_whitened(&llm, &tokenizer, query, &mu);
    let ba_white = embed_whitened(&llm, &tokenizer, fact_buenos_aires, &mu);
    let can_white = embed_whitened(&llm, &tokenizer, fact_canberra, &mu);
    let sena_white = embed_whitened(&llm, &tokenizer, fact_sena, &mu);

    let sim_ba_white = cosine_similarity(&q_white, &ba_white);
    let sim_can_white = cosine_similarity(&q_white, &can_white);
    let sim_sena_white = cosine_similarity(&q_white, &sena_white);

    println!("\n===============================================================================");
    println!("🔬 VALIDACIÓN DE DISCRIMINACIÓN SEMÁNTICA — QWEN2.5-1.5B (151K VOCAB, 1536D)");
    println!("===============================================================================");
    println!("Query: \"{}\"\n", query);
    println!("  [RAW / Sin Centrado]:");
    println!("    • Buenos Aires (Match correcto):    cos = {:.4}", sim_ba_raw);
    println!("    • Canberra (Trampa estructural):    cos = {:.4}", sim_can_raw);
    println!("    • Río Sena (Distractor léxico):     cos = {:.4}", sim_sena_raw);
    println!("    • Margen Buenos Aires vs Canberra: {:+.4}", sim_ba_raw - sim_can_raw);
    println!();
    println!("  [WHITENED / Centrado μ + L2]:");
    println!("    • Buenos Aires (Match correcto):    cos = {:.4}", sim_ba_white);
    println!("    • Canberra (Trampa estructural):    cos = {:.4}", sim_can_white);
    println!("    • Río Sena (Distractor léxico):     cos = {:.4}", sim_sena_white);
    println!("    • Margen Buenos Aires vs Canberra: {:+.4}", sim_ba_white - sim_can_white);
    println!("    • Margen Buenos Aires vs Sena:     {:+.4}", sim_ba_white - sim_sena_white);
    println!("===============================================================================\n");

    // Aseveraciones de discriminación factual
    assert!(
        sim_ba_white > sim_can_white,
        "Buenos Aires ({}) debe superar a Canberra ({})",
        sim_ba_white,
        sim_can_white
    );
    assert!(
        sim_ba_white > sim_sena_white,
        "Buenos Aires ({}) debe superar al Sena ({})",
        sim_ba_white,
        sim_sena_white
    );
    assert!(
        sim_ba_white > 0.40,
        "Similitud de Buenos Aires debe ser sólida (> 0.40), obtenida: {}",
        sim_ba_white
    );
}
