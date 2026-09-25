use _impl::io::flat_reader::GajeFlatFileReader;
use _impl::nn::repl::load_model_and_tokenizer;
use std::path::Path;

#[test]
fn test_inspect_lm_head_and_logits() {
    let p = "models/production/qwen2_5_1_5b.gaje";
    if !Path::new(p).exists() {
        return;
    }

    // 1. Verificar metadata de lm_head en el flat reader
    let reader = GajeFlatFileReader::open(p).expect("abrir .gaje");
    let entry = reader.tensor_map.get("lm_head").expect("lm_head debe existir");
    println!("FlatTensorEntry para 'lm_head':");
    println!("   bit_depth: {}", entry.bit_depth);
    println!("   out_features: {}", entry.out_features);
    println!("   in_features: {}", entry.in_features);
    println!("   dna_len: {} bytes (esperado: 247959552)", entry.dna_len);
    println!("   c_len: {} bytes (esperado: 0)", entry.c_len);

    assert_eq!(entry.bit_depth, 8, "bit_depth debe ser 8");
    assert_eq!(entry.dna_len, 247_959_552, "dna_len debe ser exactamente 247,959,552 bytes");
    assert_eq!(entry.c_len, 0, "c_len debe ser 0 para Q8_0");

    // 2. Cargar modelo y ejecutar forward de 1 token
    let (mut llm, tokenizer) = load_model_and_tokenizer(p).expect("cargar modelo y tokenizer");
    let tokens = tokenizer.encode("Hola", false).unwrap();
    llm.clear_cache_core();
    let logits = llm.forward_core(tokens[0] as usize, false).unwrap();

    let nan_count = logits.iter().filter(|x| x.is_nan()).count();
    let inf_count = logits.iter().filter(|x| x.is_infinite()).count();
    println!("\nLogits del vocabulario ({} tokens):", logits.len());
    println!("   • NaNs: {} (debe ser 0)", nan_count);
    println!("   • Infs: {} (debe ser 0)", inf_count);

    assert_eq!(nan_count, 0, "No debe haber ningún NaN en los logits");
    assert_eq!(inf_count, 0, "No debe haber ningún Inf en los logits");

    // 3. Top 5 tokens más probables
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop 5 tokens para el próximo token tras 'Hola':");
    for &(id, val) in indexed.iter().take(5) {
        let text = tokenizer.decode(&[id as u32], false).unwrap_or_default();
        println!("   Token #{}: logit = {:.4}, text = {:?}", id, val, text);
    }

    // Verificar que el token dominante NO es [PAD151935]
    assert_ne!(indexed[0].0, 151935, "El token dominante no debe ser PAD151935");
}
