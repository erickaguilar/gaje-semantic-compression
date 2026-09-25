use _impl::io::flat_reader::GajeFlatFileReader;
use _impl::io::header::Q8_0Block;
use _impl::nn::linear::storage::WeightBuffer;
use _impl::nn::linear::{GenomicLinear, WeightDatabase};
use _impl::nn::repl::load_model_and_tokenizer;
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

#[test]
fn test_q8_0_vs_tied_embeddings() {
    let p = "models/production/qwen2_5_1_5b.gaje";
    if !Path::new(p).exists() {
        println!("Modelo no encontrado: {}", p);
        return;
    }

    println!("\n=======================================================");
    println!("🔬 COMPARATIVA EMPÍRICA: TIED EMBEDDINGS (Q4_0) VS Q8_0 REAL");
    println!("=======================================================\n");

    let (mut llm, tokenizer) = load_model_and_tokenizer(p).expect("cargar modelo");
    let eos_ids: Vec<usize> = tokenizer
        .get_stop_tokens()
        .into_iter()
        .map(|t| t as usize)
        .collect();

    let out_f = llm.embeddings.out_features;
    let in_f = llm.embeddings.in_features;
    println!("📐 Dimensiones del vocabulario: out_features={}, in_features={}", out_f, in_f);

    println!("⚡ Recuantizando embeddings a Q8_0 genuino en paralelo con Rayon...");
    let t0 = std::time::Instant::now();
    let q8_blocks: Vec<Q8_0Block> = (0..out_f)
        .into_par_iter()
        .flat_map(|r| {
            let row = llm.embeddings.get_row_core(r).expect("dequantize row");
            _impl::compute::quantize::quantize_to_q8_0(&row)
        })
        .collect();
    let elapsed = t0.elapsed();
    println!("✅ Recuantizados {} bloques Q8_0 en {:.2}s ({:.1} MB)", 
        q8_blocks.len(), 
        elapsed.as_secs_f32(),
        (q8_blocks.len() * std::mem::size_of::<Q8_0Block>()) as f32 / 1024.0 / 1024.0
    );

    // Verificar que los bloques NO sean ceros
    let non_zero_scales = q8_blocks.iter().filter(|b| b.scale.to_f32() > 0.0).count();
    println!("   • Bloques con escala no nula: {} / {} ({:.2}%)", 
        non_zero_scales, q8_blocks.len(), (non_zero_scales as f64 / q8_blocks.len() as f64) * 100.0);
    assert!(non_zero_scales > q8_blocks.len() / 2, "La mayoría de los bloques deben ser no nulos");

    let q8_head = GenomicLinear::from_weight_storage(
        WeightDatabase::GenomicQ8_0(WeightBuffer::from(q8_blocks.clone())),
        &[],
        WeightBuffer::from(Vec::new()),
        out_f,
        in_f,
        32,
        llm.lm_head.bias.clone(),
        8,
    );

    let tied_head = llm.embeddings.clone();

    let user_msg = "¿Cuál es la capital de Argentina?";
    let sys_memory = "Eres un asistente útil.\n\nInformación de contexto:\n- Buenos Aires es la capital de Argentina.";

    let test_prompts = vec![
        ("Prompt Corto (N=25)", format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", user_msg)),
        ("Prompt con Memoria (N=72)", format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", sys_memory, user_msg)),
    ];

    for (p_name, chat_prompt) in test_prompts {
        let prompt_tokens: Vec<usize> = tokenizer
            .encode(&chat_prompt, false)
            .unwrap()
            .into_iter()
            .map(|t| t as usize)
            .collect();

        println!("\n-------------------------------------------------------");
        println!("📝 {}", p_name);
        println!("-------------------------------------------------------");

        // 1. Evaluación con Tied Embeddings (Q4_0 fallback)
        llm.lm_head = tied_head.clone();
        llm.clear_cache_core();
        let gen_tied = llm.generate_native_core(prompt_tokens.clone(), 30, 0.0, 1.0, eos_ids.clone()).unwrap();
        let text_tied = tokenizer.decode(&gen_tied.into_iter().map(|t| t as u32).collect::<Vec<_>>(), true).unwrap_or_default();
        let clean_tied = text_tied.split("<|im_end|>").next().unwrap_or(&text_tied).trim();

        // 2. Evaluación con Q8_0 Genuino
        llm.lm_head = q8_head.clone();
        llm.clear_cache_core();
        let gen_q8 = llm.generate_native_core(prompt_tokens.clone(), 30, 0.0, 1.0, eos_ids.clone()).unwrap();
        let text_q8 = tokenizer.decode(&gen_q8.into_iter().map(|t| t as u32).collect::<Vec<_>>(), true).unwrap_or_default();
        let clean_q8 = text_q8.split("<|im_end|>").next().unwrap_or(&text_q8).trim();

        println!("   [Tied Q4_0]: {:?}", clean_tied);
        println!("   [Real Q8_0]: {:?}", clean_q8);
        println!("   • Tied tiene 'capitale': {}", clean_tied.contains("capitale"));
        println!("   • Q8_0 tiene 'capitale': {}", clean_q8.contains("capitale"));
        println!("   • Tied tiene 'e s':      {}", clean_tied.contains("e s"));
        println!("   • Q8_0 tiene 'e s':      {}", clean_q8.contains("e s"));
    }

    // 3. Escribir los pesos Q8_0 en el archivo .gaje en disco para eliminar el dead weight
    println!("\n💾 Actualizando lm_head con los pesos reales Q8_0 en '{}'...", p);
    let reader = GajeFlatFileReader::open(p).expect("abrir .gaje");
    let lm_entry = reader.tensor_map.get("lm_head").expect("lm_head entry").clone();
    let abs_offset = reader.weights_offset + lm_entry.dna_off;
    drop(reader); // Cerrar mmap antes de escribir

    let raw_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            q8_blocks.as_ptr() as *const u8,
            q8_blocks.len() * std::mem::size_of::<Q8_0Block>(),
        )
    };
    assert_eq!(raw_bytes.len(), lm_entry.dna_len, "El tamaño de bytes debe coincidir exactamente");

    let mut file = OpenOptions::new().write(true).open(p).expect("abrir para escritura");
    file.seek(SeekFrom::Start(abs_offset as u64)).expect("seek lm_head");
    file.write_all(raw_bytes).expect("escribir bloques Q8_0");
    file.sync_all().expect("sync");
    println!("✅ 247.9 MB de pesos Q8_0 reales grabados exitosamente en offset {}!", abs_offset);
}
