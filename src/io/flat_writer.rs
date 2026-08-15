use crate::core::tokenizer::GajeTokenizer;
use crate::io::config::ModelConfig;
use crate::nn::{GenomicAttention, GenomicLLM, GenomicLinear, RustGenomicBlock};

pub fn save_genomic_model(
    path: &str,
    model: &GenomicLLM,
    config: &ModelConfig,
    tokenizer: Option<&GajeTokenizer>,
) -> std::io::Result<()> {
    let writer = crate::core::db::GajeDatabaseWriter::new(path).map_err(std::io::Error::other)?;
    let mut batch = writer.begin_batch_rust().map_err(std::io::Error::other)?;
    batch
        .write_metadata("config", &serde_json::to_string(config).unwrap())
        .unwrap();
    if let Some(tok) = tokenizer {
        batch
            .write_metadata("tokenizer", &tok.to_string(true).unwrap())
            .unwrap();
    }
    let compress = |d: &[u8]| lz4_flex::compress_prepend_size(d);
    let f32_u8 =
        |d: &[f32]| unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4) };
    let write_l = |b: &mut crate::core::db::GajeBatchWriter, p: &str, l: &GenomicLinear| {
        b.write_tensor(&format!("{}.dna", p), &compress(l.database_ref()))
            .unwrap();
        b.write_tensor(&format!("{}.centroids", p), &compress(f32_u8(&l.centroids)))
            .unwrap();
        b.write_tensor(
            &format!("{}.anchors", p),
            &compress(&l.anchors_sparse_buffer()),
        )
        .unwrap();
        if !l.bias.is_empty() {
            b.write_tensor(&format!("{}.bias", p), &compress(f32_u8(&l.bias)))
                .unwrap();
        }
    };
    write_l(&mut batch, "token_embd", &model.embeddings);
    for (i, blk) in model.blocks.iter().enumerate() {
        let p = format!("blk.{}.", i);
        write_l(&mut batch, &format!("{}attn_q", p), &blk.q_gen);
        write_l(&mut batch, &format!("{}attn_k", p), &blk.k_gen);
        write_l(&mut batch, &format!("{}attn_v", p), &blk.v_gen);
        write_l(&mut batch, &format!("{}attn_output", p), &blk.w_o);
        write_l(&mut batch, &format!("{}ffn_gate", p), &blk.gate_gen);
        write_l(&mut batch, &format!("{}ffn_up", p), &blk.up_gen);
        write_l(&mut batch, &format!("{}ffn_down", p), &blk.w_down);
        batch
            .write_tensor(
                &format!("{}attn_norm", p),
                &compress(f32_u8(&blk.attn.rmsnorm_weight)),
            )
            .unwrap();
        batch
            .write_tensor(&format!("{}ffn_norm", p), &compress(f32_u8(&blk.ffn_norm)))
            .unwrap();
        batch
            .write_tensor(&format!("{}h_scale", p), &compress(f32_u8(&[blk.h_scale])))
            .unwrap();
    }
    write_l(&mut batch, "lm_head", &model.lm_head);
    batch
        .write_tensor("output_norm", &compress(f32_u8(&model.output_norm)))
        .unwrap();
    batch.commit().unwrap();
    writer.compact().unwrap();
    Ok(())
}

pub fn init_born_genomic_model(
    path: &str,
    config: ModelConfig,
    vocab_size: usize,
) -> std::io::Result<GenomicLLM> {
    let b_s = 32;

    // Intento cargar centroides algebraicos (OpenAI Insight - Fase 5.0)
    let algebraic_c = if let Ok(f) = std::fs::File::open("models/core/algebraic_codebook.json") {
        let val: serde_json::Value = serde_json::from_reader(f).unwrap_or(serde_json::Value::Null);
        val.get("centroids")
            .and_then(|c| c.as_array())
            .and_then(|arr| {
                if arr.len() == 4 {
                    Some([
                        arr[0].as_f64()? as f32,
                        arr[1].as_f64()? as f32,
                        arr[2].as_f64()? as f32,
                        arr[3].as_f64()? as f32,
                    ])
                } else {
                    None
                }
            })
    } else {
        None
    };

    let init_l = |i: usize, o: usize| {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let n = i * o;
        let mut data = vec![0.0f32; n];
        for val in data.iter_mut() {
            *val = rng.gen_range(-0.02..0.02);
        }
        let (dna, c, a) = crate::compute::math::genomize_f32_core(&data, b_s, -1.0, algebraic_c);
        GenomicLinear::new(
            dna,
            a,
            c,
            o,
            i,
            b_s,
            Vec::new(),
            1e-6,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            2, // Default bit_depth for backwards compatibility
        )
    };
    let embeddings = init_l(config.n_embd, vocab_size);
    let mut blocks = Vec::new();
    let head_dim = config.n_embd / config.n_head;
    for i in 0..config.n_blocks {
        let attn = GenomicAttention::new(
            config.n_head,
            config.n_head_kv,
            head_dim,
            vec![1.0; config.n_embd],
            config.eps,
            config.config.rope_base,
            config.config.rope_style.clone(),
        );
        blocks.push(RustGenomicBlock::new(
            i,
            attn,
            init_l(config.n_embd, config.n_head * head_dim),
            init_l(config.n_embd, config.n_head_kv * head_dim),
            init_l(config.n_embd, config.n_head_kv * head_dim),
            init_l(config.n_head * head_dim, config.n_embd),
            init_l(config.n_embd, config.n_embd * 4),
            init_l(config.n_embd, config.n_embd * 4),
            init_l(config.n_embd * 4, config.n_embd),
            vec![1.0; config.n_embd],
            config.eps,
            config.config.ffn_act.clone(),
            config.config.use_genomic_norm,
            1.0,
            config.config.rna_threshold,
        ));
    }
    let model = GenomicLLM {
        embeddings,
        blocks,
        output_norm: vec![1.0; config.n_embd],
        lm_head: init_l(config.n_embd, vocab_size),
        eps: config.eps,
        k_wta_ratio: 0.50,
        topology: None,
    };
    save_genomic_model(path, &model, &config, None)?;
    Ok(model)
}
