use crate::core::tokenizer::GajeTokenizer;
use crate::io::config::ModelConfig;
use crate::io::flat_reader::FlatTensorEntry;
use crate::io::header::FlatHeaderV2;
use crate::nn::{GenomicAttention, GenomicLLM, GenomicLinear, RustGenomicBlock};

fn align64(v: &mut Vec<u8>) {
    let pad = (64 - (v.len() % 64)) % 64;
    v.resize(v.len() + pad, 0);
}

fn f32_u8(d: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4) }
}

fn push_tensor(
    dir: &mut Vec<FlatTensorEntry>,
    blob: &mut Vec<u8>,
    name: String,
    bit_depth: usize,
    out: usize,
    inn: usize,
    dna: &[u8],
    centroids: &[f32],
    anchors: &[u8],
    bias: &[f32],
) {
    let dna_off = blob.len();
    blob.extend_from_slice(dna);
    align64(blob);
    let c_off = blob.len();
    blob.extend_from_slice(f32_u8(centroids));
    align64(blob);
    let anc_off = blob.len();
    blob.extend_from_slice(anchors);
    align64(blob);
    let bias_off = blob.len();
    blob.extend_from_slice(f32_u8(bias));
    align64(blob);
    dir.push(FlatTensorEntry {
        name,
        bit_depth,
        out_features: out,
        in_features: inn,
        dna_off,
        dna_len: dna.len(),
        c_off,
        c_len: centroids.len() * 4,
        anc_off,
        anc_len: anchors.len(),
        bias_off,
        bias_len: bias.len() * 4,
    });
}

/// Escribe el modelo en el **formato mmap plano `GAJE`** (magic `GAJE`, zero-copy),
/// el mismo que produce `scripts/export_gaje_flat.py` y que carga el Web UI con
/// `load_genomic_auto` sin pasar por redb. Los tensores se escriben con el layout
/// **separado** (attn_q/k/v, ffn_gate/up), que el reader detecta con
/// `has_fused_qkv=false`/`has_fused_gate_up=false`.
pub fn save_genomic_flat(
    path: &str,
    model: &GenomicLLM,
    config: &ModelConfig,
    tokenizer: Option<&GajeTokenizer>,
) -> std::io::Result<()> {
    save_genomic_flat_q(path, model, config, tokenizer, 1)
}

/// Variante de `save_genomic_flat` que fija el `quant_format` de la cabecera
/// (1 = Q4_0, 3 = Q2_0). Los tensores se escriben con el layout separado
/// (attn_q/k/v, ffn_gate/up) y cada entrada lleva su propio `bit_depth`.
pub fn save_genomic_flat_q(
    path: &str,
    model: &GenomicLLM,
    config: &ModelConfig,
    tokenizer: Option<&GajeTokenizer>,
    quant_format: u32,
) -> std::io::Result<()> {
    let mut blob: Vec<u8> = Vec::new();
    let mut dir: Vec<FlatTensorEntry> = Vec::new();

    let mut write_l =
        |blob: &mut Vec<u8>, dir: &mut Vec<FlatTensorEntry>, p: &str, l: &GenomicLinear| {
            push_tensor(
                dir,
                blob,
                p.to_string(),
                l.bit_depth() as usize,
                l.out_features,
                l.in_features,
                l.database_ref(),
                &l.centroids,
                &l.anchors_sparse_buffer(),
                &l.bias,
            );
        };

    write_l(&mut blob, &mut dir, "token_embd", &model.embeddings);
    for (i, blk) in model.blocks.iter().enumerate() {
        let p = format!("blk.{i}.");
        push_tensor(
            &mut dir,
            &mut blob,
            format!("{}attn_norm", p),
            32,
            blk.attn.rmsnorm_weight.len(),
            1,
            f32_u8(&blk.attn.rmsnorm_weight),
            &[],
            &[],
            &[],
        );
        push_tensor(
            &mut dir,
            &mut blob,
            format!("{}ffn_norm", p),
            32,
            blk.ffn_norm.len(),
            1,
            f32_u8(&blk.ffn_norm),
            &[],
            &[],
            &[],
        );
        if let Some(qkv) = &blk.fused_qkv {
            write_l(&mut blob, &mut dir, &format!("{}attn_qkv", p), qkv);
        } else {
            write_l(&mut blob, &mut dir, &format!("{}attn_q", p), &blk.q_gen);
            write_l(&mut blob, &mut dir, &format!("{}attn_k", p), &blk.k_gen);
            write_l(&mut blob, &mut dir, &format!("{}attn_v", p), &blk.v_gen);
        }
        write_l(&mut blob, &mut dir, &format!("{}attn_output", p), &blk.w_o);
        if let Some(gu) = &blk.fused_gate_up {
            write_l(&mut blob, &mut dir, &format!("{}ffn_gate_up", p), gu);
        } else {
            write_l(
                &mut blob,
                &mut dir,
                &format!("{}ffn_gate", p),
                &blk.gate_gen,
            );
            write_l(&mut blob, &mut dir, &format!("{}ffn_up", p), &blk.up_gen);
        }
        write_l(&mut blob, &mut dir, &format!("{}ffn_down", p), &blk.w_down);
    }
    write_l(&mut blob, &mut dir, "lm_head", &model.lm_head);
    push_tensor(
        &mut dir,
        &mut blob,
        "output_norm".to_string(),
        32,
        model.output_norm.len(),
        1,
        f32_u8(&model.output_norm),
        &[],
        &[],
        &[],
    );

    let mut meta: serde_json::Value =
        serde_json::to_value(config).map_err(std::io::Error::other)?;
    if let Some(tok) = tokenizer {
        if let Some(obj) = meta.as_object_mut() {
            obj.insert(
                "tokenizer".to_string(),
                serde_json::Value::String(
                    tok.to_string(true).map_err(|e| {
                        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
                    })?,
                ),
            );
        }
    }
    let metadata_json = serde_json::to_vec(&meta).map_err(std::io::Error::other)?;
    let dir_json = serde_json::to_vec(&dir).map_err(std::io::Error::other)?;

    let header_fixed_size = 4096usize;
    let mut weights_offset = header_fixed_size + metadata_json.len() + dir_json.len();
    if weights_offset % 4096 != 0 {
        weights_offset = ((weights_offset / 4096) + 1) * 4096;
    }

    let header = FlatHeaderV2 {
        magic: *b"GAJE",
        version: 0x000908,
        flags: 0x0003,
        num_tensors: dir.len() as u32,
        meta_len: metadata_json.len() as u64,
        dir_len: dir_json.len() as u64,
        weights_offset: weights_offset as u64,
        weights_len: blob.len() as u64,
        group_size: 32,
        quant_format,
        arch_family: 0,
        arch_n_embd: 0,
        arch_n_head: 0,
        arch_n_head_kv: 0,
        arch_n_blocks: 0,
        arch_qk_permute: 0,
        gtok_offset: 0,
        gtok_len: 0,
        reserved: [0u8; 4000],
    };

    let mut header_bin = [0u8; 4096];
    unsafe {
        std::ptr::write(header_bin.as_mut_ptr() as *mut FlatHeaderV2, header);
    }

    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    f.write_all(&header_bin)?;
    f.write_all(&metadata_json)?;
    f.write_all(&dir_json)?;
    let current_pos = 4096 + metadata_json.len() + dir_json.len();
    if weights_offset > current_pos {
        f.write_all(&vec![0u8; weights_offset - current_pos])?;
    }
    f.write_all(&blob)?;
    f.flush()?;
    Ok(())
}

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
        let n_blocks = n / 32;
        let mut q4_blocks = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            let mut qs = [0u8; 16];
            for q in qs.iter_mut() {
                let low = rng.gen_range(0..16u8);
                let high = rng.gen_range(0..16u8);
                *q = low | (high << 4);
            }
            q4_blocks.push(crate::io::header::Q4_0Block {
                scale: half::f16::from_f32(0.002),
                min: half::f16::from_f32(-0.015),
                qs,
            });
        }
        let u8_bytes: Vec<u8> = unsafe {
            let ptr = q4_blocks.as_ptr() as *const u8;
            let len = q4_blocks.len() * std::mem::size_of::<crate::io::header::Q4_0Block>();
            std::slice::from_raw_parts(ptr, len).to_vec()
        };
        GenomicLinear::new(
            u8_bytes,
            Vec::new(),
            Vec::new(),
            o,
            i,
            32,
            Vec::new(),
            1e-6,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            4, // 4-bit Q4_0 nativo
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
        quantum_embeddings: None,
    };
    if path.ends_with(".flat") {
        save_genomic_flat(path, &model, &config, None)?;
    } else {
        save_genomic_model(path, &model, &config, None)?;
    }
    Ok(model)
}
