use crate::core::tokenizer::GajeTokenizer;
use crate::io::config::ModelConfig;
use crate::io::flat_reader::FlatTensorEntry;
use crate::io::header::FlatHeaderV2;
use crate::nn::{GenomicAttention, GenomicLLM, GenomicLinear, RustGenomicBlock};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn align64_size(len: usize) -> usize {
    let pad = (64 - (len % 64)) % 64;
    len + pad
}

fn f32_u8(d: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4) }
}

fn write_at_offset(file: &std::fs::File, data: &[u8], offset: u64) -> std::io::Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.write_all_at(data, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut written = 0;
        while written < data.len() {
            let n = file.seek_write(&data[written..], offset + written as u64)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "Error de escritura en archivo",
                ));
            }
            written += n;
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        use std::io::{Seek, SeekFrom, Write};
        let mut f = file.try_clone()?;
        f.seek(SeekFrom::Start(offset))?;
        f.write_all(data)?;
        Ok(())
    }
}

struct TensorWriteTask {
    entry: FlatTensorEntry,
    dna_data: Vec<u8>,
    c_data: Vec<u8>,
    anc_data: Vec<u8>,
    bias_data: Vec<u8>,
}

fn add_linear(
    name: &str,
    l: &GenomicLinear,
    tasks: &mut Vec<TensorWriteTask>,
    current_offset: &mut usize,
) {
    let dna = l.database_ref().to_vec();
    let centroids = f32_u8(&l.centroids).to_vec();
    let anchors = l.anchors_sparse_buffer().to_vec();
    let bias = f32_u8(&l.bias).to_vec();

    let dna_off = *current_offset;
    *current_offset = align64_size(dna_off + dna.len());

    let c_off = *current_offset;
    *current_offset = align64_size(c_off + centroids.len());

    let anc_off = *current_offset;
    *current_offset = align64_size(anc_off + anchors.len());

    let bias_off = *current_offset;
    *current_offset = align64_size(bias_off + bias.len());

    tasks.push(TensorWriteTask {
        entry: FlatTensorEntry {
            name: name.to_string(),
            bit_depth: l.bit_depth() as usize,
            out_features: l.out_features,
            in_features: l.in_features,
            dna_off,
            dna_len: dna.len(),
            c_off,
            c_len: centroids.len(),
            anc_off,
            anc_len: anchors.len(),
            bias_off,
            bias_len: bias.len(),
        },
        dna_data: dna,
        c_data: centroids,
        anc_data: anchors,
        bias_data: bias,
    });
}

fn add_raw_f32(
    name: &str,
    f32_data: &[f32],
    tasks: &mut Vec<TensorWriteTask>,
    current_offset: &mut usize,
) {
    let bytes = f32_u8(f32_data).to_vec();
    let dna_off = *current_offset;
    *current_offset = align64_size(dna_off + bytes.len());

    tasks.push(TensorWriteTask {
        entry: FlatTensorEntry {
            name: name.to_string(),
            bit_depth: 32,
            out_features: f32_data.len(),
            in_features: 1,
            dna_off,
            dna_len: bytes.len(),
            c_off: *current_offset,
            c_len: 0,
            anc_off: *current_offset,
            anc_len: 0,
            bias_off: *current_offset,
            bias_len: 0,
        },
        dna_data: bytes,
        c_data: Vec::new(),
        anc_data: Vec::new(),
        bias_data: Vec::new(),
    });
}

/// Escribe el modelo en el **formato mmap plano `GAJE`** (magic `GAJE`, zero-copy).
pub fn save_genomic_flat(
    path: &str,
    model: &GenomicLLM,
    config: &ModelConfig,
    tokenizer: Option<&GajeTokenizer>,
) -> std::io::Result<()> {
    save_genomic_flat_q(path, model, config, tokenizer, 1)
}

/// Variante de `save_genomic_flat` que pre-asigna el archivo con `file.set_len()`
/// y escribe los tensores en paralelo mediante Rayon (`pwrite` / `write_all_at`).
pub fn save_genomic_flat_q(
    path: &str,
    model: &GenomicLLM,
    config: &ModelConfig,
    tokenizer: Option<&GajeTokenizer>,
    quant_format: u32,
) -> std::io::Result<()> {
    let mut tasks: Vec<TensorWriteTask> = Vec::new();
    let mut current_offset = 0usize;

    // 1. Registrar todos los tensores y computar offsets
    add_linear("token_embd", &model.embeddings, &mut tasks, &mut current_offset);
    for (i, blk) in model.blocks.iter().enumerate() {
        let p = format!("blk.{i}.");
        add_raw_f32(&format!("{}attn_norm", p), &blk.attn.rmsnorm_weight, &mut tasks, &mut current_offset);
        add_raw_f32(&format!("{}ffn_norm", p), &blk.ffn_norm, &mut tasks, &mut current_offset);
        if let Some(qkv) = &blk.fused_qkv {
            add_linear(&format!("{}attn_qkv", p), qkv, &mut tasks, &mut current_offset);
        } else {
            add_linear(&format!("{}attn_q", p), &blk.q_gen, &mut tasks, &mut current_offset);
            add_linear(&format!("{}attn_k", p), &blk.k_gen, &mut tasks, &mut current_offset);
            add_linear(&format!("{}attn_v", p), &blk.v_gen, &mut tasks, &mut current_offset);
        }
        add_linear(&format!("{}attn_output", p), &blk.w_o, &mut tasks, &mut current_offset);
        if let Some(gu) = &blk.fused_gate_up {
            add_linear(&format!("{}ffn_gate_up", p), gu, &mut tasks, &mut current_offset);
        } else {
            add_linear(&format!("{}ffn_gate", p), &blk.gate_gen, &mut tasks, &mut current_offset);
            add_linear(&format!("{}ffn_up", p), &blk.up_gen, &mut tasks, &mut current_offset);
        }
        add_linear(&format!("{}ffn_down", p), &blk.w_down, &mut tasks, &mut current_offset);
    }
    add_linear("lm_head", &model.lm_head, &mut tasks, &mut current_offset);
    add_raw_f32("output_norm", &model.output_norm, &mut tasks, &mut current_offset);

    let total_weights_len = current_offset;
    let dir: Vec<FlatTensorEntry> = tasks.iter().map(|t| t.entry.clone()).collect();

    // 2. Metadatos JSON y Directorio
    let mut meta: serde_json::Value = serde_json::to_value(config).map_err(std::io::Error::other)?;
    if let Some(tok) = tokenizer {
        if let Some(obj) = meta.as_object_mut() {
            if let Ok(tok_str) = tok.to_string(true) {
                obj.insert("tokenizer".to_string(), serde_json::Value::String(tok_str));
            }
        }
    }
    let metadata_json = serde_json::to_vec(&meta).map_err(std::io::Error::other)?;
    let dir_json = serde_json::to_vec(&dir).map_err(std::io::Error::other)?;

    let header_fixed_size = 4096usize;
    let mut weights_offset = header_fixed_size + metadata_json.len() + dir_json.len();
    if weights_offset % 4096 != 0 {
        weights_offset = ((weights_offset / 4096) + 1) * 4096;
    }

    // 3. Obtener GTOK binario para incrustación automática
    let gtok_bytes: Vec<u8> = {
        let qwen_gtok = PathBuf::from("models/core/tokenizers/qwen2_5_tokenizer.gtok");
        let smol_gtok = PathBuf::from("models/core/tokenizers/smollm2_tokenizer.gtok");
        let def_gtok = PathBuf::from("models/core/tokenizer.gtok");

        if config.n_embd >= 1536 && qwen_gtok.exists() {
            std::fs::read(qwen_gtok).unwrap_or_default()
        } else if smol_gtok.exists() {
            std::fs::read(smol_gtok).unwrap_or_default()
        } else if def_gtok.exists() {
            std::fs::read(def_gtok).unwrap_or_default()
        } else {
            Vec::new()
        }
    };

    let gtok_offset = (weights_offset + total_weights_len) as u64;
    let gtok_len = gtok_bytes.len() as u64;
    let total_file_size = (weights_offset + total_weights_len + gtok_bytes.len()) as u64;

    // Detectar familia de arquitectura
    let arch_family = if config.n_embd == 576 {
        2 // SmolLM
    } else if config.n_embd == 896 || config.n_embd == 1536 || config.n_embd == 2048 || config.n_embd == 3584 {
        4 // Qwen2_5
    } else {
        1 // Llama / Genérico
    };

    // 4. Construir Cabecera FlatHeaderV2
    let header = FlatHeaderV2 {
        magic: *b"GAJE",
        version: 0x000908,
        flags: 0x0003,
        num_tensors: dir.len() as u32,
        meta_len: metadata_json.len() as u64,
        dir_len: dir_json.len() as u64,
        weights_offset: weights_offset as u64,
        weights_len: total_weights_len as u64,
        group_size: 32,
        quant_format,
        arch_family,
        arch_n_embd: config.n_embd as u32,
        arch_n_head: config.n_head as u32,
        arch_n_head_kv: config.n_head_kv as u32,
        arch_n_blocks: config.n_blocks as u32,
        arch_qk_permute: 0,
        gtok_offset: if gtok_len > 0 { gtok_offset } else { 0 },
        gtok_len,
        reserved: [0u8; 4000],
    };

    let mut header_bin = [0u8; 4096];
    unsafe {
        std::ptr::write(header_bin.as_mut_ptr() as *mut FlatHeaderV2, header);
    }

    // 5. Pre-asignar archivo con set_len (Zero fragmentation, instantáneo)
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    file.set_len(total_file_size)?;

    let file_arc = Arc::new(file);

    // Escribir cabecera, metadatos y directorio al inicio
    write_at_offset(&file_arc, &header_bin, 0)?;
    write_at_offset(&file_arc, &metadata_json, 4096)?;
    write_at_offset(&file_arc, &dir_json, 4096 + metadata_json.len() as u64)?;

    // 6. Escritura masiva en paralelo de tensores usando Rayon
    let base_weights = weights_offset as u64;
    tasks.into_par_iter().for_each(|task| {
        let e = &task.entry;
        if !task.dna_data.is_empty() {
            let _ = write_at_offset(&file_arc, &task.dna_data, base_weights + e.dna_off as u64);
        }
        if !task.c_data.is_empty() {
            let _ = write_at_offset(&file_arc, &task.c_data, base_weights + e.c_off as u64);
        }
        if !task.anc_data.is_empty() {
            let _ = write_at_offset(&file_arc, &task.anc_data, base_weights + e.anc_off as u64);
        }
        if !task.bias_data.is_empty() {
            let _ = write_at_offset(&file_arc, &task.bias_data, base_weights + e.bias_off as u64);
        }
    });

    // 7. Escribir GTOK si está presente
    if !gtok_bytes.is_empty() {
        write_at_offset(&file_arc, &gtok_bytes, gtok_offset)?;
    }

    file_arc.sync_all()?;
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
        if let Ok(tok_str) = tok.to_string(true) {
            batch.write_metadata("tokenizer", &tok_str).unwrap();
        }
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
            4,
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
