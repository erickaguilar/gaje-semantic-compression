use crate::core::db::{METADATA_TABLE, TENSOR_TABLE};
use crate::nn::{GenomicAttention, GenomicLinear, RustGenomicBlock, RustGenomicLLM};
use pyo3::prelude::*;
use redb::{Database, ReadTransaction};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[pyclass]
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ArchConfig {
    #[pyo3(get, set)]
    #[serde(default = "default_name")]
    pub name: String,
    #[pyo3(get, set)]
    #[serde(default = "default_tokenizer")]
    pub tokenizer_id: String,
    #[pyo3(get, set)]
    #[serde(default = "default_rope_base")]
    pub rope_base: f32,
    #[pyo3(get, set)]
    #[serde(default = "default_ffn_act")]
    pub ffn_act: String,
    #[pyo3(get, set)]
    #[serde(default = "default_false")]
    pub use_genomic_norm: bool,
    #[pyo3(get, set)]
    #[serde(default = "default_rope_style")]
    pub rope_style: String,
    #[pyo3(get, set)]
    #[serde(default = "default_anchor_threshold")]
    pub anchor_threshold: f32,
    #[pyo3(get, set)]
    #[serde(default = "default_anchor_threshold")]
    pub ffn_anchor_threshold: f32,
    #[pyo3(get, set)]
    #[serde(default = "default_false")]
    pub unpermute_weights: bool,
    #[pyo3(get, set)]
    #[serde(default = "default_false")]
    pub apply_smollm_rope_patch: bool,
}

#[pymethods]
impl ArchConfig {
    #[new]
    #[pyo3(signature = (name = "GAJE-Model".to_string(), tokenizer_id = "gpt2".to_string(), rope_base = 10000.0, ffn_act = "swiglu".to_string(), use_genomic_norm = false, rope_style = "split".to_string(), anchor_threshold = 0.1, ffn_anchor_threshold = 0.1, unpermute_weights = false, apply_smollm_rope_patch = false))]
    pub fn new(
        name: String,
        tokenizer_id: String,
        rope_base: f32,
        ffn_act: String,
        use_genomic_norm: bool,
        rope_style: String,
        anchor_threshold: f32,
        ffn_anchor_threshold: f32,
        unpermute_weights: bool,
        apply_smollm_rope_patch: bool,
    ) -> Self {
        ArchConfig {
            name,
            tokenizer_id,
            rope_base,
            ffn_act,
            use_genomic_norm,
            rope_style,
            anchor_threshold,
            ffn_anchor_threshold,
            unpermute_weights,
            apply_smollm_rope_patch,
        }
    }
}

fn default_name() -> String {
    "GAJE-Model".to_string()
}
fn default_tokenizer() -> String {
    "gpt2".to_string()
}
fn default_rope_base() -> f32 {
    10000.0
}
fn default_ffn_act() -> String {
    "swiglu".to_string()
}
fn default_false() -> bool {
    false
}
fn default_rope_style() -> String {
    "split".to_string()
}
fn default_anchor_threshold() -> f32 {
    0.1
}

#[pyclass]
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ModelConfig {
    #[pyo3(get, set)]
    pub config: ArchConfig,
    #[pyo3(get, set)]
    pub n_embd: usize,
    #[pyo3(get, set)]
    pub n_head: usize,
    #[pyo3(get, set)]
    pub n_head_kv: usize,
    #[pyo3(get, set)]
    pub n_blocks: usize,
    #[pyo3(get, set)]
    pub vocab_size: Option<usize>,
    #[pyo3(get, set)]
    pub eps: f32,
}

#[pymethods]
impl ModelConfig {
    #[new]
    #[pyo3(signature = (config, n_embd, n_head, n_head_kv, n_blocks, vocab_size=None, eps=1e-6))]
    pub fn new(
        config: ArchConfig,
        n_embd: usize,
        n_head: usize,
        n_head_kv: usize,
        n_blocks: usize,
        vocab_size: Option<usize>,
        eps: f32,
    ) -> Self {
        ModelConfig {
            config,
            n_embd,
            n_head,
            n_head_kv,
            n_blocks,
            vocab_size,
            eps,
        }
    }
}

use crate::io::gguf::{GGMLType, GGUFReader, GGUFValue};

pub struct GGUFLoader {
    pub reader: GGUFReader,
}

impl GGUFLoader {
    pub fn new(path: &str) -> std::io::Result<Self> {
        let reader = GGUFReader::open(path)?;
        Ok(GGUFLoader { reader })
    }

    pub fn get_metadata_string(&self, key: &str) -> Option<String> {
        match self.reader.metadata.get(key) {
            Some(GGUFValue::String(s)) => Some(s.clone()),
            _ => None,
        }
    }

    pub fn get_metadata_u32(&self, key: &str) -> Option<u32> {
        match self.reader.metadata.get(key) {
            Some(GGUFValue::Uint32(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn get_metadata_f32(&self, key: &str) -> Option<f32> {
        match self.reader.metadata.get(key) {
            Some(GGUFValue::Float32(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn infer_config(&self) -> std::io::Result<ModelConfig> {
        let arch = self
            .get_metadata_string("general.architecture")
            .unwrap_or_else(|| "llama".to_string());
        let p = format!("{}.", arch);

        println!("[*] Detectada arquitectura GGUF: {}", arch);

        let n_embd = self
            .get_metadata_u32(&format!("{}embedding_length", p))
            .or_else(|| self.get_metadata_u32("llama.embedding_length"))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("embedding_length not found for {}", arch),
                )
            })? as usize;

        let n_head = self
            .get_metadata_u32(&format!("{}attention.head_count", p))
            .or_else(|| self.get_metadata_u32(&format!("{}head_count", p)))
            .or_else(|| self.get_metadata_u32("llama.attention.head_count"))
            .or_else(|| self.get_metadata_u32("llama.head_count"))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("head_count not found for {}", arch),
                )
            })? as usize;

        let n_head_kv = self
            .get_metadata_u32(&format!("{}attention.head_count_kv", p))
            .or_else(|| self.get_metadata_u32(&format!("{}head_count_kv", p)))
            .or_else(|| self.get_metadata_u32("llama.attention.head_count_kv"))
            .or_else(|| self.get_metadata_u32("llama.head_count_kv"))
            .unwrap_or(n_head as u32) as usize;

        let n_blocks = self
            .get_metadata_u32(&format!("{}block_count", p))
            .or_else(|| self.get_metadata_u32("llama.block_count"))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("block_count not found for {}", arch),
                )
            })? as usize;

        let eps = self
            .get_metadata_f32(&format!("{}attention.layer_norm_rms_epsilon", p))
            .or_else(|| self.get_metadata_f32("llama.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);

        let rope_base = self
            .get_metadata_f32(&format!("{}rope.freq_base", p))
            .or_else(|| self.get_metadata_f32("llama.rope.freq_base"))
            .unwrap_or(10000.0);

        Ok(ModelConfig {
            config: ArchConfig {
                name: self
                    .get_metadata_string("general.name")
                    .unwrap_or_else(|| "GGUF-Model".to_string()),
                tokenizer_id: "tokenizer".to_string(),
                rope_base,
                ffn_act: "swiglu".to_string(),
                use_genomic_norm: false,
                rope_style: "split".to_string(),
                anchor_threshold: 0.1,
                ffn_anchor_threshold: 0.1,
                unpermute_weights: false,
                apply_smollm_rope_patch: false,
            },
            n_embd,
            n_head,
            n_head_kv,
            n_blocks,
            vocab_size: None,
            eps,
        })
    }

    pub fn load_genomic_llm(
        &mut self,
        config: ModelConfig,
        anchor_threshold: f32,
    ) -> std::io::Result<RustGenomicLLM> {
        let block_size = 32;

        // 1. Embeddings
        let embd_name = "token_embd.weight";
        let embd_dna = self.genomize_tensor(embd_name, block_size, -1.0)?;

        // 2. Blocks
        let mut blocks = Vec::new();
        let head_dim = config.n_embd / config.n_head;

        for i in 0..config.n_blocks {
            let p = format!("blk.{}.", i);

            // Attention
            let q_gen =
                self.genomize_tensor(&format!("{}attn_q.weight", p), block_size, anchor_threshold)?;
            let k_gen =
                self.genomize_tensor(&format!("{}attn_k.weight", p), block_size, anchor_threshold)?;
            let v_gen =
                self.genomize_tensor(&format!("{}attn_v.weight", p), block_size, anchor_threshold)?;
            let o_gen = self.genomize_tensor(
                &format!("{}attn_output.weight", p),
                block_size,
                anchor_threshold,
            )?;

            // FFN
            let gate_gen = self.genomize_tensor(
                &format!("{}ffn_gate.weight", p),
                block_size,
                anchor_threshold,
            )?;
            let up_gen =
                self.genomize_tensor(&format!("{}ffn_up.weight", p), block_size, anchor_threshold)?;
            let down_gen = self.genomize_tensor(
                &format!("{}ffn_down.weight", p),
                block_size,
                anchor_threshold,
            )?;

            // Norms
            let attn_norm = self.load_f32_tensor(&format!("{}attn_norm.weight", p))?;
            let ffn_norm = self.load_f32_tensor(&format!("{}ffn_norm.weight", p))?;

            let attn = GenomicAttention::new(
                config.n_head,
                config.n_head_kv,
                head_dim,
                attn_norm,
                config.eps,
                config.config.rope_base,
                config.config.rope_style.clone(),
            );

            blocks.push(RustGenomicBlock::new(
                i,
                attn,
                q_gen,
                k_gen,
                v_gen,
                o_gen,
                gate_gen,
                up_gen,
                down_gen,
                ffn_norm,
                config.eps,
                config.config.ffn_act.clone(),
                config.config.use_genomic_norm,
                1.0, // Default h_scale
            ));
        }

        // 3. Output
        let output_norm = self.load_f32_tensor("output_norm.weight")?;

        let lm_head_name = if self.reader.tensors.contains_key("output.weight") {
            "output.weight"
        } else {
            println!("[*] output.weight no encontrado, rehusando token_embd.weight para LM Head (Tied Embeddings)");
            "token_embd.weight"
        };

        let lm_head = self.genomize_tensor(lm_head_name, block_size, anchor_threshold)?;

        println!("[+] Organismo Genómico ensamblado exitosamente.");

        Ok(RustGenomicLLM::new(
            embd_dna,
            blocks,
            output_norm,
            lm_head,
            config.eps,
        ))
    }

    fn load_f32_tensor(&mut self, name: &str) -> std::io::Result<Vec<f32>> {
        println!("    [~] Cargando tensor de precisión: {}...", name);
        let data = self.reader.get_tensor_data(name)?;
        let info = self.reader.tensors.get(name).unwrap();

        match info.tensor_type {
            GGMLType::F32 => {
                let count = data.len() / 4;
                let mut res = vec![0.0f32; count];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        res.as_mut_ptr() as *mut u8,
                        data.len(),
                    );
                }
                Ok(res)
            }
            GGMLType::F16 => {
                let count = data.len() / 2;
                let mut res = vec![0.0f32; count];
                let f16_ptr = data.as_ptr() as *const half::f16;
                for i in 0..count {
                    unsafe {
                        res[i] = (*f16_ptr.add(i)).to_f32();
                    }
                }
                Ok(res)
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Tensor {} must be F32 or F16", name),
            )),
        }
    }

    fn genomize_tensor(
        &mut self,
        name: &str,
        block_size: usize,
        anchor_threshold: f32,
    ) -> std::io::Result<GenomicLinear> {
        println!("    [~] Genomizando tensor: {}...", name);
        let data = self.reader.get_tensor_data(name)?;
        let info = self.reader.tensors.get(name).unwrap();

        let out_features = info.shape[info.n_dims as usize - 1] as usize;
        let in_features = info.shape[0] as usize;

        let f32_data: Vec<f32> = match info.tensor_type {
            GGMLType::F32 => data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            GGMLType::F16 => {
                let count = data.len() / 2;
                let mut res = vec![0.0f32; count];
                let f16_ptr = data.as_ptr() as *const half::f16;
                for i in 0..count {
                    unsafe {
                        res[i] = (*f16_ptr.add(i)).to_f32();
                    }
                }
                res
            }
            GGMLType::Q8_0 => crate::compute::math::dequantize_q8_0_native(
                data.to_vec(),
                out_features,
                in_features,
            )
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Unsupported tensor type for genomization: {:?}",
                        info.tensor_type
                    ),
                ))
            }
        };

        let (dna, centroids, anchors_u8) =
            crate::compute::math::genomize_f32_core(&f32_data, block_size, anchor_threshold, None);
        let rmsnorm_weight = Vec::new();
        let eps = 1e-6;

        Ok(GenomicLinear::new(
            dna,
            anchors_u8,
            centroids,
            out_features,
            in_features,
            block_size,
            rmsnorm_weight,
            eps,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
    }
}

pub struct NativeLoader {
    db: Arc<Database>,
}

#[pyfunction]
#[pyo3(name = "save_genomic_model")]
pub fn save_genomic_model_py(
    path: &str,
    model: &RustGenomicLLM,
    config: &ModelConfig,
    tokenizer_path: Option<&str>,
) -> PyResult<()> {
    let tok = if let Some(p) = tokenizer_path {
        Some(tokenizers::Tokenizer::from_file(p).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?)
    } else {
        None
    };
    save_genomic_model(path, model, config, tok.as_ref())
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

pub fn save_genomic_model(
    path: &str,
    model: &RustGenomicLLM,
    config: &ModelConfig,
    tokenizer: Option<&tokenizers::Tokenizer>,
) -> std::io::Result<()> {
    let mut writer = crate::core::db::GajeDatabaseWriter::new(path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    let mut batch = writer
        .begin_batch()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    // 1. Metadata & Config
    batch
        .write_metadata(
            "config",
            &serde_json::to_string(config)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?,
        )
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    if let Some(tok) = tokenizer {
        let tok_json = tok
            .to_string(true)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        batch
            .write_metadata("tokenizer", &tok_json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    }

    // Helper zero-copy f32 -> u8
    let f32_as_u8 = |data: &[f32]| -> &[u8] {
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
    };

    let compress = |data: &[u8]| -> Vec<u8> { lz4_flex::compress_prepend_size(data) };

    fn write_linear(
        batch: &mut crate::core::db::GajeBatchWriter,
        prefix: &str,
        layer: &GenomicLinear,
    ) -> std::io::Result<()> {
        let f32_as_u8 = |data: &[f32]| -> &[u8] {
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
        };

        let compress = |data: &[u8]| -> Vec<u8> { lz4_flex::compress_prepend_size(data) };

        batch
            .write_tensor(&format!("{}.dna", prefix), &compress(&layer.database))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        batch
            .write_tensor(
                &format!("{}.centroids", prefix),
                &compress(f32_as_u8(&layer.centroids)),
            )
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        let anchors_u8 = layer.anchors_sparse_buffer();
        batch
            .write_tensor(&format!("{}.anchors", prefix), &compress(&anchors_u8))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        if !layer.bias.is_empty() {
            batch
                .write_tensor(
                    &format!("{}.bias", prefix),
                    &compress(f32_as_u8(&layer.bias)),
                )
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        }

        if !layer.precision_mask.is_empty() {
            batch
                .write_tensor(
                    &format!("{}.precision_mask", prefix),
                    &compress(&layer.precision_mask),
                )
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        }
        Ok(())
    }

    // 2. Tensors - Embeddings
    write_linear(&mut batch, "token_embd", &model.embeddings)?;

    // 3. Tensors - Blocks
    for (i, block) in model.blocks.iter().enumerate() {
        let p = format!("blk.{}.", i);
        write_linear(&mut batch, &format!("{}attn_q", p), &block.q_gen)?;
        write_linear(&mut batch, &format!("{}attn_k", p), &block.k_gen)?;
        write_linear(&mut batch, &format!("{}attn_v", p), &block.v_gen)?;
        write_linear(&mut batch, &format!("{}attn_output", p), &block.w_o)?;
        write_linear(&mut batch, &format!("{}ffn_gate", p), &block.gate_gen)?;
        write_linear(&mut batch, &format!("{}ffn_up", p), &block.up_gen)?;
        write_linear(&mut batch, &format!("{}ffn_down", p), &block.w_down)?;

        batch
            .write_tensor(
                &format!("{}attn_norm", p),
                &compress(f32_as_u8(&block.attn.rmsnorm_weight)),
            )
            .unwrap();
        batch
            .write_tensor(
                &format!("{}ffn_norm", p),
                &compress(f32_as_u8(&block.ffn_norm)),
            )
            .unwrap();
        batch
            .write_tensor(
                &format!("{}h_scale", p),
                &compress(f32_as_u8(&[block.h_scale])),
            )
            .unwrap();
    }

    // 4. Output
    write_linear(&mut batch, "lm_head", &model.lm_head)?;
    batch
        .write_tensor("output_norm", &compress(f32_as_u8(&model.output_norm)))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    // Commit the entire batch
    batch
        .commit()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    // 5. Compact database to minimize disk space
    println!("[*] Compactando base de datos genómica con compresión LZ4...");
    let _ = writer.compact();

    Ok(())
}

impl NativeLoader {
    pub fn new(path: &str) -> std::io::Result<Self> {
        let db = crate::core::db::get_or_create_db(path, false)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(NativeLoader { db })
    }

    pub fn load_config(&self) -> std::io::Result<ModelConfig> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        let table = read_txn
            .open_table(METADATA_TABLE)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        let json_str = table
            .get("config")
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "config not found"))?;

        let config: ModelConfig = serde_json::from_str(json_str.value())?;
        Ok(config)
    }

    pub fn load_tokenizer(&self) -> std::io::Result<tokenizers::Tokenizer> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        let table = read_txn
            .open_table(METADATA_TABLE)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        let json_str = table
            .get("tokenizer")
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "tokenizer not found in database",
                )
            })?;

        tokenizers::Tokenizer::from_bytes(json_str.value().as_bytes())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
    }

    fn get_tensor(txn: &ReadTransaction, key: &str) -> Vec<u8> {
        if let Ok(table) = txn.open_table(TENSOR_TABLE) {
            if let Ok(Some(val)) = table.get(key) {
                let data = val.value();
                // Decompress if it looks like LZ4
                return lz4_flex::decompress_size_prepended(data).unwrap_or_else(|_| data.to_vec());
            }
        }
        Vec::new()
    }

    fn get_tensor_f32(txn: &ReadTransaction, key: &str) -> Vec<f32> {
        let bytes = Self::get_tensor(txn, key);
        if bytes.is_empty() {
            return Vec::new();
        }
        let count = bytes.len() / 4;
        let mut res = vec![0.0f32; count];
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), res.as_mut_ptr() as *mut u8, bytes.len());
        }
        res
    }

    fn get_linear(
        txn: &ReadTransaction,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        block_size: usize,
    ) -> GenomicLinear {
        let dna = Self::get_tensor(txn, &format!("{}.dna", prefix));
        let centroids = Self::get_tensor_f32(txn, &format!("{}.centroids", prefix));
        let anchors_u8 = Self::get_tensor(txn, &format!("{}.anchors", prefix));
        let bias = Self::get_tensor_f32(txn, &format!("{}.bias", prefix));
        let precision_mask = Self::get_tensor(txn, &format!("{}.precision_mask", prefix));
        let epi_dna = Self::get_tensor(txn, &format!("{}.epi_dna", prefix));
        let epi_centroids = Self::get_tensor_f32(txn, &format!("{}.epi_centroids", prefix));
        let tri_dna = Self::get_tensor(txn, &format!("{}.tri_dna", prefix));
        let tri_centroids = Self::get_tensor_f32(txn, &format!("{}.tri_centroids", prefix));

        GenomicLinear::new(
            dna,
            anchors_u8,
            centroids,
            out_features,
            in_features,
            block_size,
            Vec::new(),
            1e-6,
            precision_mask,
            epi_dna,
            epi_centroids,
            tri_dna,
            tri_centroids,
            bias,
        )
    }

    pub fn load_llm(&self) -> std::io::Result<RustGenomicLLM> {
        let config = self.load_config()?;
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        let block_size = 32;
        let vocab_size = config.vocab_size.unwrap_or_else(|| {
            let token_embd_dna = Self::get_tensor(&read_txn, "token_embd.dna");
            token_embd_dna.len() * 4 / config.n_embd
        });

        let embeddings = Self::get_linear(
            &read_txn,
            "token_embd",
            config.n_embd,
            vocab_size,
            block_size,
        );
        let lm_head = Self::get_linear(&read_txn, "lm_head", config.n_embd, vocab_size, block_size);

        let mut output_norm = Self::get_tensor_f32(&read_txn, "output_norm");
        if output_norm.is_empty() {
            output_norm = vec![1.0f32; config.n_embd];
        }

        let mut blocks = Vec::new();
        let head_dim = config.n_embd / config.n_head;

        for i in 0..config.n_blocks {
            let p = format!("blk.{}.", i);
            let q_gen = Self::get_linear(
                &read_txn,
                &format!("{}attn_q", p),
                config.n_embd,
                config.n_head * head_dim,
                block_size,
            );
            let k_gen = Self::get_linear(
                &read_txn,
                &format!("{}attn_k", p),
                config.n_embd,
                config.n_head_kv * head_dim,
                block_size,
            );
            let v_gen = Self::get_linear(
                &read_txn,
                &format!("{}attn_v", p),
                config.n_embd,
                config.n_head_kv * head_dim,
                block_size,
            );
            let w_o = Self::get_linear(
                &read_txn,
                &format!("{}attn_output", p),
                config.n_head * head_dim,
                config.n_embd,
                block_size,
            );

            let get_out_features = |prefix: &str, in_features: usize| -> usize {
                let centroids = Self::get_tensor_f32(&read_txn, &format!("{}.centroids", prefix));
                if centroids.is_empty() {
                    return config.n_embd;
                }
                if in_features == 0 || block_size == 0 {
                    return config.n_embd;
                }
                centroids.len() / (in_features / block_size * 4)
            };

            let ffn_hidden = get_out_features(&format!("{}ffn_gate", p), config.n_embd);

            let gate_gen = Self::get_linear(
                &read_txn,
                &format!("{}ffn_gate", p),
                config.n_embd,
                ffn_hidden,
                block_size,
            );
            let up_gen = Self::get_linear(
                &read_txn,
                &format!("{}ffn_up", p),
                config.n_embd,
                ffn_hidden,
                block_size,
            );
            let w_down = Self::get_linear(
                &read_txn,
                &format!("{}ffn_down", p),
                ffn_hidden,
                config.n_embd,
                block_size,
            );

            let mut attn_norm = Self::get_tensor_f32(&read_txn, &format!("{}attn_norm", p));
            if attn_norm.is_empty() {
                attn_norm = vec![1.0f32; config.n_embd];
            }

            let mut ffn_norm = Self::get_tensor_f32(&read_txn, &format!("{}ffn_norm", p));
            if ffn_norm.is_empty() {
                ffn_norm = vec![1.0f32; config.n_embd];
            }

            let h_scale_vec = Self::get_tensor_f32(&read_txn, &format!("{}h_scale", p));
            let h_scale = if h_scale_vec.is_empty() {
                1.0f32
            } else {
                h_scale_vec[0]
            };

            let attn = GenomicAttention::new(
                config.n_head,
                config.n_head_kv,
                head_dim,
                attn_norm,
                config.eps,
                config.config.rope_base,
                config.config.rope_style.clone(),
            );

            blocks.push(RustGenomicBlock::new(
                i,
                attn,
                q_gen,
                k_gen,
                v_gen,
                w_o,
                gate_gen,
                up_gen,
                w_down,
                ffn_norm,
                config.eps,
                config.config.ffn_act.clone(),
                config.config.use_genomic_norm,
                h_scale,
            ));
        }

        Ok(RustGenomicLLM::new(
            embeddings,
            blocks,
            output_norm,
            lm_head,
            config.eps,
        ))
    }

    pub fn list_mutations(&self) -> std::io::Result<Vec<(u64, Vec<u8>)>> {
        let reader = crate::core::db::GajeDatabaseReader::new_from_db(self.db.clone());
        reader
            .list_mutations()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }

    pub fn save_mutation(
        &self,
        timestamp: u64,
        mutation: &crate::core::db::Mutation,
    ) -> std::io::Result<()> {
        let data = bincode::serialize(mutation)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(crate::core::db::MUTATIONS_TABLE)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            table
                .insert(timestamp, data.as_slice())
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        }
        write_txn
            .commit()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }
}

#[pyfunction]
#[pyo3(name = "init_born_genomic_model")]
pub fn init_born_genomic_model_py(
    path: &str,
    config: ModelConfig,
    vocab_size: usize,
) -> PyResult<RustGenomicLLM> {
    init_born_genomic_model(path, config, vocab_size)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

pub fn init_born_genomic_model(
    path: &str,
    config: ModelConfig,
    vocab_size: usize,
) -> std::io::Result<RustGenomicLLM> {
    let block_size = 32;

    let init_linear = |in_features: usize, out_features: usize| -> GenomicLinear {
        let n_elements = in_features * out_features;
        let n_blocks = n_elements / block_size;
        let dna = crate::compute::math::generate_random_dna(n_elements);
        let centroids = crate::compute::math::generate_default_centroids(n_blocks);

        GenomicLinear::new(
            dna,
            Vec::new(),
            centroids,
            out_features,
            in_features,
            block_size,
            Vec::new(),
            1e-6,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    };

    // 1. Embeddings
    let embeddings = init_linear(config.n_embd, vocab_size);

    // 2. Blocks
    let mut blocks = Vec::new();
    let head_dim = config.n_embd / config.n_head;
    let ffn_hidden = config.n_embd * 4;

    for i in 0..config.n_blocks {
        let q_gen = init_linear(config.n_embd, config.n_head * head_dim);
        let k_gen = init_linear(config.n_embd, config.n_head_kv * head_dim);
        let v_gen = init_linear(config.n_embd, config.n_head_kv * head_dim);
        let w_o = init_linear(config.n_head * head_dim, config.n_embd);

        let gate_gen = init_linear(config.n_embd, ffn_hidden);
        let up_gen = init_linear(config.n_embd, ffn_hidden);
        let w_down = init_linear(ffn_hidden, config.n_embd);

        let attn_norm = vec![1.0f32; config.n_embd];
        let ffn_norm = vec![1.0f32; config.n_embd];

        let attn = GenomicAttention::new(
            config.n_head,
            config.n_head_kv,
            head_dim,
            attn_norm,
            config.eps,
            config.config.rope_base,
            config.config.rope_style.clone(),
        );

        blocks.push(RustGenomicBlock::new(
            i,
            attn,
            q_gen,
            k_gen,
            v_gen,
            w_o,
            gate_gen,
            up_gen,
            w_down,
            ffn_norm,
            config.eps,
            config.config.ffn_act.clone(),
            config.config.use_genomic_norm,
            1.0, // Default h_scale
        ));
    }

    // 3. Output
    let output_norm = vec![1.0f32; config.n_embd];
    let lm_head = init_linear(config.n_embd, vocab_size);

    let model = RustGenomicLLM::new(embeddings, blocks, output_norm, lm_head, config.eps);

    save_genomic_model(path, &model, &config, None)?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_native_loader_reads_gaje_db() {
        let test_db_path = "test_model.gaje";
        if !Path::new(test_db_path).exists() {
            println!("Skipping test since {} does not exist.", test_db_path);
            return;
        }

        let loader = NativeLoader::new(test_db_path).expect("Failed to open database");
        let config = loader.load_config().expect("Failed to load config");
        assert_eq!(config.n_embd, 768);
        assert_eq!(config.n_blocks, 2);

        let llm = loader.load_llm().expect("Failed to load LLM");
        assert_eq!(llm.blocks.len(), 2);
        println!(
            "Successfully loaded LLM from NativeLoader! Vocab size: {}",
            llm.lm_head.out_features
        );
    }
}

#[cfg(test)]
mod gguf_tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_gguf_loader_reads_qwen2() {
        let path = "/data/data/com.termux/files/home/models/qwen2-0_5b-q8_0.gguf";
        if !Path::new(path).exists() {
            println!("Skipping test since {} does not exist.", path);
            return;
        }

        let loader = GGUFLoader::new(path).expect("Failed to open GGUF");
        let name = loader
            .get_metadata_string("general.name")
            .expect("Failed to read name");
        println!("GGUF Model Name: {}", name);
        assert!(name.contains("Qwen2") || name.contains("qwen2"));

        let n_embd = loader
            .get_metadata_u32("qwen2.embedding_length")
            .or_else(|| loader.get_metadata_u32("llama.embedding_length"))
            .expect("Failed to read n_embd");

        println!("GGUF Embedding Length: {}", n_embd);
    }
}
