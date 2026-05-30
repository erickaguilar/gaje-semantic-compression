use crate::core::db::TENSOR_TABLE;
use crate::nn::{GenomicAttention, GenomicLLM, GenomicLinear, RustGenomicBlock};
use redb::{Database, ReadTransaction, ReadableTable};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ArchConfig {
    #[serde(default = "default_name")]
    pub name: String,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default = "default_tokenizer")]
    pub tokenizer_id: String,
    #[serde(default = "default_rope_base")]
    pub rope_base: f32,
    #[serde(default = "default_ffn_act")]
    pub ffn_act: String,
    #[serde(default = "default_false")]
    pub use_genomic_norm: bool,
    #[serde(default = "default_rope_style")]
    pub rope_style: String,
    #[serde(default = "default_anchor_threshold")]
    pub anchor_threshold: f32,
    #[serde(default = "default_anchor_threshold")]
    pub ffn_anchor_threshold: f32,
    #[serde(default = "default_false")]
    pub unpermute_weights: bool,
    #[serde(default = "default_false")]
    pub apply_smollm_rope_patch: bool,
    #[serde(default = "default_dni")]
    pub dni: String,
    #[serde(default = "default_state")]
    pub state: String,
}

#[cfg_attr(feature = "python", pymethods)]
impl ArchConfig {
    #[cfg(feature = "python")]
    #[new]
    #[pyo3(signature = (name = "GAJE-Model".to_string(), version = "1.0.0-alpha".to_string(), tokenizer_id = "gpt2".to_string(), rope_base = 10000.0, ffn_act = "swiglu".to_string(), use_genomic_norm = false, rope_style = "split".to_string(), anchor_threshold = 0.1, ffn_anchor_threshold = 0.1, unpermute_weights = false, apply_smollm_rope_patch = false, dni = "".to_string(), state = "stable".to_string()))]
    pub fn py_new(
        name: String,
        version: String,
        tokenizer_id: String,
        rope_base: f32,
        ffn_act: String,
        use_genomic_norm: bool,
        rope_style: String,
        anchor_threshold: f32,
        ffn_anchor_threshold: f32,
        unpermute_weights: bool,
        apply_smollm_rope_patch: bool,
        dni: String,
        state: String,
    ) -> Self {
        let actual_dni = if dni.is_empty() { default_dni() } else { dni };
        ArchConfig {
            name,
            version,
            tokenizer_id,
            rope_base,
            ffn_act,
            use_genomic_norm,
            rope_style,
            anchor_threshold,
            ffn_anchor_threshold,
            unpermute_weights,
            apply_smollm_rope_patch,
            dni: actual_dni,
            state,
        }
    }
}

fn default_name() -> String {
    "GAJE-Model".to_string()
}
fn default_version() -> String {
    "1.0.0-alpha".to_string()
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
fn default_state() -> String {
    "stable".to_string()
}

fn default_dni() -> String {
    use chrono::Utc;
    use rand::Rng;
    let now = Utc::now().format("%Y%m%d%H%M%S").to_string();
    let rand_id: u32 = rand::thread_rng().gen();
    format!("GAJE-DNI-{}-{:08X}", now, rand_id)
}

pub fn py_new_dni() -> String {
    default_dni()
}

#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ModelConfig {
    pub config: ArchConfig,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_blocks: usize,
    pub vocab_size: Option<usize>,
    pub eps: f32,
}

#[cfg_attr(feature = "python", pymethods)]
impl ModelConfig {
    #[cfg(feature = "python")]
    #[new]
    #[pyo3(signature = (config, n_embd, n_head, n_head_kv, n_blocks, vocab_size=None, eps=1e-6))]
    pub fn py_new(
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
        Ok(GGUFLoader {
            reader: GGUFReader::open(path)?,
        })
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
        let n_embd = self
            .get_metadata_u32(&format!("{}embedding_length", p))
            .or_else(|| self.get_metadata_u32("llama.embedding_length"))
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, "embedding_length not found")
            })? as usize;
        let n_head = self
            .get_metadata_u32(&format!("{}attention.head_count", p))
            .or_else(|| self.get_metadata_u32("llama.attention.head_count"))
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, "head_count not found")
            })? as usize;
        let n_head_kv = self
            .get_metadata_u32(&format!("{}attention.head_count_kv", p))
            .or_else(|| self.get_metadata_u32("llama.attention.head_count_kv"))
            .unwrap_or(n_head as u32) as usize;
        let n_blocks = self
            .get_metadata_u32(&format!("{}block_count", p))
            .or_else(|| self.get_metadata_u32("llama.block_count"))
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, "block_count not found")
            })? as usize;
        let eps = self
            .get_metadata_f32(&format!("{}attention.layer_norm_rms_epsilon", p))
            .unwrap_or(1e-6);
        let rope_base = self
            .get_metadata_f32(&format!("{}rope.freq_base", p))
            .unwrap_or(10000.0);
        Ok(ModelConfig {
            config: ArchConfig {
                name: self
                    .get_metadata_string("general.name")
                    .unwrap_or_else(|| "GGUF-Model".to_string()),
                version: "1.0.0-alpha".to_string(),
                tokenizer_id: "tokenizer".to_string(),
                rope_base,
                ffn_act: "swiglu".to_string(),
                use_genomic_norm: false,
                rope_style: "split".to_string(),
                anchor_threshold: 0.1,
                ffn_anchor_threshold: 0.1,
                unpermute_weights: false,
                apply_smollm_rope_patch: false,
                dni: default_dni(),
                state: "stable".to_string(),
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
        &self,
        config: ModelConfig,
        anchor_threshold: f32,
    ) -> std::io::Result<GenomicLLM> {
        let block_size = 32;
        let embd_dna = self.genomize_tensor("token_embd.weight", block_size, -1.0)?;
        let mut blocks = Vec::new();
        let head_dim = config.n_embd / config.n_head;
        for i in 0..config.n_blocks {
            let p = format!("blk.{}.", i);
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
                1.0,
            ));
        }
        let output_norm = self.load_f32_tensor("output_norm.weight")?;
        let lm_head_name = if self.reader.tensors.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let lm_head = self.genomize_tensor(lm_head_name, block_size, anchor_threshold)?;
        Ok(GenomicLLM {
            embeddings: embd_dna,
            blocks,
            output_norm,
            lm_head,
            eps: config.eps,
            topology: None,
        })
    }

    fn load_f32_tensor(&self, name: &str) -> std::io::Result<Vec<f32>> {
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
        &self,
        name: &str,
        block_size: usize,
        anchor_threshold: f32,
    ) -> std::io::Result<GenomicLinear> {
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
                let ptr = data.as_ptr() as *const half::f16;
                for i in 0..count {
                    unsafe {
                        res[i] = (*ptr.add(i)).to_f32();
                    }
                }
                res
            }
            GGMLType::Q8_0 => {
                crate::compute::math::dequantize_q8_0_core(&data, out_features, in_features)
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Unsupported tensor type",
                ))
            }
        };
        let (dna, centroids, anchors_u8) =
            crate::compute::math::genomize_f32_core(&f32_data, block_size, anchor_threshold, None);
        Ok(GenomicLinear::new(
            dna,
            anchors_u8,
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
        ))
    }
}

#[cfg_attr(feature = "python", pyclass)]
pub struct NativeLoader {
    pub db: Arc<Database>,
}

#[cfg_attr(feature = "python", pymethods)]
impl NativeLoader {
    #[cfg(feature = "python")]
    #[new]
    pub fn py_new(path: &str) -> PyResult<Self> {
        Self::new(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    #[cfg(feature = "python")]
    pub fn py_load_config(&self) -> PyResult<ModelConfig> {
        self.load_config()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[cfg(feature = "python")]
    pub fn py_load_llm(&self) -> PyResult<GenomicLLM> {
        self.load_llm()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

impl NativeLoader {
    pub fn new(path: &str) -> std::io::Result<Self> {
        Self::new_with_mode(path, true)
    }

    pub fn new_with_mode(path: &str, read_only: bool) -> std::io::Result<Self> {
        Ok(NativeLoader {
            db: crate::core::db::get_or_create_db(path, read_only)
                .map_err(std::io::Error::other)?,
        })
    }

    pub fn load_config(&self) -> std::io::Result<ModelConfig> {
        let reader = crate::core::db::GajeDatabaseReader {
            db: self.db.clone(),
        };
        let json_str = reader
            .read_metadata_core("config")
            .map_err(std::io::Error::other)?;
        Ok(serde_json::from_str(&json_str)?)
    }
    pub fn load_llm(&self) -> std::io::Result<GenomicLLM> {
        let config = self.load_config()?;
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let block_size = 32;
        let vocab_size = config.vocab_size.unwrap_or_else(|| {
            let dna = Self::get_tensor(&read_txn, "token_embd.dna");
            dna.len() * 4 / config.n_embd
        });
        let embeddings = self.get_linear(
            &read_txn,
            "token_embd",
            config.n_embd,
            vocab_size,
            block_size,
        );
        let lm_head = self.get_linear(&read_txn, "lm_head", config.n_embd, vocab_size, block_size);
        let output_norm = {
            let n = Self::get_tensor_f32(&read_txn, "output_norm");
            if n.is_empty() {
                vec![1.0f32; config.n_embd]
            } else {
                n
            }
        };
        let mut blocks = Vec::new();
        let head_dim = config.n_embd / config.n_head;
        for i in 0..config.n_blocks {
            let p = format!("blk.{}.", i);
            let q_gen = self.get_linear(
                &read_txn,
                &format!("{}attn_q", p),
                config.n_embd,
                config.n_head * head_dim,
                block_size,
            );
            let k_gen = self.get_linear(
                &read_txn,
                &format!("{}attn_k", p),
                config.n_embd,
                config.n_head_kv * head_dim,
                block_size,
            );
            let v_gen = self.get_linear(
                &read_txn,
                &format!("{}attn_v", p),
                config.n_embd,
                config.n_head_kv * head_dim,
                block_size,
            );
            let w_o = self.get_linear(
                &read_txn,
                &format!("{}attn_output", p),
                config.n_head * head_dim,
                config.n_embd,
                block_size,
            );
            let ffn_h = {
                let c = Self::get_tensor_f32(&read_txn, &format!("{}ffn_gate.centroids", p));
                if c.is_empty() {
                    config.n_embd * 4
                } else {
                    c.len() / (config.n_embd / block_size * 4)
                }
            };
            let gate_gen = self.get_linear(
                &read_txn,
                &format!("{}ffn_gate", p),
                config.n_embd,
                ffn_h,
                block_size,
            );
            let up_gen = self.get_linear(
                &read_txn,
                &format!("{}ffn_up", p),
                config.n_embd,
                ffn_h,
                block_size,
            );
            let w_down = self.get_linear(
                &read_txn,
                &format!("{}ffn_down", p),
                ffn_h,
                config.n_embd,
                block_size,
            );
            let attn_norm = {
                let n = Self::get_tensor_f32(&read_txn, &format!("{}attn_norm", p));
                if n.is_empty() {
                    vec![1.0f32; config.n_embd]
                } else {
                    n
                }
            };
            let ffn_norm = {
                let n = Self::get_tensor_f32(&read_txn, &format!("{}ffn_norm", p));
                if n.is_empty() {
                    vec![1.0f32; config.n_embd]
                } else {
                    n
                }
            };
            let h_scale = {
                let s = Self::get_tensor_f32(&read_txn, &format!("{}h_scale", p));
                if s.is_empty() {
                    1.0f32
                } else {
                    s[0]
                }
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
        Ok(GenomicLLM {
            embeddings,
            blocks,
            output_norm,
            lm_head,
            eps: config.eps,
            topology: None,
        })
    }
    fn get_tensor(txn: &ReadTransaction, key: &str) -> Vec<u8> {
        if let Ok(t) = txn.open_table(TENSOR_TABLE) {
            if let Ok(Some(v)) = t.get(key) {
                return lz4_flex::decompress_size_prepended(v.value())
                    .unwrap_or_else(|_| v.value().to_vec());
            }
        }
        Vec::new()
    }
    fn get_tensor_f32(txn: &ReadTransaction, key: &str) -> Vec<f32> {
        let b = Self::get_tensor(txn, key);
        if b.is_empty() {
            return Vec::new();
        }
        let mut r = vec![0.0f32; b.len() / 4];
        unsafe {
            std::ptr::copy_nonoverlapping(b.as_ptr(), r.as_mut_ptr() as *mut u8, b.len());
        }
        r
    }
    fn get_linear(
        &self,
        txn: &ReadTransaction,
        p: &str,
        i_f: usize,
        o_f: usize,
        b_s: usize,
    ) -> GenomicLinear {
        let dna = Self::get_tensor(txn, &format!("{}.dna", p));
        let centroids = Self::get_tensor_f32(txn, &format!("{}.centroids", p));
        let anchors = Self::get_tensor(txn, &format!("{}.anchors", p));
        let bias = Self::get_tensor_f32(txn, &format!("{}.bias", p));
        let mask = Self::get_tensor(txn, &format!("{}.precision_mask", p));
        GenomicLinear::new(
            dna,
            anchors,
            centroids,
            o_f,
            i_f,
            b_s,
            Vec::new(),
            1e-6,
            mask,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            bias,
        )
    }
    pub fn load_tokenizer(&self) -> std::io::Result<GajeTokenizer> {
        let reader = crate::core::db::GajeDatabaseReader {
            db: self.db.clone(),
        };
        let json_str = reader
            .read_metadata_core("tokenizer")
            .map_err(std::io::Error::other)?;
        GajeTokenizer::from_bytes(json_str.as_bytes())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
    }
    pub fn list_mutations(&self) -> std::io::Result<Vec<(u64, Vec<u8>)>> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        if let Ok(table) = read_txn.open_table(crate::core::db::MUTATIONS_TABLE) {
            let mut res = Vec::new();
            for (ts, val) in table
                .iter()
                .map_err(|e| std::io::Error::other(e.to_string()))?
                .flatten()
            {
                res.push((ts.value(), val.value().to_vec()));
            }
            Ok(res)
        } else {
            Ok(Vec::new())
        }
    }
}

pub fn load_topology(path: &str) -> std::io::Result<crate::core::topology::CentroidGraph> {
    let file = std::fs::File::open(path)?;
    let topo: crate::core::topology::CentroidGraph = serde_json::from_reader(file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    Ok(topo)
}

pub fn save_genomic_model(
    path: &str,
    model: &GenomicLLM,
    config: &ModelConfig,
    tokenizer: Option<&GajeTokenizer>,
) -> std::io::Result<()> {
    let mut writer =
        crate::core::db::GajeDatabaseWriter::new(path).map_err(std::io::Error::other)?;
    let mut batch = writer.begin_batch().map_err(std::io::Error::other)?;
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
        b.write_tensor(&format!("{}.dna", p), &compress(&l.database))
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

use crate::core::tokenizer::GajeTokenizer;

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
        ));
    }
    let model = GenomicLLM {
        embeddings,
        blocks,
        output_norm: vec![1.0; config.n_embd],
        lm_head: init_l(config.n_embd, vocab_size),
        eps: config.eps,
        topology: None,
    };
    save_genomic_model(path, &model, &config, None)?;
    Ok(model)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "save_genomic_model", signature = (path, model, config, tokenizer_path=None))]
pub fn save_genomic_model_py(
    path: &str,
    model: &crate::nn::llm::GenomicLLM,
    config: &ModelConfig,
    tokenizer_path: Option<&str>,
) -> PyResult<()> {
    let tok = if let Some(p) = tokenizer_path {
        Some(
            GajeTokenizer::from_file(p)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?,
        )
    } else {
        None
    };
    save_genomic_model(path, model, config, tok.as_ref())
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "init_born_genomic_model")]
pub fn init_born_genomic_model_py(
    path: &str,
    config: ModelConfig,
    vocab_size: usize,
) -> PyResult<crate::nn::llm::GenomicLLM> {
    let inner = init_born_genomic_model(path, config, vocab_size)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(inner)
}
