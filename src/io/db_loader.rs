use crate::core::db::TENSOR_TABLE;
use crate::core::tokenizer::GajeTokenizer;
use crate::io::config::ModelConfig;
use crate::nn::{GenomicAttention, GenomicLLM, GenomicLinear, RustGenomicBlock};
use redb::{Database, ReadTransaction, ReadableTable};
use std::sync::Arc;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg_attr(feature = "python", pyclass)]
pub struct NativeLoader {
    pub db: Arc<Database>,
}

impl NativeLoader {
    pub fn new(path: &str) -> std::io::Result<Self> {
        Self::new_with_mode(path, true)
    }

    pub fn new_with_mode(path: &str, read_only: bool) -> std::io::Result<Self> {
        Ok(NativeLoader {
            db: crate::core::db::get_or_create_db(path, read_only).map_err(std::io::Error::other)?,
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
            let has_fused_qkv =
                !Self::get_tensor(&read_txn, &format!("{}attn_qkv.dna", p)).is_empty();
            let (q_gen, k_gen, v_gen, fused_qkv) = if has_fused_qkv {
                let qkv_out_features = config.n_head * head_dim + 2 * config.n_head_kv * head_dim;
                let qkv_linear = self.get_linear(
                    &read_txn,
                    &format!("{}attn_qkv", p),
                    config.n_embd,
                    qkv_out_features,
                    block_size,
                );
                (
                    GenomicLinear::empty(),
                    GenomicLinear::empty(),
                    GenomicLinear::empty(),
                    Some(qkv_linear),
                )
            } else {
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
                (q_gen, k_gen, v_gen, None)
            };

            let w_o = self.get_linear(
                &read_txn,
                &format!("{}attn_output", p),
                config.n_head * head_dim,
                config.n_embd,
                block_size,
            );

            let has_fused_gate_up =
                !Self::get_tensor(&read_txn, &format!("{}ffn_gate_up.dna", p)).is_empty();
            let (gate_gen, up_gen, w_down, fused_gate_up) = if has_fused_gate_up {
                let c = Self::get_tensor_f32(&read_txn, &format!("{}ffn_gate_up.centroids", p));
                let n_b = config.n_embd / block_size;
                let total_rows = if c.is_empty() {
                    config.n_embd * 8
                } else {
                    c.len() / (n_b * 16)
                };
                let ffn_h = total_rows / 2;
                let gate_up_linear = self.get_linear(
                    &read_txn,
                    &format!("{}ffn_gate_up", p),
                    config.n_embd,
                    total_rows,
                    block_size,
                );
                let w_down = self.get_linear(
                    &read_txn,
                    &format!("{}ffn_down", p),
                    ffn_h,
                    config.n_embd,
                    block_size,
                );
                (
                    GenomicLinear::empty(),
                    GenomicLinear::empty(),
                    w_down,
                    Some(gate_up_linear),
                )
            } else {
                let ffn_h = {
                    let dna = Self::get_tensor(&read_txn, &format!("{}ffn_gate.dna", p));
                    let c = Self::get_tensor_f32(&read_txn, &format!("{}ffn_gate.centroids", p));
                    if c.is_empty() {
                        config.n_embd * 4
                    } else {
                        let is_4bit = c.len() >= dna.len();
                        if is_4bit {
                            dna.len() * 2 / config.n_embd
                        } else {
                            dna.len() * 4 / config.n_embd
                        }
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
                (gate_gen, up_gen, w_down, None)
            };
            let attn_norm = {
                let n = Self::get_tensor_f32(&read_txn, &format!("{}attn_norm", p));
                if n.is_empty() {
                    eprintln!("[Loader Warning] Missing attn_norm for block {}", i);
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
            let mut block = RustGenomicBlock::new(
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
                config.config.rna_threshold,
            );
            block.fused_qkv = fused_qkv;
            block.fused_gate_up = fused_gate_up;
            blocks.push(block);
        }
        Ok(GenomicLLM {
            embeddings,
            blocks,
            output_norm,
            lm_head,
            eps: config.eps,
            k_wta_ratio: 0.50,
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

        // Inferencia robusta de profundidad de bits basada en el tamaño real del buffer DNA
        let n_elements = i_f * o_f;
        let expected_2bit = (n_elements + 3) / 4;
        let expected_4bit = (n_elements + 1) / 2;

        let bit_depth = if dna.len() == n_elements * 4 {
            32
        } else if dna.len() == expected_4bit {
            4
        } else if dna.len() == expected_2bit {
            2
        } else {
            panic!("[Loader Critical] Tamaño de buffer DNA ({}) para capa '{}' no coincide con 2-bit ({}) ni 4-bit ({})",
                    dna.len(), p, expected_2bit, expected_4bit);
        };

        GenomicLinear::new(
            dna,
            anchors,
            centroids,
            o_f,
            i_f,
            b_s,
            Vec::new(), // explicitly ensure no internal norm
            1e-6,
            mask,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            bias,
            bit_depth,
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