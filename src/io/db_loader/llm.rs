// =============================================================================
// llm — load_llm: reconstrucción del modelo genómico completo desde la base
// =============================================================================
use crate::io::config::ModelConfig;
use crate::io::db_loader::NativeLoader;
use crate::nn::{GenomicAttention, GenomicLLM, GenomicLinear, RustGenomicBlock};
use redb::{ReadTransaction, ReadableTable};

use crate::core::db::TENSOR_TABLE;

impl NativeLoader {
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
            quantum_embeddings: None,
            gpu_layers: 0,
            use_gpu: false,
        })
    }
}
