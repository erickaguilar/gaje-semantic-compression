use crate::archive::GAJEArchive;
use crate::nn::{GenomicLinear, RustGenomicBlock, GenomicAttention, RustGenomicLLM};
use std::collections::HashMap;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct ModelConfig {
    pub name: String,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_blocks: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub vocab_size: usize,
    pub block_size: usize,
}

pub struct NativeLoader {
    pub archive: GAJEArchive,
}

impl NativeLoader {
    pub fn new(path: &str) -> std::io::Result<Self> {
        let archive = GAJEArchive::load(path)?;
        Ok(NativeLoader { archive })
    }

    pub fn get_linear(&self, prefix: &str, in_features: usize, out_features: usize, block_size: usize) -> GenomicLinear {
        let dna = self.archive.entries.get(&format!("{}.dna", prefix))
            .map(|e| e.dna.clone())
            .unwrap_or_else(Vec::new);
        
        let centroids = self.archive.codebook.get("centroids")
            .cloned()
            .unwrap_or_else(Vec::new);

        let bias = self.archive.entries.get(&format!("{}.bias", prefix))
            .map(|e| {
                let mut b = vec![0.0f32; out_features];
                if e.dna.len() == out_features * 4 {
                    for i in 0..out_features {
                        let mut buf = [0u8; 4];
                        buf.copy_from_slice(&e.dna[i*4..(i+1)*4]);
                        b[i] = f32::from_le_bytes(buf);
                    }
                }
                b
            })
            .unwrap_or_else(Vec::new);

        let anchors = Vec::new(); // TODO: Load from archive if exists

        GenomicLinear::new(
            dna,
            anchors,
            centroids,
            out_features,
            in_features,
            block_size,
            Vec::new(), // rmsnorm_weight
            1e-6,      // eps
            Vec::new(), // precision_mask
            Vec::new(), // epi_dna
            Vec::new(), // epi_centroids
            Vec::new(), // tri_dna
            Vec::new(), // tri_centroids
            bias,
        )
    }

    fn get_vec_f32(&self, label: &str) -> Vec<f32> {
        self.archive.entries.get(label)
            .map(|e| {
                let count = e.dna.len() / 4;
                let mut res = vec![0.0f32; count];
                for i in 0..count {
                    let mut buf = [0u8; 4];
                    buf.copy_from_slice(&e.dna[i*4..(i+1)*4]);
                    res[i] = f32::from_le_bytes(buf);
                }
                res
            })
            .unwrap_or_else(Vec::new)
    }

    pub fn load_llm(&self, config: &ModelConfig) -> RustGenomicLLM {
        let embeddings = self.get_linear("token_embd", config.n_embd, config.vocab_size, config.block_size);
        let lm_head = self.get_linear("lm_head", config.n_embd, config.vocab_size, config.block_size);
        let output_norm = self.get_vec_f32("output_norm");
        
        let mut blocks = Vec::new();
        for i in 0..config.n_blocks {
            let prefix = format!("blk.{}.", i);
            let q_gen = self.get_linear(&format!("{}attn_q", prefix), config.n_embd, config.n_embd, config.block_size);
            let k_gen = self.get_linear(&format!("{}attn_k", prefix), config.n_embd, (config.n_head_kv * config.n_embd) / config.n_head, config.block_size);
            let v_gen = self.get_linear(&format!("{}attn_v", prefix), config.n_embd, (config.n_head_kv * config.n_embd) / config.n_head, config.block_size);
            let w_o = self.get_linear(&format!("{}attn_output", prefix), config.n_embd, config.n_embd, config.block_size);
            
            let gate_gen = self.get_linear(&format!("{}ffn_gate", prefix), config.n_embd, (config.n_embd * 8 / 3), config.block_size);
            let up_gen = self.get_linear(&format!("{}ffn_up", prefix), config.n_embd, (config.n_embd * 8 / 3), config.block_size);
            let w_down = self.get_linear(&format!("{}ffn_down", prefix), (config.n_embd * 8 / 3), config.n_embd, config.block_size);

            let attn_norm = self.get_vec_f32(&format!("{}attn_norm", prefix));
            let ffn_norm = self.get_vec_f32(&format!("{}ffn_norm", prefix));

            let attn = GenomicAttention::new(
                config.n_head,
                config.n_head_kv,
                config.n_embd / config.n_head,
                attn_norm,
                config.eps,
                config.rope_base,
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
                "swiglu".to_string(),
                false,
            ));
        }

        RustGenomicLLM::new(
            embeddings,
            blocks,
            output_norm,
            lm_head,
            config.eps,
        )
    }
}
