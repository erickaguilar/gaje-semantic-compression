use serde::Deserialize;
use redb::{Database, ReadTransaction};
use crate::nn::{GenomicLinear, RustGenomicBlock, GenomicAttention, RustGenomicLLM};
use crate::db::{TENSOR_TABLE, METADATA_TABLE};

#[derive(Deserialize, Debug, Clone)]
pub struct ArchConfig {
    #[serde(default = "default_name")]
    pub name: String,
    #[serde(default = "default_tokenizer")]
    pub tokenizer_id: String,
    #[serde(default = "default_rope_base")]
    pub rope_base: f32,
    #[serde(default = "default_ffn_act")]
    pub ffn_act: String,
    #[serde(default = "default_false")]
    pub use_genomic_norm: bool,
}

fn default_name() -> String { "GAJE-Model".to_string() }
fn default_tokenizer() -> String { "gpt2".to_string() }
fn default_rope_base() -> f32 { 10000.0 }
fn default_ffn_act() -> String { "swiglu".to_string() }
fn default_false() -> bool { false }

#[derive(Deserialize, Debug, Clone)]
pub struct ModelConfig {
    pub config: ArchConfig,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_blocks: usize,
    pub vocab_size: Option<usize>,
    pub eps: f32,
}

pub struct NativeLoader {
    db: Database,
}

impl NativeLoader {
    pub fn new(path: &str) -> std::io::Result<Self> {
        let db = Database::open(path).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        Ok(NativeLoader { db })
    }

    pub fn load_config(&self) -> std::io::Result<ModelConfig> {
        let read_txn = self.db.begin_read().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        let table = read_txn.open_table(METADATA_TABLE).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        
        let json_str = table.get("config").map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "config not found"))?;
        
        let config: ModelConfig = serde_json::from_str(json_str.value())?;
        Ok(config)
    }

    pub fn load_tokenizer(&self) -> std::io::Result<tokenizers::Tokenizer> {
        let read_txn = self.db.begin_read().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        let table = read_txn.open_table(METADATA_TABLE).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        
        let json_str = table.get("tokenizer").map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "tokenizer not found in database"))?;
            
        tokenizers::Tokenizer::from_bytes(json_str.value().as_bytes())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
    }

    fn get_tensor(txn: &ReadTransaction, key: &str) -> Vec<u8> {
        if let Ok(table) = txn.open_table(TENSOR_TABLE) {
            if let Ok(Some(val)) = table.get(key) {
                return val.value().to_vec();
            }
        }
        Vec::new()
    }
    
    fn get_tensor_f32(txn: &ReadTransaction, key: &str) -> Vec<f32> {
        let bytes = Self::get_tensor(txn, key);
        if bytes.is_empty() { return Vec::new(); }
        let count = bytes.len() / 4;
        let mut res = vec![0.0f32; count];
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), res.as_mut_ptr() as *mut u8, bytes.len());
        }
        res
    }

    fn get_linear(txn: &ReadTransaction, prefix: &str, in_features: usize, out_features: usize, block_size: usize) -> GenomicLinear {
        let dna = Self::get_tensor(txn, &format!("{}.dna", prefix));
        let centroids = Self::get_tensor_f32(txn, &format!("{}.centroids", prefix));
        
        let anchors_bytes = Self::get_tensor(txn, &format!("{}.anchors", prefix));
        let mut anchors = Vec::new();
        if !anchors_bytes.is_empty() {
            let count = anchors_bytes.len() / 2;
            anchors = vec![0.0f32; count];
            for i in 0..count {
                let mut buf = [0u8; 2];
                buf.copy_from_slice(&anchors_bytes[i*2..(i+1)*2]);
                anchors[i] = half::f16::from_le_bytes(buf).to_f32();
            }
        }

        let bias = Self::get_tensor_f32(txn, &format!("{}.bias", prefix));
        let precision_mask = Self::get_tensor(txn, &format!("{}.precision_mask", prefix));
        let epi_dna = Self::get_tensor(txn, &format!("{}.epi_dna", prefix));
        let epi_centroids = Self::get_tensor_f32(txn, &format!("{}.epi_centroids", prefix));
        let tri_dna = Self::get_tensor(txn, &format!("{}.tri_dna", prefix));
        let tri_centroids = Self::get_tensor_f32(txn, &format!("{}.tri_centroids", prefix));

        GenomicLinear::new(
            dna,
            anchors,
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
        let read_txn = self.db.begin_read().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        
        let block_size = 32;
        let vocab_size = config.vocab_size.unwrap_or_else(|| {
            let token_embd_dna = Self::get_tensor(&read_txn, "token_embd.dna");
            token_embd_dna.len() * 4 / config.n_embd
        });

        let embeddings = Self::get_linear(&read_txn, "token_embd", config.n_embd, vocab_size, block_size);
        let lm_head = Self::get_linear(&read_txn, "lm_head", config.n_embd, vocab_size, block_size);
        
        let mut output_norm = Self::get_tensor_f32(&read_txn, "output_norm");
        if output_norm.is_empty() {
            output_norm = vec![1.0f32; config.n_embd];
        }
        
        let mut blocks = Vec::new();
        let head_dim = config.n_embd / config.n_head;
        
        for i in 0..config.n_blocks {
            let p = format!("blk.{}.", i);
            let q_gen = Self::get_linear(&read_txn, &format!("{}attn_q", p), config.n_embd, config.n_head * head_dim, block_size);
            let k_gen = Self::get_linear(&read_txn, &format!("{}attn_k", p), config.n_embd, config.n_head_kv * head_dim, block_size);
            let v_gen = Self::get_linear(&read_txn, &format!("{}attn_v", p), config.n_embd, config.n_head_kv * head_dim, block_size);
            let w_o = Self::get_linear(&read_txn, &format!("{}attn_output", p), config.n_head * head_dim, config.n_embd, block_size);
            
            let get_out_features = |prefix: &str, in_features: usize| -> usize {
                let centroids = Self::get_tensor_f32(&read_txn, &format!("{}.centroids", prefix));
                if centroids.is_empty() { return config.n_embd; }
                if in_features == 0 || block_size == 0 { return config.n_embd; }
                centroids.len() / (in_features / block_size * 4)
            };
            
            let ffn_hidden = get_out_features(&format!("{}ffn_gate", p), config.n_embd);
            
            let gate_gen = Self::get_linear(&read_txn, &format!("{}ffn_gate", p), config.n_embd, ffn_hidden, block_size);
            let up_gen = Self::get_linear(&read_txn, &format!("{}ffn_up", p), config.n_embd, ffn_hidden, block_size);
            let w_down = Self::get_linear(&read_txn, &format!("{}ffn_down", p), ffn_hidden, config.n_embd, block_size);

            let mut attn_norm = Self::get_tensor_f32(&read_txn, &format!("{}attn_norm", p));
            if attn_norm.is_empty() { attn_norm = vec![1.0f32; config.n_embd]; }
            
            let mut ffn_norm = Self::get_tensor_f32(&read_txn, &format!("{}ffn_norm", p));
            if ffn_norm.is_empty() { ffn_norm = vec![1.0f32; config.n_embd]; }

            let attn = GenomicAttention::new(
                config.n_head,
                config.n_head_kv,
                head_dim,
                attn_norm,
                config.eps,
                config.config.rope_base,
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
        println!("Successfully loaded LLM from NativeLoader! Vocab size: {}", llm.lm_head.out_features);
    }
}
