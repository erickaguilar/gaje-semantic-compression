// =============================================================================
// metadata — lecturas de metadatos e inferencia de configuración del modelo
// =============================================================================
use crate::io::config::{default_dni, ArchConfig, ModelConfig};
use crate::io::gguf::GGUFValue;
use super::GGUFLoader;

impl GGUFLoader {
    pub fn new(path: &str) -> std::io::Result<Self> {
        Ok(GGUFLoader {
            reader: crate::io::gguf::GGUFReader::open(path)?,
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

    /// Extrae el tokenizador BPE nativo GTOK directamente desde los metadatos GGUF
    pub fn extract_gtok_tokenizer(&self) -> Option<crate::core::gtok::GtokNativeTokenizer> {
        use crate::core::gtok::GtokNativeTokenizer;
        use std::collections::HashMap;

        let tokens_val = self.reader.metadata.get("tokenizer.ggml.tokens")?;
        let tokens_arr = match tokens_val {
            GGUFValue::Array(arr) => arr,
            _ => return None,
        };

        let mut vocab = Vec::with_capacity(tokens_arr.len());
        let mut token_to_id = HashMap::with_capacity(tokens_arr.len());

        for (id, val) in tokens_arr.iter().enumerate() {
            if let GGUFValue::String(tok) = val {
                vocab.push(tok.clone());
                token_to_id.insert(tok.clone(), id as u32);
            } else {
                return None;
            }
        }

        let mut merges = Vec::new();
        let mut merges_map = HashMap::new();

        if let Some(GGUFValue::Array(merges_arr)) = self.reader.metadata.get("tokenizer.ggml.merges") {
            for val in merges_arr {
                if let GGUFValue::String(m_str) = val {
                    let parts: Vec<&str> = m_str.split_whitespace().collect();
                    if parts.len() == 2 {
                        let left_str = parts[0];
                        let right_str = parts[1];
                        let target_str = format!("{}{}", left_str, right_str);

                        if let (Some(&left_id), Some(&right_id)) =
                            (token_to_id.get(left_str), token_to_id.get(right_str))
                        {
                            if let Some(&target_id) = token_to_id.get(&target_str) {
                                merges.push((left_id, right_id, target_id));
                                merges_map.insert((left_id, right_id), target_id);
                            }
                        }
                    }
                }
            }
        }

        let bos_id = self.get_metadata_u32("tokenizer.ggml.bos_token_id").unwrap_or(0);
        let eos_id = self.get_metadata_u32("tokenizer.ggml.eos_token_id").unwrap_or(151645);
        let pad_id = self.get_metadata_u32("tokenizer.ggml.padding_token_id").unwrap_or(eos_id);
        let unk_id = 0;

        let mut extra_stop_ids = Vec::new();
        if !extra_stop_ids.contains(&eos_id) {
            extra_stop_ids.push(eos_id);
        }
        for stop_str in &["<|im_end|>", "<|endoftext|>"] {
            if let Some(&sid) = token_to_id.get(*stop_str) {
                if !extra_stop_ids.contains(&sid) {
                    extra_stop_ids.push(sid);
                }
            }
        }

        Some(GtokNativeTokenizer {
            vocab,
            token_to_id,
            merges,
            merges_map,
            bos_id,
            eos_id,
            unk_id,
            pad_id,
            extra_stop_ids,
            version: crate::core::gtok::GTOK_VERSION,
            flags: 0,
        })
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
            .or_else(|| self.get_metadata_f32("llama.rope.freq_base"))
            .unwrap_or(10000.0);
        let name = self
            .get_metadata_string("general.name")
            .unwrap_or_else(|| "GGUF-Model".to_string());
        let is_smollm2 = name.to_lowercase().contains("smollm2");
        let actual_rope_base = rope_base; // Usamos el valor real del GGUF (100k)
        let rope_style = if arch == "qwen2" || arch == "llama" {
            "split".to_string()
        } else {
            "interleaved".to_string()
        };

        Ok(ModelConfig {
            config: ArchConfig {
                name,
                version: env!("CARGO_PKG_VERSION").to_string(),
                tokenizer_id: "tokenizer".to_string(),
                rope_base: actual_rope_base,
                ffn_act: "swiglu".to_string(),
                use_genomic_norm: false,
                rope_style,
                anchor_threshold: 0.1,
                ffn_anchor_threshold: 0.1,
                rna_threshold: 0.5,
                unpermute_weights: arch == "llama", // Solo Llama requiere unpermute; Qwen2 ya usa split directo
                apply_smollm_rope_patch: is_smollm2,
                tie_word_embeddings: false,
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
}
