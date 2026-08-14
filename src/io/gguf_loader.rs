use crate::io::config::{default_dni, ArchConfig, ModelConfig};
use crate::io::gguf::{GGMLType, GGUFReader, GGUFValue};
use crate::nn::{GenomicAttention, GenomicLLM, GenomicLinear, RustGenomicBlock};

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
                version: "1.0.0-alpha".to_string(),
                tokenizer_id: "tokenizer".to_string(),
                rope_base: actual_rope_base,
                ffn_act: "swiglu".to_string(),
                use_genomic_norm: false,
                rope_style,
                anchor_threshold: 0.1,
                ffn_anchor_threshold: 0.1,
                rna_threshold: 0.5,
                unpermute_weights: true, // Re-habilitamos unpermute para Llama
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

    pub fn load_genomic_llm(
        &self,
        config: ModelConfig,
        anchor_threshold: f32,
    ) -> std::io::Result<GenomicLLM> {
        let block_size = 32;

        // Detectar si los pesos de entrada y salida están unidos (Tied Weights)
        let has_output_weight = self.reader.tensors.contains_key("output.weight");

        // Si están unidos, aplicamos el threshold de anclas también a la entrada para mantener simetría
        let embd_threshold = if has_output_weight {
            -1.0
        } else {
            anchor_threshold
        };

        let embd_dna = self.genomize_tensor(
            "token_embd.weight",
            block_size,
            embd_threshold,
            false,
            0,
            0,
            None,
        )?;

        let mut blocks = Vec::new();
        let head_dim = config.n_embd / config.n_head;
        for i in 0..config.n_blocks {
            let p = format!("blk.{}.", i);

            // Carga de Bias (Opcional en GGUF)
            let q_bias = self.load_f32_tensor_optional(&format!("{}attn_q.bias", p));
            let k_bias = self.load_f32_tensor_optional(&format!("{}attn_k.bias", p));
            let v_bias = self.load_f32_tensor_optional(&format!("{}attn_v.bias", p));
            let o_bias = self.load_f32_tensor_optional(&format!("{}attn_output.bias", p));

            let q_gen = self.genomize_tensor(
                &format!("{}attn_q.weight", p),
                block_size,
                anchor_threshold,
                config.config.unpermute_weights,
                config.n_head,
                head_dim,
                q_bias,
            )?;
            let k_gen = self.genomize_tensor(
                &format!("{}attn_k.weight", p),
                block_size,
                anchor_threshold,
                config.config.unpermute_weights,
                config.n_head_kv,
                head_dim,
                k_bias,
            )?;
            let v_gen = self.genomize_tensor(
                &format!("{}attn_v.weight", p),
                block_size,
                anchor_threshold,
                false,
                0,
                0,
                v_bias,
            )?;
            let o_gen = self.genomize_tensor(
                &format!("{}attn_output.weight", p),
                block_size,
                anchor_threshold,
                false,
                0,
                0,
                o_bias,
            )?;

            // FFN Tensors (Normalmente sin bias en Llama/SmolLM, pero Qwen puede tenerlos)
            let gate_bias = self.load_f32_tensor_optional(&format!("{}ffn_gate.bias", p));
            let up_bias = self.load_f32_tensor_optional(&format!("{}ffn_up.bias", p));
            let down_bias = self.load_f32_tensor_optional(&format!("{}ffn_down.bias", p));

            let gate_gen = self.genomize_tensor(
                &format!("{}ffn_gate.weight", p),
                block_size,
                anchor_threshold,
                false,
                0,
                0,
                gate_bias,
            )?;
            let up_gen = self.genomize_tensor(
                &format!("{}ffn_up.weight", p),
                block_size,
                anchor_threshold,
                false,
                0,
                0,
                up_bias,
            )?;
            let down_gen = self.genomize_tensor(
                &format!("{}ffn_down.weight", p),
                block_size,
                anchor_threshold,
                false,
                0,
                0,
                down_bias,
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
                config.config.rna_threshold,
            ));
        }
        let output_norm = self.load_f32_tensor("output_norm.weight")?;

        let lm_head = if has_output_weight {
            let lm_head_bias = self.load_f32_tensor_optional("output.bias");
            self.genomize_tensor(
                "output.weight",
                block_size,
                anchor_threshold,
                false,
                0,
                0,
                lm_head_bias,
            )?
        } else {
            // Tied Weights: La salida es una copia exacta de la entrada
            embd_dna.clone()
        };

        Ok(GenomicLLM {
            embeddings: embd_dna,
            blocks,
            output_norm,
            lm_head,
            eps: config.eps,
            k_wta_ratio: 0.50,
            topology: None,
        })
    }

    fn load_f32_tensor_optional(&self, name: &str) -> Option<Vec<f32>> {
        if !self.reader.tensors.contains_key(name) {
            return None;
        }
        self.load_f32_tensor(name).ok()
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
        unpermute: bool,
        n_head: usize,
        head_dim: usize,
        bias: Option<Vec<f32>>,
    ) -> std::io::Result<GenomicLinear> {
        let data = self.reader.get_tensor_data(name)?;
        let info = self.reader.tensors.get(name).unwrap();
        let out_features = info.shape[info.n_dims as usize - 1] as usize;
        let in_features = info.shape[0] as usize;
        let mut f32_data: Vec<f32> = match info.tensor_type {
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
                crate::compute::math::dequantize_q8_0_core(data, out_features, in_features)
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Unsupported tensor type",
                ))
            }
        };

        if unpermute && n_head > 0 && head_dim > 0 {
            unpermute_f32(&mut f32_data, n_head, head_dim, out_features, in_features);
        }

        // Auto-detección de bit_depth para Mixed-Bit Import
        let bit_depth = if name.contains("attn")
            || name.contains("q_proj")
            || name.contains("k_proj")
            || name.contains("v_proj")
            || name.contains("o_proj")
        {
            4
        } else {
            2
        };

        let (dna, centroids, anchors_u8) = if bit_depth == 4 {
            crate::compute::math::genomize_4bit_core(&f32_data, block_size, anchor_threshold)
        } else {
            crate::compute::math::genomize_f32_core(&f32_data, block_size, anchor_threshold, None)
        };

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
            bias.unwrap_or_default(),
            bit_depth,
        ))
    }
}

fn unpermute_f32(
    data: &mut [f32],
    _n_head: usize,
    head_dim: usize,
    out_features: usize,
    in_features: usize,
) {
    let mut scratch = vec![0.0f32; out_features * in_features];
    for i in 0..out_features {
        let h = i / head_dim;
        let j = i % head_dim;
        let new_j = if j < head_dim / 2 {
            2 * j
        } else {
            2 * (j - head_dim / 2) + 1
        };
        let interleaved_i = h * head_dim + new_j;
        for k in 0..in_features {
            scratch[i * in_features + k] = data[interleaved_i * in_features + k];
        }
    }
    data.copy_from_slice(&scratch);
}