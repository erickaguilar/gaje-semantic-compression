// =============================================================================
// load — load_genomic_llm: construcción del modelo genómico completo
// =============================================================================
use crate::io::config::ModelConfig;
use super::GGUFLoader;
use crate::nn::{GenomicAttention, GenomicLLM, RustGenomicBlock};

impl GGUFLoader {
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
            quantum_embeddings: None,
            gpu_layers: 0,
            use_gpu: false,
        })
    }
}
