use crate::compute::kernels::rms_norm;
use crate::nn::block::RustGenomicBlock;
use crate::nn::linear::GenomicLinear;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct RustGenomicLLM {
    #[pyo3(get)]
    pub embeddings: GenomicLinear,
    #[pyo3(get)]
    pub blocks: Vec<RustGenomicBlock>,
    #[pyo3(get)]
    pub output_norm: Vec<f32>,
    #[pyo3(get)]
    pub lm_head: GenomicLinear,
    #[pyo3(get)]
    pub eps: f32,
}

#[pymethods]
impl RustGenomicLLM {
    #[new]
    pub fn new(
        embeddings: GenomicLinear,
        blocks: Vec<RustGenomicBlock>,
        output_norm: Vec<f32>,
        lm_head: GenomicLinear,
        eps: f32,
    ) -> Self {
        RustGenomicLLM {
            embeddings,
            blocks,
            output_norm,
            lm_head,
            eps,
        }
    }

    pub fn forward(&mut self, token_id: usize, clear_cache: bool) -> PyResult<Vec<f32>> {
        if clear_cache {
            self.clear_cache()?;
        }

        // The position is exactly the number of tokens already in the cache
        let pos = if self.blocks.is_empty() {
            0
        } else {
            self.blocks[0].attn.k_cache_len()
        };

        if token_id >= self.embeddings.out_features {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Token id {} out of bounds",
                token_id
            )));
        }

        // 1. Fetch embedding
        let mut h = self.embeddings.get_row(token_id)?;

        // 2. Pass through all blocks (position is updated per token)
        for block in &mut self.blocks {
            h = block.forward(h, pos)?;
        }

        // 3. Final RMSNorm
        let h_norm = unsafe { rms_norm(&h, &self.output_norm, self.eps) };

        // 4. LM Head Projection
        let logits = self.lm_head.forward(h_norm)?;

        Ok(logits)
    }

    pub fn train_step(&mut self, token_id: usize, target_token: usize, lr: f32) -> PyResult<f32> {
        let pos = if self.blocks.is_empty() {
            0
        } else {
            self.blocks[0].attn.k_cache_len()
        };

        if token_id >= self.embeddings.out_features {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Token id {} out of bounds", token_id)));
        }

        let mut h = self.embeddings.get_row(token_id)?;
        for block in &mut self.blocks {
            h = block.forward(h, pos)?;
        }

        let h_norm = unsafe { rms_norm(&h, &self.output_norm, self.eps) };
        let logits = self.lm_head.forward(h_norm.clone())?;

        let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut exps = vec![0.0f32; logits.len()];
        let mut sum_exp = 0.0f32;
        for i in 0..logits.len() {
            let e = (logits[i] - max_l).exp();
            exps[i] = e;
            sum_exp += e;
        }

        let mut probs = vec![0.0f32; logits.len()];
        for i in 0..logits.len() {
            probs[i] = exps[i] / sum_exp;
        }

        let loss = -(probs[target_token] + 1e-12).ln();

        let mut d_logits = probs;
        d_logits[target_token] -= 1.0;

        self.lm_head.refine_with_grads(h_norm, d_logits, lr)?;

        Ok(loss)
    }

    pub fn embeddings_forward(&self, token_id: usize) -> PyResult<Vec<f32>> {
        if token_id >= self.embeddings.out_features {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Token id {} out of bounds",
                token_id
            )));
        }
        self.embeddings.get_row(token_id)
    }

    pub fn apply_mutation(
        &mut self,
        layer_name: &str,
        delta_centroids: Vec<f32>,
        undo: bool,
    ) -> PyResult<()> {
        if layer_name == "token_embd" {
            return self.embeddings.apply_mutation(delta_centroids, undo);
        }
        if layer_name == "lm_head" {
            return self.lm_head.apply_mutation(delta_centroids, undo);
        }
        if layer_name.starts_with("blk.") {
            let parts: Vec<&str> = layer_name.split('.').collect();
            if parts.len() >= 3 {
                let block_idx: usize = parts[1].parse().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Invalid block index: {}", e))
                })?;
                let field = parts[2];
                if block_idx < self.blocks.len() {
                    let block = &mut self.blocks[block_idx];
                    match field {
                        "attn_q" => return block.q_gen.apply_mutation(delta_centroids, undo),
                        "attn_k" => return block.k_gen.apply_mutation(delta_centroids, undo),
                        "attn_v" => return block.v_gen.apply_mutation(delta_centroids, undo),
                        "attn_output" => return block.w_o.apply_mutation(delta_centroids, undo),
                        "ffn_gate" => return block.gate_gen.apply_mutation(delta_centroids, undo),
                        "ffn_up" => return block.up_gen.apply_mutation(delta_centroids, undo),
                        "ffn_down" => return block.w_down.apply_mutation(delta_centroids, undo),
                        _ => {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "Unknown field: {}",
                                field
                            )))
                        }
                    }
                }
            }
        }
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Layer not found: {}",
            layer_name
        )))
    }

    pub fn refine_lm_head(
        &mut self,
        input: Vec<f32>,
        grads: Vec<f32>,
        lr: f32,
    ) -> PyResult<()> {
        self.lm_head.refine_with_grads(input, grads, lr)
    }

    pub fn refine_embeddings(
        &mut self,
        input: Vec<f32>,
        grads: Vec<f32>,
        lr: f32,
    ) -> PyResult<()> {
        self.embeddings.refine_with_grads(input, grads, lr)
    }

    pub fn mutate_layer(&mut self, layer_name: &str, scale: f32) -> PyResult<Vec<f32>> {
        if layer_name == "token_embd" {
            return self.embeddings.mutate_random(scale);
        }
        if layer_name == "lm_head" {
            return self.lm_head.mutate_random(scale);
        }
        if layer_name.starts_with("blk.") {
            let parts: Vec<&str> = layer_name.split('.').collect();
            if parts.len() >= 3 {
                let block_idx: usize = parts[1].parse().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Invalid block index: {}", e))
                })?;
                let field = parts[2];
                if block_idx < self.blocks.len() {
                    let block = &mut self.blocks[block_idx];
                    match field {
                        "attn_q" => return block.q_gen.mutate_random(scale),
                        "attn_k" => return block.k_gen.mutate_random(scale),
                        "attn_v" => return block.v_gen.mutate_random(scale),
                        "attn_output" => return block.w_o.mutate_random(scale),
                        "ffn_gate" => return block.gate_gen.mutate_random(scale),
                        "ffn_up" => return block.up_gen.mutate_random(scale),
                        "ffn_down" => return block.w_down.mutate_random(scale),
                        _ => {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "Unknown field: {}",
                                field
                            )))
                        }
                    }
                }
            }
        }
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Layer not found: {}",
            layer_name
        )))
    }

    pub fn undo_layer_mutation(&mut self, layer_name: &str, delta: Vec<f32>) -> PyResult<()> {
        if layer_name == "token_embd" {
            return self.embeddings.undo_delta(delta);
        }
        if layer_name == "lm_head" {
            return self.lm_head.undo_delta(delta);
        }
        if layer_name.starts_with("blk.") {
            let parts: Vec<&str> = layer_name.split('.').collect();
            if parts.len() >= 3 {
                let block_idx: usize = parts[1].parse().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Invalid block index: {}", e))
                })?;
                let field = parts[2];
                if block_idx < self.blocks.len() {
                    let block = &mut self.blocks[block_idx];
                    match field {
                        "attn_q" => return block.q_gen.undo_delta(delta),
                        "attn_k" => return block.k_gen.undo_delta(delta),
                        "attn_v" => return block.v_gen.undo_delta(delta),
                        "attn_output" => return block.w_o.undo_delta(delta),
                        "ffn_gate" => return block.gate_gen.undo_delta(delta),
                        "ffn_up" => return block.up_gen.undo_delta(delta),
                        "ffn_down" => return block.w_down.undo_delta(delta),
                        _ => {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "Unknown field: {}",
                                field
                            )))
                        }
                    }
                }
            }
        }
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Layer not found: {}",
            layer_name
        )))
    }

    pub fn apply_weighted_layer_mutation(&mut self, layer_name: &str, delta: Vec<f32>, weight: f32) -> PyResult<()> {
        if layer_name == "token_embd" {
            return self.embeddings.apply_weighted_mutation(delta, weight);
        }
        if layer_name == "lm_head" {
            return self.lm_head.apply_weighted_mutation(delta, weight);
        }
        if layer_name.starts_with("blk.") {
            let parts: Vec<&str> = layer_name.split('.').collect();
            if parts.len() >= 3 {
                let block_idx: usize = parts[1].parse().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Invalid block index: {}", e))
                })?;
                let field = parts[2];
                if block_idx < self.blocks.len() {
                    let block = &mut self.blocks[block_idx];
                    match field {
                        "attn_q" => return block.q_gen.apply_weighted_mutation(delta, weight),
                        "attn_k" => return block.k_gen.apply_weighted_mutation(delta, weight),
                        "attn_v" => return block.v_gen.apply_weighted_mutation(delta, weight),
                        "attn_output" => return block.w_o.apply_weighted_mutation(delta, weight),
                        "ffn_gate" => return block.gate_gen.apply_weighted_mutation(delta, weight),
                        "ffn_up" => return block.up_gen.apply_weighted_mutation(delta, weight),
                        "ffn_down" => return block.w_down.apply_weighted_mutation(delta, weight),
                        _ => {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "Unknown field: {}",
                                field
                            )))
                        }
                    }
                }
            }
        }
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Layer not found: {}",
            layer_name
        )))
    }

    pub fn train_on_sequence(&mut self, tokens: Vec<usize>, lr: f32) -> PyResult<f32> {
        if tokens.len() < 2 { return Ok(0.0); }
        let mut total_loss = 0.0;
        self.clear_cache()?;

        for i in 0..tokens.len() - 1 {
            let token_id = tokens[i];
            let target_token = tokens[i+1];
            
            // Reutilizamos la lógica de train_step pero sin limpiar caché para mantener contexto
            let pos = if self.blocks.is_empty() { 0 } else { self.blocks[0].attn.k_cache_len() };
            let mut h = self.embeddings.get_row(token_id)?;
            for block in &mut self.blocks {
                h = block.forward(h, pos)?;
            }
            let h_norm = unsafe { rms_norm(&h, &self.output_norm, self.eps) };
            let logits = self.lm_head.forward(h_norm.clone())?;

            let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0f32;
            let mut exps = vec![0.0f32; logits.len()];
            for j in 0..logits.len() {
                let e = (logits[j] - max_l).exp();
                exps[j] = e;
                sum_exp += e;
            }

            let prob_target = (exps[target_token] / sum_exp).max(1e-12);
            total_loss -= prob_target.ln();

            let mut d_logits = vec![0.0f32; logits.len()];
            for j in 0..logits.len() {
                d_logits[j] = exps[j] / sum_exp;
            }
            d_logits[target_token] -= 1.0;

            self.lm_head.refine_with_grads(h_norm, d_logits, lr)?;
        }

        Ok(total_loss / (tokens.len() - 1) as f32)
    }

    pub fn clear_cache(&mut self) -> PyResult<()> {
        for block in &mut self.blocks {
            block.clear_cache()?;
        }
        Ok(())
    }
}
