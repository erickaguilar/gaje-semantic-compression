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

    pub fn forward_with_hidden(&mut self, token_id: usize, clear_cache: bool) -> PyResult<(Vec<f32>, Vec<f32>)> {
        if clear_cache {
            self.clear_cache()?;
        }

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

        Ok((logits, h_norm))
    }

    pub fn forward_spiking(&mut self, token_id: usize, steps: usize, threshold: f32, decay: f32) -> PyResult<Vec<f32>> {
        // 1. Process through blocks normally to get the final hidden state
        let (_, h_norm) = self.forward_with_hidden(token_id, false)?;

        // 2. Use neuromorphic spiking for the LM Head
        self.lm_head.spiking_forward(h_norm, steps, threshold, decay)
    }

    pub fn forward_phase_gaje(&mut self, token_id: usize, k_wta: usize) -> PyResult<Vec<f32>> {
        // 1. Inferencia de alta precisión para el cuerpo del Transformer
        let (_, h_norm) = self.forward_with_hidden(token_id, false)?;

        // 2. Proyección LM Head (Excitación base)
        let excitation = self.lm_head.forward(h_norm)?;
        let n_tokens = excitation.len();
        
        // 3. Cálculo de Umbral Dinámico (Homeostasis)
        let max_excitation = excitation.iter().fold(0.0f32, |a, &b| a.max(b));
        let threshold = (max_excitation * 0.7).max(0.1); // Umbral al 70% del pico

        // 4. Simulación de Dinámica Temporal (Phase Coding)
        let mut candidates = Vec::with_capacity(n_tokens);
        for i in 0..n_tokens {
            let energy = excitation[i];
            if energy >= threshold {
                let intensity = 1.0 + (energy - threshold) / threshold;
                let excess_ratio = (energy - threshold) / threshold;
                let phase = if excess_ratio >= 1.0 {
                    0
                } else {
                    15 - (excess_ratio * 15.0) as u8
                };
                candidates.push((i, intensity, phase));
            }
        }

        // 5. Inhibición Lateral (K-WTA)
        candidates.sort_by(|a, b| {
            let res = a.2.cmp(&b.2); // Fase (menor primero)
            if res == std::cmp::Ordering::Equal {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal) // Intensidad (mayor primero)
            } else {
                res
            }
        });

        let mut resonance_logits = vec![-100.0f32; n_tokens];
        let num_winners = candidates.len().min(k_wta);
        
        for i in 0..num_winners {
            let (idx, intensity, phase) = candidates[i];
            let phase_score = (16 - phase) as f32;
            resonance_logits[idx] = intensity * phase_score;
        }

        Ok(resonance_logits)
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
            
            // 1. Forward Pass con captura de estados para Backward
            let pos = if self.blocks.is_empty() { 0 } else { self.blocks[0].attn.k_cache_len() };
            
            let mut hidden_states = Vec::with_capacity(self.blocks.len() + 1);
            let mut current_h = self.embeddings.get_row(token_id)?;
            hidden_states.push(current_h.clone());
            
            for block in &mut self.blocks {
                current_h = block.forward(current_h, pos)?;
                hidden_states.push(current_h.clone());
            }
            
            let h_final = unsafe { rms_norm(&current_h, &self.output_norm, self.eps) };
            let logits = self.lm_head.forward(h_final.clone())?;

            // 2. Calcular Error en la Cabeza (Softmax + CrossEntropy)
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

            // 3. Backward Pass y Refinamiento Multi-capa
            // A. Refinar LM Head
            let d_h_final = self.lm_head.backward(d_logits.clone())?;
            self.lm_head.refine_with_grads(h_final, d_logits, lr)?;

            // B. Refinar Bloques (Propagar d_h hacia atrás)
            let mut d_h = d_h_final;
            
            // Refinamos solo los últimos 3 bloques para estabilidad en móviles
            let start_block = self.blocks.len().saturating_sub(3);
            for b_idx in (start_block..self.blocks.len()).rev() {
                let x_input = hidden_states[b_idx].clone();
                d_h = self.blocks[b_idx].refine_with_grads(x_input, d_h, pos, lr * 0.5)?; // LR menor para bloques
            }
        }

        Ok(total_loss / (tokens.len() - 1) as f32)
    }

    pub fn clear_cache(&mut self) -> PyResult<()> {
        for block in &mut self.blocks {
            block.clear_cache()?;
        }
        Ok(())
    }

    pub fn mutate_all_homeostasis(&mut self, scale: f32) -> PyResult<Vec<f32>> {
        let mut deltas = Vec::with_capacity(self.blocks.len());
        for block in &mut self.blocks {
            deltas.push(block.mutate_homeostasis(scale)?);
        }
        Ok(deltas)
    }

    pub fn undo_homeostasis_mutation(&mut self, deltas: Vec<f32>) -> PyResult<()> {
        if deltas.len() != self.blocks.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Homeostasis delta size mismatch"));
        }
        for (block, d) in self.blocks.iter_mut().zip(deltas) {
            block.h_scale -= d;
        }
        Ok(())
    }
}
