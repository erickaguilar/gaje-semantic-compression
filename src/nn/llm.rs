use crate::compute::kernels::rms_norm;
use crate::core::topology::CentroidGraph;
use crate::nn::block::RustGenomicBlock;
use crate::nn::linear::GenomicLinear;
use std::sync::Arc;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Núcleo del Modelo de Lenguaje Genómico (Pure Rust)
#[cfg_attr(feature = "python", pyclass(name = "RustGenomicLLM"))]
#[derive(Clone)]
pub struct GenomicLLM {
    pub embeddings: GenomicLinear,
    pub blocks: Vec<RustGenomicBlock>,
    pub output_norm: Vec<f32>,
    pub lm_head: GenomicLinear,
    pub eps: f32,
    pub topology: Option<Arc<CentroidGraph>>,
}

impl GenomicLLM {
    pub fn forward_core(&mut self, token_id: usize, clear_cache: bool) -> Result<Vec<f32>, String> {
        if clear_cache {
            self.clear_cache_core();
        }
        let pos = if self.blocks.is_empty() {
            0
        } else {
            self.blocks[0].attn.k_cache_len()
        };
        let token_id = if self.embeddings.out_features > 0 {
            token_id % self.embeddings.out_features
        } else {
            return Err("Embeddings layer has 0 out_features".to_string());
        };

        // Modulación granular para la capa de salida (usamos la última capa como referencia)
        let modulation = self
            .topology
            .as_ref()
            .map(|t| t.get_modulation_factors(self.blocks.len(), 2, 0.5));

        // Embeddings: Usamos activación completa para máxima estabilidad inicial
        let mut h = self.embeddings.get_row_core(token_id)?;
        for block in &mut self.blocks {
            h = block.forward_core(h, pos)?;
        }
        let h_norm = unsafe { rms_norm(&h, &self.output_norm, self.eps) };

        // LM Head: Activación dinámica basada en la entropía del estado final
        let entropy = crate::compute::math::calculate_activation_entropy(&h_norm);
        let rna_threshold = if self.blocks.is_empty() {
            0.5
        } else {
            self.blocks[0].rna_threshold
        };
        let activate_rna = crate::compute::math::should_activate_rna(entropy, rna_threshold);

        if self.lm_head.out_features == 0 {
            return Err("LM Head out_features is 0!".to_string());
        }

        self.lm_head.forward_core(h_norm, modulation, activate_rna)
    }

    pub fn load_topology_core(&mut self, path: &str) -> Result<(), String> {
        let topo = crate::io::loader::load_topology(path).map_err(|e| e.to_string())?;
        let shared_topo = Arc::new(topo);
        self.topology = Some(shared_topo.clone());
        for block in &mut self.blocks {
            block.topology = Some(shared_topo.clone());
        }
        Ok(())
    }

    pub fn forward_with_hidden_core(
        &mut self,
        token_id: usize,
        clear_cache: bool,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        if clear_cache {
            self.clear_cache_core();
        }
        let pos = if self.blocks.is_empty() {
            0
        } else {
            self.blocks[0].attn.k_cache_len()
        };
        let token_id = if self.embeddings.out_features > 0 {
            token_id % self.embeddings.out_features
        } else {
            return Err("Embeddings layer has 0 out_features".to_string());
        };

        let modulation = self
            .topology
            .as_ref()
            .map(|t| t.get_modulation_factors(self.blocks.len(), 2, 0.5));

        let mut h = self.embeddings.get_row_core(token_id)?;
        for block in &mut self.blocks {
            h = block.forward_core(h, pos)?;
        }
        let h_norm = unsafe { rms_norm(&h, &self.output_norm, self.eps) };

        let entropy = crate::compute::math::calculate_activation_entropy(&h_norm);
        let rna_threshold = if self.blocks.is_empty() {
            0.5
        } else {
            self.blocks[0].rna_threshold
        };
        let activate_rna = crate::compute::math::should_activate_rna(entropy, rna_threshold);

        let logits = self
            .lm_head
            .forward_core(h_norm.clone(), modulation, activate_rna)?;
        Ok((logits, h_norm))
    }

    pub fn forward_phase_gaje_core(
        &mut self,
        token_id: usize,
        k_wta: usize,
    ) -> Result<Vec<f32>, String> {
        let (_, h_norm) = self.forward_with_hidden_core(token_id, false)?;

        let modulation = self
            .topology
            .as_ref()
            .map(|t| t.get_modulation_factors(self.blocks.len(), 2, 0.5));

        let entropy = crate::compute::math::calculate_activation_entropy(&h_norm);
        let rna_threshold = if self.blocks.is_empty() {
            0.5
        } else {
            self.blocks[0].rna_threshold
        };
        let activate_rna = crate::compute::math::should_activate_rna(entropy, rna_threshold);

        let excitation = self
            .lm_head
            .forward_core(h_norm, modulation, activate_rna)?;

        let n_tokens = excitation.len();
        let max_excitation = excitation.iter().fold(0.0f32, |a, &b| a.max(b));
        let threshold = (max_excitation * 0.7).max(0.1);
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
        candidates.sort_by(|a, b| {
            let res = a.2.cmp(&b.2);
            if res == std::cmp::Ordering::Equal {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
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

    pub fn clear_cache_core(&mut self) {
        for block in &mut self.blocks {
            block.clear_cache_core();
        }
    }

    pub fn train_on_sequence_core(&mut self, tokens: Vec<usize>, lr: f32) -> Result<f32, String> {
        if tokens.len() < 2 {
            return Ok(0.0);
        }
        let mut total_loss = 0.0;
        self.clear_cache_core();

        let _modulation = self
            .topology
            .as_ref()
            .map(|t| t.get_modulation_factors(self.blocks.len(), 2, 0.5));

        for i in 0..tokens.len() - 1 {
            let (logits, h_final) = self.forward_with_hidden_core(tokens[i], false)?;
            let target = tokens[i + 1];
            let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_e = 0.0f32;
            let mut exps = vec![0.0f32; logits.len()];
            for j in 0..logits.len() {
                let e = (logits[j] - max_l).exp();
                exps[j] = e;
                sum_e += e;
            }
            let prob = (exps[target] / sum_e).max(1e-12);
            total_loss -= prob.ln();
            let mut d_logits = vec![0.0f32; logits.len()];
            for j in 0..logits.len() {
                d_logits[j] = exps[j] / sum_e;
            }
            d_logits[target] -= 1.0;
            self.lm_head.refine_with_grads_core(h_final, d_logits, lr)?;
        }
        Ok(total_loss / (tokens.len() - 1) as f32)
    }

    pub fn apply_mutation_core(
        &mut self,
        layer_name: &str,
        delta: Vec<f32>,
        undo: bool,
    ) -> Result<(), String> {
        if layer_name == "token_embd" {
            return self.embeddings.apply_mutation_core(delta, undo);
        }
        if layer_name == "lm_head" {
            return self.lm_head.apply_mutation_core(delta, undo);
        }
        if layer_name.starts_with("blk.") {
            let parts: Vec<&str> = layer_name.split('.').collect();
            if parts.len() >= 3 {
                let b_idx: usize = parts[1]
                    .parse()
                    .map_err(|e| format!("Invalid block: {}", e))?;
                if b_idx < self.blocks.len() {
                    let b = &mut self.blocks[b_idx];
                    match parts[2] {
                        "attn_q" => return b.q_gen.apply_mutation_core(delta, undo),
                        "attn_k" => return b.k_gen.apply_mutation_core(delta, undo),
                        "attn_v" => return b.v_gen.apply_mutation_core(delta, undo),
                        "attn_output" => return b.w_o.apply_mutation_core(delta, undo),
                        "ffn_gate" => return b.gate_gen.apply_mutation_core(delta, undo),
                        "ffn_up" => return b.up_gen.apply_mutation_core(delta, undo),
                        "ffn_down" => return b.w_down.apply_mutation_core(delta, undo),
                        _ => return Err(format!("Field not found: {}", parts[2])),
                    }
                }
            }
        }
        Err(format!("Layer not found: {}", layer_name))
    }

    pub fn apply_weighted_layer_mutation_core(
        &mut self,
        layer_name: &str,
        delta: Vec<f32>,
        weight: f32,
    ) -> Result<(), String> {
        if layer_name == "token_embd" {
            return self.embeddings.apply_weighted_mutation_core(delta, weight);
        }
        if layer_name == "lm_head" {
            return self.lm_head.apply_weighted_mutation_core(delta, weight);
        }
        if layer_name.starts_with("blk.") {
            let parts: Vec<&str> = layer_name.split('.').collect();
            if parts.len() >= 3 {
                let b_idx: usize = parts[1]
                    .parse()
                    .map_err(|e| format!("Invalid block index: {}", e))?;
                if b_idx < self.blocks.len() {
                    let b = &mut self.blocks[b_idx];
                    match parts[2] {
                        "attn_q" => return b.q_gen.apply_weighted_mutation_core(delta, weight),
                        "attn_k" => return b.k_gen.apply_weighted_mutation_core(delta, weight),
                        "attn_v" => return b.v_gen.apply_weighted_mutation_core(delta, weight),
                        "attn_output" => return b.w_o.apply_weighted_mutation_core(delta, weight),
                        "ffn_gate" => {
                            return b.gate_gen.apply_weighted_mutation_core(delta, weight)
                        }
                        "ffn_up" => return b.up_gen.apply_weighted_mutation_core(delta, weight),
                        "ffn_down" => return b.w_down.apply_weighted_mutation_core(delta, weight),
                        _ => return Err(format!("Unknown field: {}", parts[2])),
                    }
                }
            }
        }
        Err(format!("Layer not found: {}", layer_name))
    }

    pub fn mutate_layer_core(&mut self, layer_name: &str, scale: f32) -> Result<Vec<f32>, String> {
        if layer_name == "token_embd" {
            return self.embeddings.mutate_random_core(scale);
        }
        if layer_name == "lm_head" {
            return self.lm_head.mutate_random_core(scale);
        }
        if layer_name.starts_with("blk.") {
            let parts: Vec<&str> = layer_name.split('.').collect();
            if parts.len() >= 3 {
                let b_idx: usize = parts[1]
                    .parse()
                    .map_err(|e| format!("Invalid block index: {}", e))?;
                if b_idx < self.blocks.len() {
                    let b = &mut self.blocks[b_idx];
                    match parts[2] {
                        "attn_q" => return b.q_gen.mutate_random_core(scale),
                        "attn_k" => return b.k_gen.mutate_random_core(scale),
                        "attn_v" => return b.v_gen.mutate_random_core(scale),
                        "attn_output" => return b.w_o.mutate_random_core(scale),
                        "ffn_gate" => return b.gate_gen.mutate_random_core(scale),
                        "ffn_up" => return b.up_gen.mutate_random_core(scale),
                        "ffn_down" => return b.w_down.mutate_random_core(scale),
                        _ => return Err(format!("Unknown field: {}", parts[2])),
                    }
                }
            }
        }
        Err(format!("Layer not found: {}", layer_name))
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl GenomicLLM {
    #[new]
    pub fn py_new(
        embeddings: GenomicLinear,
        blocks: Vec<RustGenomicBlock>,
        output_norm: Vec<f32>,
        lm_head: GenomicLinear,
        eps: f32,
    ) -> Self {
        GenomicLLM {
            embeddings,
            blocks,
            output_norm,
            lm_head,
            eps,
            topology: None,
        }
    }

    #[getter]
    pub fn vocab_size(&self) -> usize {
        self.embeddings.out_features
    }

    pub fn load_topology(&mut self, json_path: &str) -> PyResult<()> {
        self.load_topology_core(json_path)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        println!("[*] Topología Relacional inyectada desde: {}", json_path);
        Ok(())
    }

    pub fn forward(&mut self, token_id: usize, clear_cache: bool) -> PyResult<Vec<f32>> {
        self.forward_core(token_id, clear_cache)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    pub fn forward_with_hidden(
        &mut self,
        token_id: usize,
        clear_cache: bool,
    ) -> PyResult<(Vec<f32>, Vec<f32>)> {
        self.forward_with_hidden_core(token_id, clear_cache)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    pub fn refine_lm_head(
        &mut self,
        hidden_state: Vec<f32>,
        grad_logits: Vec<f32>,
        lr: f32,
    ) -> PyResult<()> {
        self.lm_head
            .refine_with_grads_core(hidden_state, grad_logits, lr)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    pub fn train_on_sequence(&mut self, tokens: Vec<usize>, lr: f32) -> PyResult<f32> {
        self.train_on_sequence_core(tokens, lr)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    pub fn clear_cache_py(&mut self) -> PyResult<()> {
        self.clear_cache_core();
        Ok(())
    }

    pub fn recalibrate_all_centroids(&mut self, _shift: f32) -> PyResult<()> {
        // ...
        Ok(())
    }

    pub fn apply_vector_equilibrium_alignment_all(&mut self, strength: f32) -> PyResult<()> {
        self.embeddings
            .apply_vector_equilibrium_alignment_core(strength)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        for block in &mut self.blocks {
            block
                .q_gen
                .apply_vector_equilibrium_alignment_core(strength)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            block
                .k_gen
                .apply_vector_equilibrium_alignment_core(strength)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            block
                .v_gen
                .apply_vector_equilibrium_alignment_core(strength)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            block
                .w_o
                .apply_vector_equilibrium_alignment_core(strength)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            block
                .gate_gen
                .apply_vector_equilibrium_alignment_core(strength)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            block
                .up_gen
                .apply_vector_equilibrium_alignment_core(strength)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            block
                .w_down
                .apply_vector_equilibrium_alignment_core(strength)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
        }
        self.lm_head
            .apply_vector_equilibrium_alignment_core(strength)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(())
    }
}
