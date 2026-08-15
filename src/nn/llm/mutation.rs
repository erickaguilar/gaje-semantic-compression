// =============================================================================
// mutation — Mutaciones por capa y homeostasis de GenomicLLM
// =============================================================================
use crate::nn::llm::GenomicLLM;

impl GenomicLLM {
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

    pub fn mutate_all_homeostasis_core(&mut self, scale: f32) -> Result<Vec<f32>, String> {
        let mut deltas = Vec::with_capacity(self.blocks.len());
        for block in &mut self.blocks {
            let delta = block.mutate_homeostasis_core(scale)?;
            deltas.push(delta);
        }
        Ok(deltas)
    }

    pub fn undo_homeostasis_mutation_core(&mut self, deltas: Vec<f32>) -> Result<(), String> {
        if deltas.len() != self.blocks.len() {
            return Err(format!(
                "Deltas length {} does not match blocks length {}",
                deltas.len(),
                self.blocks.len()
            ));
        }
        for (block, delta) in self.blocks.iter_mut().zip(deltas) {
            block.h_scale -= delta;
            block.h_scale = block.h_scale.clamp(0.01, 10.0);
        }
        Ok(())
    }
}
