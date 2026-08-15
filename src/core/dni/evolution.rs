// =============================================================================
// evolution — Mutación dirigida, evaluación y perfilado de DNIEngine
// =============================================================================
use rand::Rng;

use crate::nn::linear::GenomicOperable;
use crate::nn::llm::GenomicLLM;

use crate::core::dni::DNIEngine;

impl DNIEngine {
    pub fn initialize_original_hash(&mut self) {
        self.original_dna_hash = Self::calculate_dna_hash(&self.model);
    }

    fn calculate_dna_hash(model: &GenomicLLM) -> Vec<u64> {
        let mut hashes = Vec::new();
        for block in &model.blocks {
            hashes.push(
                block
                    .gate_gen
                    .database_ref()
                    .iter()
                    .map(|&b| b as u64)
                    .sum(),
            );
            hashes.push(block.up_gen.database_ref().iter().map(|&b| b as u64).sum());
            hashes.push(block.w_down.database_ref().iter().map(|&b| b as u64).sum());
        }
        hashes
    }

    pub(crate) fn profile_activations(&mut self, tokens: &[u32]) -> Vec<Vec<f32>> {
        let n_blocks = self.model.blocks.len();
        let mut activation_stats = vec![Vec::new(); n_blocks];
        self.model.clear_cache_core();
        for &token in tokens {
            if let Ok((_, h_final)) = self.model.forward_with_hidden_core(token as usize, false) {
                for stats in activation_stats.iter_mut() {
                    if stats.is_empty() {
                        *stats = vec![0.0f32; h_final.len()];
                    }
                    for (j, &val) in h_final.iter().enumerate() {
                        stats[j] += val.abs();
                    }
                }
            }
        }
        activation_stats
    }

    pub(crate) fn calculate_fuzzy_membership(
        idx: usize,
        sorted_anchors: &[usize],
        sigma: f32,
    ) -> f32 {
        if sorted_anchors.is_empty() || sigma <= 1e-6 {
            return 0.0;
        }

        // Búsqueda binaria para encontrar el ancla más cercana en O(log N)
        let pos = match sorted_anchors.binary_search(&idx) {
            Ok(_) => return 1.0, // Coincidencia exacta (ancla protegida)
            Err(p) => p,
        };

        let mut min_dist_sq = f32::MAX;

        // Comprobar vecinos inmediatos (izquierdo y derecho)
        if pos < sorted_anchors.len() {
            let dist = (sorted_anchors[pos] as f32 - idx as f32).abs();
            if dist < 10.0 {
                min_dist_sq = min_dist_sq.min(dist * dist);
            }
        }
        if pos > 0 {
            let dist = (idx as f32 - sorted_anchors[pos - 1] as f32).abs();
            if dist < 10.0 {
                min_dist_sq = min_dist_sq.min(dist * dist);
            }
        }

        if min_dist_sq == f32::MAX {
            0.0
        } else {
            (-min_dist_sq / (2.0 * sigma * sigma)).exp()
        }
    }

    pub fn apply_targeted_mutation_v2(
        &self,
        mutant: &mut GenomicLLM,
        rate: f32,
        activations: &[Vec<f32>],
        sigma: f32,
    ) {
        let mut rng = rand::thread_rng();
        let n_blocks = mutant.blocks.len();
        for i in 0..n_blocks {
            let block = &mut mutant.blocks[i];
            let layer_stats = activations.get(i);
            let layers = [&mut block.gate_gen, &mut block.up_gen, &mut block.w_down];
            for layer in layers {
                // Cálculo de entropía local para ajustar sigma dinámicamente
                let h = crate::compute::math::calculate_genomic_entropy_core(layer.database_ref());
                let local_sigma = sigma * (1.0 + h);

                // Optimizacion 1: Usar Vec ordenado en lugar de HashSet para búsqueda binaria
                let mut sorted_anchors: Vec<usize> = layer
                    .anchor_indices
                    .iter()
                    .map(|&idx| idx as usize)
                    .collect();
                sorted_anchors.sort_unstable();

                let n_neurons = layer.out_features;
                let row_len_bytes = layer.weight_db.len_bytes() / n_neurons;
                let bit_depth = layer.weight_db.bit_depth();
                let params_per_byte = 8 / bit_depth;

                for row in 0..n_neurons {
                    let mut row_rate = rate;
                    if let Some(stats) = layer_stats {
                        if let Some(&act) = stats.get(row) {
                            if act < 0.1 {
                                row_rate *= 5.0;
                            } else if act > 10.0 {
                                row_rate *= 0.1;
                            }
                        }
                    }

                    // Optimizacion 2: Si row_rate es extremadamente bajo, podemos saltar la fila
                    if row_rate < 1e-8 {
                        continue;
                    }

                    let row_start = row * row_len_bytes;
                    for byte_idx in 0..row_len_bytes {
                        let global_byte_idx = row_start + byte_idx;

                        for s in 0..params_per_byte as usize {
                            if rng.gen::<f32>() < row_rate {
                                let input_idx = byte_idx * params_per_byte as usize + s;
                                let global_weight_idx = row * layer.in_features + input_idx;

                                let membership = Self::calculate_fuzzy_membership(
                                    global_weight_idx,
                                    &sorted_anchors,
                                    local_sigma,
                                );

                                // Aplicamos la penalización de membership
                                if rng.gen::<f32>() < (1.0 - membership) {
                                    let current_bits = layer.weight_db.read(global_byte_idx, s);
                                    let max_val = (1 << bit_depth) - 1;
                                    let mutation = rng.gen::<u8>() % (max_val + 1);

                                    if mutation != current_bits {
                                        layer.weight_db.mutate(global_byte_idx, s, mutation);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn evaluate_mutant(&self, mutant: &mut GenomicLLM, tokens: &[u32]) -> f32 {
        mutant.clear_cache_core();
        let mut total_prob = 0.0;
        let mut count = 0;
        for i in 0..tokens.len() - 1 {
            if let Ok(logits) = mutant.forward_core(tokens[i] as usize, false) {
                let target = if logits.is_empty() {
                    0
                } else {
                    (tokens[i + 1] as usize) % logits.len()
                };
                let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum_exp = 0.0f32;
                for &l in &logits {
                    sum_exp += (l - max_l).exp();
                }
                total_prob += (logits[target] - max_l).exp() / (sum_exp + 1e-12);
                count += 1;
            }
        }
        if count > 0 {
            total_prob / count as f32
        } else {
            0.0
        }
    }
}
