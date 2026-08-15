// =============================================================================
// backward — Gradientes, refine con gradientes y mutaciones de GenomicLinear
// =============================================================================
use crate::nn::linear::database::WeightDatabase;
use crate::nn::linear::GenomicLinear;

impl GenomicLinear {
    pub fn backward_core(&self, _d_output: Vec<f32>) -> Result<Vec<f32>, String> {
        Ok(vec![0.0; self.in_features])
    }
    pub fn refine_with_grads_core(
        &mut self,
        input: Vec<f32>,
        grads: Vec<f32>,
        lr: f32,
    ) -> Result<(), String> {
        if self.centroids.is_empty() {
            return Ok(());
        }

        let n_blocks = (self.in_features + self.block_size - 1) / self.block_size;

        match &self.weight_db {
            WeightDatabase::Genomic2Bit(db) => {
                let mut centroid_grads = vec![0.0f32; self.centroids.len()];
                let mut centroid_counts = vec![0.0f32; self.centroids.len()];

                for i in 0..self.out_features {
                    let g_val = grads.get(i).cloned().unwrap_or(0.0);
                    if g_val == 0.0 {
                        continue;
                    }

                    let row_off = i * n_blocks * self.stride;
                    if row_off + n_blocks * self.stride <= db.len() {
                        for b in 0..n_blocks {
                            let c_off = (i * n_blocks + b) * 4;
                            let has_block_centroids =
                                self.centroids.len() > 4 && c_off + 4 <= self.centroids.len();

                            let block_db =
                                &db[row_off + b * self.stride..row_off + (b + 1) * self.stride];
                            for k in 0..self.block_size {
                                let j = b * self.block_size + k;
                                if j >= self.in_features {
                                    break;
                                }
                                let x_val = input.get(j).cloned().unwrap_or(0.0);

                                let byte_idx = k / 4;
                                let sub_idx = k % 4;
                                if byte_idx < block_db.len() {
                                    let bit_val = ((block_db[byte_idx] >> ((3 - sub_idx) * 2))
                                        & 0b11)
                                        as usize;
                                    let c_idx = if has_block_centroids {
                                        c_off + bit_val
                                    } else {
                                        bit_val % self.centroids.len()
                                    };

                                    centroid_grads[c_idx] += g_val * x_val;
                                    centroid_counts[c_idx] += 1.0;
                                }
                            }
                        }
                    }
                }

                for c_idx in 0..self.centroids.len() {
                    if centroid_counts[c_idx] > 0.0 {
                        self.centroids[c_idx] -=
                            lr * (centroid_grads[c_idx] / centroid_counts[c_idx]);
                    }
                }
            }
            WeightDatabase::Genomic4Bit(db) => {
                let mut centroid_grads = vec![0.0f32; self.centroids.len()];
                let mut centroid_counts = vec![0.0f32; self.centroids.len()];

                for i in 0..self.out_features {
                    let g_val = grads.get(i).cloned().unwrap_or(0.0);
                    if g_val == 0.0 {
                        continue;
                    }
                    let row_off = i * n_blocks * self.stride;
                    if row_off + n_blocks * self.stride <= db.len() {
                        for b in 0..n_blocks {
                            let c_off = (i * n_blocks + b) * 16;
                            let has_block_centroids =
                                self.centroids.len() > 16 && c_off + 16 <= self.centroids.len();

                            for k in 0..self.block_size {
                                let j = b * self.block_size + k;
                                if j >= self.in_features {
                                    break;
                                }
                                let x_val = input.get(j).cloned().unwrap_or(0.0);

                                let byte_idx = k / 2;
                                let sub_idx = k % 2;
                                if byte_idx < self.stride {
                                    let byte = db[row_off + b * self.stride + byte_idx];
                                    let bit_val = if sub_idx == 0 {
                                        (byte >> 4) as usize
                                    } else {
                                        (byte & 0x0F) as usize
                                    };
                                    let c_idx = if has_block_centroids {
                                        c_off + bit_val
                                    } else {
                                        bit_val % self.centroids.len()
                                    };
                                    centroid_grads[c_idx] += g_val * x_val;
                                    centroid_counts[c_idx] += 1.0;
                                }
                            }
                        }
                    }
                }
                for c_idx in 0..self.centroids.len() {
                    if centroid_counts[c_idx] > 0.0 {
                        self.centroids[c_idx] -=
                            lr * (centroid_grads[c_idx] / centroid_counts[c_idx]);
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
    pub fn apply_mutation_core(&mut self, _delta: Vec<f32>, _undo: bool) -> Result<(), String> {
        Ok(())
    }
    pub fn apply_weighted_mutation_core(
        &mut self,
        _delta: Vec<f32>,
        _weight: f32,
    ) -> Result<(), String> {
        Ok(())
    }
    pub fn mutate_random_core(&mut self, _scale: f32) -> Result<Vec<f32>, String> {
        Ok(vec![])
    }
    pub fn recalibrate_centroids_core(&mut self, _shift: f32) -> Result<(), String> {
        Ok(())
    }
    pub fn apply_vector_equilibrium_alignment_core(
        &mut self,
        _strength: f32,
    ) -> Result<(), String> {
        Ok(())
    }
    pub fn anchors_sparse_buffer(&self) -> Vec<u8> {
        vec![]
    }
}
