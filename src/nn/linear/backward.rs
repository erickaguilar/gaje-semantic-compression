// =============================================================================
// backward — Gradientes, refine con gradientes y mutaciones de GenomicLinear
// =============================================================================
use rayon::prelude::*;
use std::sync::Arc;

use crate::nn::linear::database::WeightDatabase;
use crate::nn::linear::GenomicLinear;

impl GenomicLinear {
    /// Multiplicación transpuesta: d_input[j] = Σ_i W[i,j] * d_output[i].
    /// Devuelve el gradiente respecto a la entrada de la capa (backprop).
    ///
    /// Paralelizado por filas (i), pues el lm_head puede tener decenas de miles de
    /// filas (vocab) y su backward serial era el cuello de botella del entrenamiento.
    pub fn backward_core(&self, d_output: Vec<f32>) -> Result<Vec<f32>, String> {
        if self.out_features == 0 || self.in_features == 0 {
            return Ok(vec![0.0f32; self.in_features]);
        }
        let n_blocks = self.in_features / self.block_size;
        let in_f = self.in_features;

        let partial = |range: std::ops::Range<usize>| -> Vec<f32> {
            let mut acc = vec![0.0f32; in_f];
            for i in range {
                let g = d_output.get(i).copied().unwrap_or(0.0);
                if g == 0.0 {
                    continue;
                }
                match &self.weight_db {
                    WeightDatabase::GenomicF32(db) => {
                        let row = i * in_f;
                        if row + in_f > db.len() {
                            continue;
                        }
                        for j in 0..in_f {
                            acc[j] += g * db[row + j];
                        }
                    }
                    WeightDatabase::Genomic4Bit(db) => {
                        // Genomic4Bit: W[i,j] = centroids[(i*n_blocks + b)*16 + nibble],
                        // con b = j/block_size. Layout: row_off = i*n_blocks*stride,
                        // stride=block_size/2.
                        let stride = self.stride;
                        if stride == 0 {
                            continue;
                        }
                        let row_off = i * n_blocks * stride;
                        if row_off + n_blocks * stride > db.len() {
                            continue;
                        }
                        for b in 0..n_blocks {
                            let c_off = (i * n_blocks + b) * 16;
                            let c_start = if self.centroids.len() >= c_off + 16 {
                                c_off
                            } else if self.centroids.len() >= 16 {
                                0
                            } else {
                                continue;
                            };
                            for k in 0..self.block_size {
                                let j = b * self.block_size + k;
                                if j >= in_f {
                                    break;
                                }
                                let byte_idx = k / 2;
                                let sub = k % 2;
                                let byte = db[row_off + b * stride + byte_idx];
                                // Elemento par (k%2==0) = nibble alto, impar = nibble bajo
                                // (alineado con `GenomicOperable::read` y el forward).
                                let nibble = if sub == 0 {
                                    (byte >> 4) as usize
                                } else {
                                    (byte & 0x0F) as usize
                                };
                                acc[j] += g * self.centroids[c_start + nibble];
                            }
                        }
                    }
                    WeightDatabase::GenomicQ4_0(db) => {
                        let row_off = i * n_blocks;
                        if row_off + n_blocks > db.len() {
                            continue;
                        }
                        for b in 0..n_blocks {
                            let block = db[row_off + b];
                            let scale = block.scale.to_f32();
                            let min = block.min.to_f32();
                            for k in 0..self.block_size {
                                let j = b * self.block_size + k;
                                if j >= in_f {
                                    break;
                                }
                                let q = block.q_value(k) as f32;
                                acc[j] += g * (q * scale + min);
                            }
                        }
                    }
                    WeightDatabase::GenomicQ8_0(db) => {
                        let row_off = i * n_blocks;
                        if row_off + n_blocks > db.len() {
                            continue;
                        }
                        for b in 0..n_blocks {
                            let block = db[row_off + b];
                            let scale = block.scale.to_f32();
                            for k in 0..self.block_size {
                                let j = b * self.block_size + k;
                                if j >= in_f {
                                    break;
                                }
                                acc[j] += g * (block.qs[k] as f32) * scale;
                            }
                        }
                    }
                    _ => {}
                }
            }
            acc
        };

        let out = self.out_features;
        let n_threads = rayon::current_num_threads();
        let parts: Vec<Vec<f32>> = (0..n_threads)
            .into_par_iter()
            .map(|c| {
                let start = c * out / n_threads;
                let end = (((c + 1) * out) / n_threads).min(out);
                partial(start..end)
            })
            .collect();
        let mut d_input = vec![0.0f32; in_f];
        for part in parts {
            for j in 0..in_f {
                d_input[j] += part[j];
            }
        }
        Ok(d_input)
    }
    pub fn refine_with_grads_core(
        &mut self,
        input: Vec<f32>,
        grads: Vec<f32>,
        lr: f32,
    ) -> Result<(), String> {
        let n_blocks = (self.in_features + self.block_size - 1) / self.block_size;

        // Las variantes basadas en centroides (2/4-bit) no pueden refinar sin
        // centroids. GenomicF32 (lm_head/embeddings densos) no usa centroids y
        // debe actualizarse siempre; de lo contrario queda como no-op silencioso.
        let centroid_based = matches!(
            self.weight_db,
            WeightDatabase::Genomic2Bit(_) | WeightDatabase::Genomic4Bit(_)
        );
        if self.centroids.is_empty() && centroid_based {
            return Ok(());
        }

        match &mut self.weight_db {
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
                        // Gradiente verdadero del centroide: la SUMA de
                        // g_val*x_val (los pesos comparten el valor c, así que
                        // dL/dc = Σ contribuciones). NO dividir por el conteo.
                        let delta = (lr * centroid_grads[c_idx]).clamp(-0.05, 0.05);
                        self.centroids[c_idx] = (self.centroids[c_idx] - delta).clamp(-20.0, 20.0);
                    }
                }
            }
            WeightDatabase::GenomicF32(db) => {
                // Pesos densos FP32 (lm_head/embeddings en el formato híbrido).
                // Layout row-major: W[i,j] = db[i*in_features + j], y el gradiente
                // `grads` es ∂L/∂logits (tamaño = out_features). SGD por token:
                //   W[i,j] -= lr * grads[i] * input[j]
                let db_mut = Arc::make_mut(db);
                let in_f = self.in_features;
                if in_f == 0 {
                    return Ok(());
                }
                for i in 0..self.out_features {
                    let g = grads.get(i).cloned().unwrap_or(0.0);
                    if g == 0.0 {
                        continue;
                    }
                    let row = i * in_f;
                    if row + in_f > db_mut.len() {
                        continue;
                    }
                    for (j, &x) in input.iter().enumerate() {
                        if j >= in_f {
                            break;
                        }
                        db_mut[row + j] -= lr * g * x;
                    }
                }
            }
            WeightDatabase::GenomicQ4_0(db) => {
                // QAT de escala/min (calibración in-flight): mantiene `q` fijo.
                // W[i,j] = q[i,b,k]*scale[i,b] + min[i,b], con b = j/block_size.
                //   grad_scale[i,b] += Σ_k grad_W[i,b·bs+k] * q[i,b,k]
                //   grad_min[i,b]   += Σ_k grad_W[i,b·bs+k]
                // `grads` es ∂L/∂logits (tamaño = out_features); grad_W = grads[i]*input[j].
                let db_mut = Arc::make_mut(db);
                let bs = self.block_size;
                if bs == 0 || self.in_features == 0 {
                    return Ok(());
                }
                let n_blk = self.in_features / bs;
                let max_scale = 10.0f32;
                for i in 0..self.out_features {
                    let g = grads.get(i).cloned().unwrap_or(0.0);
                    if g == 0.0 {
                        continue;
                    }
                    let row_off = i * n_blk;
                    if row_off + n_blk > db_mut.len() {
                        continue;
                    }
                    for b in 0..n_blk {
                        let block = db_mut[row_off + b];
                        let mut g_scale = 0.0f32;
                        let mut g_min = 0.0f32;
                        for k in 0..bs {
                            let j = b * bs + k;
                            if j >= self.in_features {
                                break;
                            }
                            let x = input.get(j).cloned().unwrap_or(0.0);
                            let q = block.q_value(k) as f32;
                            g_scale += g * x * q;
                            g_min += g * x;
                        }
                        let mut nb = block;
                        nb.scale = half::f16::from_f32(
                            (block.scale.to_f32() - lr * g_scale).clamp(-max_scale, max_scale),
                        );
                        nb.min = half::f16::from_f32(block.min.to_f32() - lr * g_min);
                        db_mut[row_off + b] = nb;
                    }
                }
            }
            WeightDatabase::GenomicQ8_0(db) => {
                // QAT de escala: mantiene `q8` fijo. W = q8*scale.
                //   grad_scale[i,b] += Σ_k grad_W[i,b·bs+k] * q8[i,b,k]
                let db_mut = Arc::make_mut(db);
                let bs = self.block_size;
                if bs == 0 || self.in_features == 0 {
                    return Ok(());
                }
                let n_blk = self.in_features / bs;
                let max_scale = 100.0f32;
                for i in 0..self.out_features {
                    let g = grads.get(i).cloned().unwrap_or(0.0);
                    if g == 0.0 {
                        continue;
                    }
                    let row_off = i * n_blk;
                    if row_off + n_blk > db_mut.len() {
                        continue;
                    }
                    for b in 0..n_blk {
                        let block = db_mut[row_off + b];
                        let mut g_scale = 0.0f32;
                        for k in 0..bs {
                            let j = b * bs + k;
                            if j >= self.in_features {
                                break;
                            }
                            let x = input.get(j).cloned().unwrap_or(0.0);
                            g_scale += g * x * block.qs[k] as f32;
                        }
                        let mut nb = block;
                        nb.scale = half::f16::from_f32(
                            (block.scale.to_f32() - lr * g_scale).clamp(-max_scale, max_scale),
                        );
                        db_mut[row_off + b] = nb;
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
