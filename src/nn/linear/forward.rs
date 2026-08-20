// =============================================================================
// forward — Cómputo de filas y forwards de GenomicLinear (single/fused)
// =============================================================================
use rayon::prelude::*;

use crate::nn::linear::database::WeightDatabase;
use crate::nn::linear::GenomicLinear;

impl GenomicLinear {
    #[inline(always)]
    pub fn compute_single_row(
        &self,
        i: usize,
        input: &[f32],
        m_factors: &[f32; 4],
        n_blocks: usize,
    ) -> f32 {
        let mut sum = 0.0f32;
        match &self.weight_db {
            WeightDatabase::GenomicF32(db) => {
                let row_off = i * self.in_features;
                let row_weights = &db[row_off..row_off + self.in_features];
                if row_weights.len() == input.len() {
                    sum = unsafe { crate::compute::kernels::dot_product(input, row_weights) };
                } else {
                    sum = input
                        .iter()
                        .zip(row_weights.iter())
                        .map(|(x, w)| x * w)
                        .sum();
                }
            }
            WeightDatabase::Genomic2Bit(db) => {
                let row_off = i * n_blocks * self.stride;
                let db_slice = db
                    .get(row_off..row_off + n_blocks * self.stride)
                    .unwrap_or(&[]);
                let c_start = i * n_blocks * 4;
                let c_slice = if self.centroids.len() >= c_start + n_blocks * 4 {
                    &self.centroids[c_start..c_start + n_blocks * 4]
                } else if self.centroids.len() >= 4 {
                    &self.centroids[0..4]
                } else {
                    &[]
                };
                if !db_slice.is_empty() && !c_slice.is_empty() {
                    sum = unsafe {
                        crate::compute::kernels::genomic_dot_product(
                            db_slice,
                            input,
                            c_slice,
                            self.stride,
                            n_blocks,
                            m_factors,
                        )
                    };
                }
            }
            WeightDatabase::Genomic4Bit(db) => {
                let row_off = i * n_blocks * self.stride;
                let db_slice = db
                    .get(row_off..row_off + n_blocks * self.stride)
                    .unwrap_or(&[]);
                let c_start = i * n_blocks * 16;
                let c_slice = if self.centroids.len() >= c_start + n_blocks * 16 {
                    &self.centroids[c_start..c_start + n_blocks * 16]
                } else if self.centroids.len() >= 16 {
                    &self.centroids[0..16]
                } else {
                    &[]
                };
                if !db_slice.is_empty() && !c_slice.is_empty() {
                    sum = unsafe {
                        crate::compute::kernels::genomic_dot_product_4bit(
                            db_slice,
                            input,
                            c_slice,
                            self.stride,
                            n_blocks,
                        )
                    };
                }
            }
            WeightDatabase::GenomicQ4_0(db) => {
                let row_off = i * n_blocks;
                let db_slice = db.get(row_off..row_off + n_blocks).unwrap_or(&[]);
                if !db_slice.is_empty() {
                    sum = unsafe {
                        crate::compute::kernels::genomic_dot_product_q4_0(db_slice, input, n_blocks)
                    };
                }
            }
            WeightDatabase::GenomicQ8_0(db) => {
                let row_off = i * n_blocks;
                let db_slice = db.get(row_off..row_off + n_blocks).unwrap_or(&[]);
                if !db_slice.is_empty() {
                    sum = unsafe {
                        crate::compute::kernels::genomic_dot_product_q8_0(db_slice, input, n_blocks)
                    };
                }
            }
        }
        let a_s = self.anchor_row_ptrs.get(i).copied().unwrap_or(0);
        let a_e = self.anchor_row_ptrs.get(i + 1).copied().unwrap_or(a_s);
        if a_s < a_e && a_e <= self.anchor_indices.len() && a_e <= self.anchor_values.len() {
            for k in a_s..a_e {
                let col_idx = (self.anchor_indices[k] as usize) % self.in_features;
                if let Some(&in_val) = input.get(col_idx) {
                    sum += in_val * self.anchor_values[k].to_f32();
                }
            }
        }
        if let Some(&b_val) = self.bias.get(i) {
            sum += b_val;
        }
        sum
    }

    pub fn forward_core(
        &self,
        input: Vec<f32>,
        modulation_factors: Option<[f32; 4]>,
        _activate_rna: bool,
    ) -> Result<Vec<f32>, String> {
        let n_blocks = self.in_features / self.block_size;
        let m_factors = modulation_factors.unwrap_or([1.0f32; 4]);
        // Procesar por bloques contiguos (una tarea rayon por rango), no una tarea
        // por fila: para lineales grandes (p.ej. lm_head con vocab 151936) la
        // sobrecarga de una tarea por fila era el cuello de botella del entrenamiento.
        let out = self.out_features;
        let n_threads = rayon::current_num_threads();
        let results: Vec<f32> = (0..n_threads)
            .into_par_iter()
            .flat_map(|c| {
                let start = c * out / n_threads;
                let end = (((c + 1) * out) / n_threads).min(out);
                let mut chunk = Vec::with_capacity(end - start);
                for i in start..end {
                    chunk.push(self.compute_single_row(i, &input, &m_factors, n_blocks));
                }
                chunk
            })
            .collect();
        Ok(results)
    }

    pub fn forward_fused_3(
        l1: &GenomicLinear,
        l2: &GenomicLinear,
        l3: &GenomicLinear,
        input: &[f32],
        modulation_factors: Option<[f32; 4]>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let n_blocks = l1.in_features / l1.block_size;
        let m_factors = modulation_factors.unwrap_or([1.0f32; 4]);
        let o1 = l1.out_features;
        let o2 = l2.out_features;
        let o3 = l3.out_features;
        let total = o1 + o2 + o3;

        let n_threads = rayon::current_num_threads();
        let fused_out: Vec<f32> = (0..n_threads)
            .into_par_iter()
            .flat_map(|c| {
                let start = c * total / n_threads;
                let end = (((c + 1) * total) / n_threads).min(total);
                let mut chunk = Vec::with_capacity(end - start);
                for idx in start..end {
                    let v = if idx < o1 {
                        l1.compute_single_row(idx, input, &m_factors, n_blocks)
                    } else if idx < o1 + o2 {
                        l2.compute_single_row(idx - o1, input, &m_factors, n_blocks)
                    } else {
                        l3.compute_single_row(idx - o1 - o2, input, &m_factors, n_blocks)
                    };
                    chunk.push(v);
                }
                chunk
            })
            .collect();

        let res1 = fused_out[0..o1].to_vec();
        let res2 = fused_out[o1..o1 + o2].to_vec();
        let res3 = fused_out[o1 + o2..total].to_vec();
        Ok((res1, res2, res3))
    }

    pub fn forward_fused_2(
        l1: &GenomicLinear,
        l2: &GenomicLinear,
        input: &[f32],
        modulation_factors: Option<[f32; 4]>,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let n_blocks = l1.in_features / l1.block_size;
        let m_factors = modulation_factors.unwrap_or([1.0f32; 4]);
        let o1 = l1.out_features;
        let o2 = l2.out_features;
        let total = o1 + o2;

        let n_threads = rayon::current_num_threads();
        let fused_out: Vec<f32> = (0..n_threads)
            .into_par_iter()
            .flat_map(|c| {
                let start = c * total / n_threads;
                let end = (((c + 1) * total) / n_threads).min(total);
                let mut chunk = Vec::with_capacity(end - start);
                for idx in start..end {
                    let v = if idx < o1 {
                        l1.compute_single_row(idx, input, &m_factors, n_blocks)
                    } else {
                        l2.compute_single_row(idx - o1, input, &m_factors, n_blocks)
                    };
                    chunk.push(v);
                }
                chunk
            })
            .collect();

        let res1 = fused_out[0..o1].to_vec();
        let res2 = fused_out[o1..total].to_vec();
        Ok((res1, res2))
    }

    pub fn get_row_core(&self, idx: usize) -> Result<Vec<f32>, String> {
        let mut res = vec![0.0f32; self.in_features];
        if self.out_features == 0 || self.in_features == 0 {
            return Ok(res);
        }
        let safe_idx = idx % self.out_features;
        let n_blocks = self.in_features / self.block_size;

        match &self.weight_db {
            WeightDatabase::GenomicF32(db) => {
                let row_off = safe_idx * self.in_features;
                if row_off + self.in_features <= db.len() {
                    res.copy_from_slice(&db[row_off..row_off + self.in_features]);
                }
            }
            WeightDatabase::Genomic2Bit(db) => {
                let row_off = safe_idx * n_blocks * self.stride;
                let req_bytes = n_blocks * self.stride;
                if row_off + req_bytes <= db.len() {
                    for b in 0..n_blocks {
                        let c_off = (safe_idx * n_blocks + b) * 4;
                        let c_slice = if self.centroids.len() == 4 {
                            &self.centroids[0..4]
                        } else if c_off + 4 <= self.centroids.len() {
                            &self.centroids[c_off..c_off + 4]
                        } else if !self.centroids.is_empty() {
                            &self.centroids[0..4.min(self.centroids.len())]
                        } else {
                            &[0.0, 0.0, 0.0, 0.0]
                        };
                        if let Ok(decoded) = crate::compute::math::dequantize_embedding_core(
                            &db[row_off + b * self.stride..row_off + (b + 1) * self.stride],
                            self.block_size,
                            Some(c_slice),
                        ) {
                            res[b * self.block_size..(b + 1) * self.block_size]
                                .copy_from_slice(&decoded);
                        }
                    }
                }
            }
            WeightDatabase::Genomic4Bit(db) => {
                let row_off = safe_idx * n_blocks * self.stride;
                let req_bytes = n_blocks * self.stride;
                if row_off + req_bytes <= db.len() {
                    for b in 0..n_blocks {
                        let c_off = (safe_idx * n_blocks + b) * 16;
                        let centroids = if self.centroids.len() == 16 {
                            &self.centroids[0..16]
                        } else if c_off + 16 <= self.centroids.len() {
                            &self.centroids[c_off..c_off + 16]
                        } else if !self.centroids.is_empty() {
                            &self.centroids[0..16.min(self.centroids.len())]
                        } else {
                            &[0.0; 16]
                        };
                        for k in 0..self.stride {
                            let byte = db[row_off + b * self.stride + k];
                            res[b * self.block_size + k * 2] = centroids[(byte >> 4) as usize];
                            res[b * self.block_size + k * 2 + 1] =
                                centroids[(byte & 0x0F) as usize];
                        }
                    }
                }
            }
            WeightDatabase::GenomicQ4_0(db) => {
                let row_off = safe_idx * n_blocks;
                if row_off + n_blocks <= db.len() {
                    for b in 0..n_blocks {
                        let block = &db[row_off + b];
                        for k in 0..32 {
                            res[b * self.block_size + k] = block.dequantize_weight(k);
                        }
                    }
                }
            }
            WeightDatabase::GenomicQ8_0(db) => {
                let row_off = safe_idx * n_blocks;
                if row_off + n_blocks <= db.len() {
                    for b in 0..n_blocks {
                        let block = &db[row_off + b];
                        for k in 0..32 {
                            res[b * self.block_size + k] = block.dequantize_weight(k);
                        }
                    }
                }
            }
        }
        Ok(res)
    }
}
