use pyo3::prelude::*;
use rayon::prelude::*;
use crate::kernels::*;
use half::f16;

#[pyclass]
#[derive(Clone)]
pub struct GenomicLinear {
    #[pyo3(get)]
    pub database: Vec<u8>,
    #[pyo3(get)]
    pub epigenetic_database: Vec<u8>,
    #[pyo3(get)]
    pub triplet_database: Vec<u8>,
    pub anchors: Vec<f16>,
    #[pyo3(get)]
    pub centroids: Vec<f32>,
    #[pyo3(get)]
    pub epigenetic_centroids: Vec<f32>,
    #[pyo3(get)]
    pub triplet_centroids: Vec<f32>,
    #[pyo3(get)]
    pub out_features: usize,
    #[pyo3(get)]
    pub in_features: usize,
    #[pyo3(get)]
    pub block_size: usize,
    #[pyo3(get)]
    pub rmsnorm_weight: Vec<f32>,
    #[pyo3(get)]
    pub eps: f32,
    #[pyo3(get)]
    pub precision_mask: Vec<u8>,
    #[pyo3(get)]
    pub stride: usize,
}

#[pymethods]
impl GenomicLinear {
    #[new]
    #[pyo3(signature = (database, anchors, centroids, out_features, in_features, block_size, rmsnorm_weight=Vec::new(), eps=1e-6, precision_mask=Vec::new(), epigenetic_database=Vec::new(), epigenetic_centroids=Vec::new(), triplet_database=Vec::new(), triplet_centroids=Vec::new()))]
    pub fn new(database: Vec<u8>, anchors: Vec<f32>, centroids: Vec<f32>, out_features: usize, in_features: usize, block_size: usize, rmsnorm_weight: Vec<f32>, eps: f32, precision_mask: Vec<u8>, epigenetic_database: Vec<u8>, epigenetic_centroids: Vec<f32>, triplet_database: Vec<u8>, triplet_centroids: Vec<f32>) -> Self {
        let stride = block_size / 4;
        let anchors = anchors.into_iter().map(f16::from_f32).collect();
        GenomicLinear { database, epigenetic_database, triplet_database, anchors, centroids, epigenetic_centroids, triplet_centroids, out_features, in_features, block_size, rmsnorm_weight, eps, precision_mask, stride }
    }

    #[getter]
    pub fn anchors(&self) -> Vec<f32> {
        self.anchors.iter().map(|&x| x.to_f32()).collect()
    }

    pub fn forward(&self, mut input: Vec<f32>) -> PyResult<Vec<f32>> {
        if !self.rmsnorm_weight.is_empty() {
            input = unsafe { rms_norm_neon(&input, &self.rmsnorm_weight, self.eps) };
        }

        let n_blocks = self.in_features / self.block_size;
        let has_anchors = !self.anchors.is_empty();
        let has_mask = !self.precision_mask.is_empty();
        let has_epi = !self.epigenetic_database.is_empty();
        let has_tri = !self.triplet_database.is_empty();

        let results: Vec<f32> = (0..self.out_features).into_par_iter().map(|i| {
            let row_offset = i * n_blocks * self.stride;

            let mut row_sum = if !has_mask && !has_epi && !has_tri {
                let row_weights = &self.database[row_offset .. row_offset + n_blocks * self.stride];
                let row_centroids = &self.centroids[i * n_blocks * 4 .. (i + 1) * n_blocks * 4];
                unsafe { genomic_dot_product_neon(row_weights, &input, row_centroids, self.stride, n_blocks) }
            } else {
                let mut sum = 0.0f32;
                for j in 0..n_blocks {
                    let block_start = row_offset + j * self.stride;
                    let c_offset = (i * n_blocks + j) * 4;
                    let c = &self.centroids[c_offset..c_offset + 4];
                    let input_block = &input[j * self.block_size .. (j + 1) * self.block_size];
                    let weights = &self.database[block_start .. block_start + self.stride];

                    let mut dims = 0;
                    for k in 0..self.stride {
                        let byte = weights[k];
                        let mode = if has_mask { self.precision_mask[j * self.stride + k] } else { 0 };

                        for s in 0..4 {
                            let shift = (3 - s) * 2;
                            let bits = (byte >> shift) & 0b11;
                            let mut val = match bits {
                                0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3],
                                _ => 0.0
                            };

                            if mode >= 1 && has_epi {
                                let ce = &self.epigenetic_centroids[c_offset..c_offset + 4];
                                let eb = (self.epigenetic_database[block_start + k] >> shift) & 0b11;
                                val += match eb { 0b00 => ce[0], 0b01 => ce[1], 0b11 => ce[2], 0b10 => ce[3], _ => 0.0 };
                            }

                            if mode >= 2 && has_tri {
                                let ct = &self.triplet_centroids[c_offset..c_offset + 4];
                                let tb = (self.triplet_database[block_start + k] >> shift) & 0b11;
                                val += match tb { 0b00 => ct[0], 0b01 => ct[1], 0b11 => ct[2], 0b10 => ct[3], _ => 0.0 };
                            }

                            sum += input_block[dims] * val;
                            dims += 1;
                        }
                    }
                }
                sum
            };

            if has_anchors {
                let anchor_row = &self.anchors[i * self.in_features .. (i + 1) * self.in_features];
                // Convert f16 anchors to f32 temporarily to use NEON
                let a32: Vec<f32> = anchor_row.iter().map(|&a| a.to_f32()).collect();
                row_sum += unsafe { dot_product_neon(&a32, &input) };
            }
            row_sum.clamp(-128.0, 128.0)
        }).collect();

        Ok(results)
    }

    pub fn get_row(&self, idx: usize) -> PyResult<Vec<f32>> {
        if idx >= self.out_features {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Index {} out of bounds for vocab size (out_features) {}", idx, self.out_features)));
        }
        let n_blocks = self.in_features / self.block_size;
        let mut res = vec![0.0f32; self.in_features];
        
        let row_start = idx * n_blocks * self.stride;
        let row_dna = &self.database[row_start .. row_start + n_blocks * self.stride];
        
        for b in 0..n_blocks {
            let block_dna = &row_dna[b * self.stride .. (b + 1) * self.stride];
            let c_offset = (idx * n_blocks + b) * 4;
            let centroids_f32 = &self.centroids[c_offset .. c_offset + 4];
            
            let decoded = crate::utils::dequantize_embedding(block_dna.to_vec(), self.block_size, Some(centroids_f32.to_vec()))?;
            for i in 0..self.block_size {
                res[b * self.block_size + i] = decoded[i];
            }
        }
        
        if !self.anchors.is_empty() {
            let anchor_row = &self.anchors[idx * self.in_features .. (idx + 1) * self.in_features];
            for i in 0..self.in_features {
                res[i] += anchor_row[i].to_f32();
            }
        }
        
        Ok(res)
    }

    pub fn refine_centroids(&mut self, mut input: Vec<f32>, target: Vec<f32>, lr: f32) -> PyResult<()> {
        if !self.rmsnorm_weight.is_empty() {
            input = unsafe { rms_norm_neon(&input, &self.rmsnorm_weight, self.eps) };
        }
        let n_blocks = self.in_features / self.block_size;
        let has_anchors = !self.anchors.is_empty();
        let has_mask = !self.precision_mask.is_empty();
        let has_epi = !self.epigenetic_database.is_empty();
        let has_tri = !self.triplet_database.is_empty();

        for i in 0..self.out_features {
            let mut row_sum = 0.0f32;
            let row_offset = i * n_blocks * self.stride;
            for j in 0..n_blocks {
                let block_start = row_offset + j * self.stride;
                let c_offset = (i * n_blocks + j) * 4;
                let c = &self.centroids[c_offset..c_offset + 4];
                let input_block = &input[j * self.block_size .. (j + 1) * self.block_size];
                let weights = &self.database[block_start .. block_start + self.stride];
                let mut dims = 0;
                for k in 0..self.stride {
                    let byte = weights[k];
                    let mode = if has_mask { self.precision_mask[j * self.stride + k] } else { 0 };
                    for s in 0..4 {
                        let shift = (3 - s) * 2;
                        let bits = (byte >> shift) & 0b11;
                        let mut val = match bits {
                            0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3],
                            _ => 0.0
                        };
                        if mode >= 1 && has_epi {
                            let ce = &self.epigenetic_centroids[c_offset..c_offset + 4];
                            let eb = (self.epigenetic_database[block_start + k] >> shift) & 0b11;
                            val += match eb { 0b00 => ce[0], 0b01 => ce[1], 0b11 => ce[2], 0b10 => ce[3], _ => 0.0 };
                        }
                        if mode >= 2 && has_tri {
                            let ct = &self.triplet_centroids[c_offset..c_offset + 4];
                            let tb = (self.triplet_database[block_start + k] >> shift) & 0b11;
                            val += match tb { 0b00 => ct[0], 0b01 => ct[1], 0b11 => ct[2], 0b10 => ct[3], _ => 0.0 };
                        }
                        row_sum += input_block[dims] * val;
                        dims += 1;
                    }
                }
            }
            if has_anchors {
                let anchor_row = &self.anchors[i * self.in_features .. (i + 1) * self.in_features];
                for (j, &a) in anchor_row.iter().enumerate() { row_sum += a.to_f32() * input[j]; }
            }

            let grad_scale = (row_sum - target[i]) * lr;
            for j in 0..n_blocks {
                let block_start = row_offset + j * self.stride;
                let c_offset = (i * n_blocks + j) * 4;
                let input_block = &input[j * self.block_size .. (j + 1) * self.block_size];
                let weights = &self.database[block_start .. block_start + self.stride];
                let mut dims = 0;
                for k in 0..self.stride {
                    let byte = weights[k];
                    let mode = if has_mask { self.precision_mask[j * self.stride + k] } else { 0 };
                    for s in 0..4 {
                        let shift = (3 - s) * 2;
                        let bits = (byte >> shift) & 0b11;
                        let c_idx = match bits { 0b00 => 0, 0b01 => 1, 0b11 => 2, 0b10 => 3, _ => 4 };
                        if c_idx < 4 { 
                            let current = self.centroids[c_offset + c_idx];
                            self.centroids[c_offset + c_idx] = current - grad_scale * input_block[dims]; 
                            if mode >= 1 && has_epi {
                                let current_e = self.epigenetic_centroids[c_offset + c_idx];
                                self.epigenetic_centroids[c_offset + c_idx] = current_e - grad_scale * 0.5 * input_block[dims];
                            }
                            if mode >= 2 && has_tri {
                                let current_t = self.triplet_centroids[c_offset + c_idx];
                                self.triplet_centroids[c_offset + c_idx] = current_t - grad_scale * 0.25 * input_block[dims];
                            }
                        }
                        dims += 1;
                    }
                }
            }
        }
        Ok(())
    }
}
