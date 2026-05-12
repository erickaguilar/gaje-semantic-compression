use pyo3::prelude::*;
use rayon::prelude::*;
use crate::kernels::*;
use half::f16;

#[pyclass]
pub struct GenomicLinear {
    #[pyo3(get)]
    pub database: Vec<u8>,
    pub epigenetic_database: Vec<u8>,
    pub triplet_database: Vec<u8>,
    pub anchors: Vec<f16>,
    pub centroids: Vec<f16>,
    pub epigenetic_centroids: Vec<f16>,
    pub triplet_centroids: Vec<f16>,
    #[pyo3(get)]
    pub out_features: usize,
    #[pyo3(get)]
    pub in_features: usize,
    #[pyo3(get)]
    pub block_size: usize,
    pub rmsnorm_weight: Vec<f32>,
    pub eps: f32,
    pub precision_mask: Vec<u8>,
    pub stride: usize,
}

#[pymethods]
impl GenomicLinear {
    #[new]
    #[pyo3(signature = (database, anchors, centroids, out_features, in_features, block_size, rmsnorm_weight=Vec::new(), eps=1e-6, precision_mask=Vec::new(), epigenetic_database=Vec::new(), epigenetic_centroids=Vec::new(), triplet_database=Vec::new(), triplet_centroids=Vec::new()))]
    pub fn new(database: Vec<u8>, anchors: Vec<f32>, centroids: Vec<f32>, out_features: usize, in_features: usize, block_size: usize, rmsnorm_weight: Vec<f32>, eps: f32, precision_mask: Vec<u8>, epigenetic_database: Vec<u8>, epigenetic_centroids: Vec<f32>, triplet_database: Vec<u8>, triplet_centroids: Vec<f32>) -> Self {
        let stride = block_size / 4;
        let anchors = anchors.into_iter().map(f16::from_f32).collect();
        let centroids = centroids.into_iter().map(f16::from_f32).collect();
        let epigenetic_centroids = epigenetic_centroids.into_iter().map(f16::from_f32).collect();
        let triplet_centroids = triplet_centroids.into_iter().map(f16::from_f32).collect();
        GenomicLinear { database, epigenetic_database, triplet_database, anchors, centroids, epigenetic_centroids, triplet_centroids, out_features, in_features, block_size, rmsnorm_weight, eps, precision_mask, stride }
    }

    #[getter]
    pub fn anchors(&self) -> Vec<f32> {
        self.anchors.iter().map(|&x| x.to_f32()).collect()
    }

    #[getter]
    pub fn centroids(&self) -> Vec<f32> {
        self.centroids.iter().map(|&x| x.to_f32()).collect()
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
                // Convert f16 centroids to f32 for NEON (SIMD optimization)
                let c32: Vec<f32> = row_centroids.iter().map(|&c| c.to_f32()).collect();
                unsafe { genomic_dot_product_neon(row_weights, &input, &c32, self.stride, n_blocks) }
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
                                0b00 => c[0].to_f32(), 0b01 => c[1].to_f32(), 0b11 => c[2].to_f32(), 0b10 => c[3].to_f32(),
                                _ => 0.0
                            };

                            if mode >= 1 && has_epi {
                                let ce = &self.epigenetic_centroids[c_offset..c_offset + 4];
                                let eb = (self.epigenetic_database[block_start + k] >> shift) & 0b11;
                                val += match eb { 0b00 => ce[0].to_f32(), 0b01 => ce[1].to_f32(), 0b11 => ce[2].to_f32(), 0b10 => ce[3].to_f32(), _ => 0.0 };
                            }

                            if mode >= 2 && has_tri {
                                let ct = &self.triplet_centroids[c_offset..c_offset + 4];
                                let tb = (self.triplet_database[block_start + k] >> shift) & 0b11;
                                val += match tb { 0b00 => ct[0].to_f32(), 0b01 => ct[1].to_f32(), 0b11 => ct[2].to_f32(), 0b10 => ct[3].to_f32(), _ => 0.0 };
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
                let a32: Vec<f32> = anchor_row.iter().map(|&a| a.to_f32()).collect();
                row_sum += unsafe { dot_product_neon(&a32, &input) };
            }
            row_sum.clamp(-64.0, 64.0)
        }).collect();

        Ok(results)
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
                            0b00 => c[0].to_f32(), 0b01 => c[1].to_f32(), 0b11 => c[2].to_f32(), 0b10 => c[3].to_f32(),
                            _ => 0.0
                        };
                        if mode >= 1 && has_epi {
                            let ce = &self.epigenetic_centroids[c_offset..c_offset + 4];
                            let eb = (self.epigenetic_database[block_start + k] >> shift) & 0b11;
                            val += match eb { 0b00 => ce[0].to_f32(), 0b01 => ce[1].to_f32(), 0b11 => ce[2].to_f32(), 0b10 => ce[3].to_f32(), _ => 0.0 };
                        }
                        if mode >= 2 && has_tri {
                            let ct = &self.triplet_centroids[c_offset..c_offset + 4];
                            let tb = (self.triplet_database[block_start + k] >> shift) & 0b11;
                            val += match tb { 0b00 => ct[0].to_f32(), 0b01 => ct[1].to_f32(), 0b11 => ct[2].to_f32(), 0b10 => ct[3].to_f32(), _ => 0.0 };
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
                            let current = self.centroids[c_offset + c_idx].to_f32();
                            self.centroids[c_offset + c_idx] = f16::from_f32(current - grad_scale * input_block[dims]); 
                            if mode >= 1 && has_epi {
                                let current_e = self.epigenetic_centroids[c_offset + c_idx].to_f32();
                                self.epigenetic_centroids[c_offset + c_idx] = f16::from_f32(current_e - grad_scale * 0.5 * input_block[dims]);
                            }
                            if mode >= 2 && has_tri {
                                let current_t = self.triplet_centroids[c_offset + c_idx].to_f32();
                                self.triplet_centroids[c_offset + c_idx] = f16::from_f32(current_t - grad_scale * 0.25 * input_block[dims]);
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

#[pyclass]
pub struct GenomicAttention {
    #[pyo3(get)]
    pub n_head: usize,
    #[pyo3(get)]
    pub n_head_kv: usize,
    #[pyo3(get)]
    pub head_dim: usize,
    pub k_cache: Vec<Vec<f16>>,
    pub v_cache: Vec<Vec<f16>>,
    pub rmsnorm_weight: Vec<f32>,
    pub eps: f32,
}

#[pymethods]
impl GenomicAttention {
    #[new]
    #[pyo3(signature = (n_head, n_head_kv, head_dim, rmsnorm_weight=Vec::new(), eps=1e-6))]
    pub fn new(n_head: usize, n_head_kv: usize, head_dim: usize, rmsnorm_weight: Vec<f32>, eps: f32) -> Self {
        GenomicAttention { n_head, n_head_kv, head_dim, k_cache: Vec::new(), v_cache: Vec::new(), rmsnorm_weight, eps }
    }

    #[getter]
    pub fn k_cache(&self) -> Vec<Vec<f32>> {
        self.k_cache.iter().map(|v| v.iter().map(|&x| x.to_f32()).collect()).collect()
    }

    #[getter]
    pub fn v_cache(&self) -> Vec<Vec<f32>> {
        self.v_cache.iter().map(|v| v.iter().map(|&x| x.to_f32()).collect()).collect()
    }

    #[getter]
    pub fn k_cache_len(&self) -> usize {
        self.k_cache.len()
    }


    pub fn apply_rmsnorm(&self, input: Vec<f32>) -> PyResult<Vec<f32>> {
        if self.rmsnorm_weight.is_empty() { return Ok(input); }
        Ok(unsafe { rms_norm_neon(&input, &self.rmsnorm_weight, self.eps) })
    }

    pub fn forward_attention(&mut self, q: Vec<f32>, k: Vec<f32>, v: Vec<f32>, pos: usize) -> PyResult<Vec<f32>> {
        let head_dim = self.head_dim;
        let n_head = self.n_head;
        let n_head_kv = self.n_head_kv;
        let n_groups = n_head / n_head_kv;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut q_rope = q.clone();
        let k_rope_f32 = k.clone();

        // Parallel RoPE applying with correct theta
        q_rope.par_chunks_exact_mut(head_dim).for_each(|h_q| {
            for i in 0..(head_dim / 2) {
                let freq = 1.0 / 10000.0f32.powf((2 * i) as f32 / head_dim as f32);
                let theta = pos as f32 * freq;
                let cos = theta.cos();
                let sin = theta.sin();
                let v0 = h_q[i];
                let v1 = h_q[i + head_dim / 2];
                h_q[i] = (v0 * cos - v1 * sin) * scale;
                h_q[i + head_dim / 2] = (v0 * sin + v1 * cos) * scale;
            }
        });

        let mut k_rope_processed = k_rope_f32.clone();
        k_rope_processed.par_chunks_exact_mut(head_dim).for_each(|h_k| {
            for i in 0..(head_dim / 2) {
                let freq = 1.0 / 10000.0f32.powf((2 * i) as f32 / head_dim as f32);
                let theta = pos as f32 * freq;
                let cos = theta.cos();
                let sin = theta.sin();
                let v0 = h_k[i];
                let v1 = h_k[i + head_dim / 2];
                h_k[i] = v0 * cos - v1 * sin;
                h_k[i + head_dim / 2] = v0 * sin + v1 * cos;
            }
        });

        // Store in Cache as F16
        self.k_cache.push(k_rope_processed.into_iter().map(f16::from_f32).collect());
        self.v_cache.push(v.into_iter().map(f16::from_f32).collect());

        let k_cache = &self.k_cache;
        let v_cache = &self.v_cache;

        let attn_out: Vec<f32> = (0..n_head).into_par_iter().flat_map(|h| {
            let kv_h = h / n_groups;
            let q_slice = &q_rope[h * head_dim .. (h + 1) * head_dim];

            let mut scores = vec![0.0f32; k_cache.len()];
            let mut max_score = -f32::INFINITY;

            for t in 0..k_cache.len() {
                let k_slice = &k_cache[t][kv_h * head_dim .. (kv_h + 1) * head_dim];
                let mut score = 0.0f32;
                for i in 0..head_dim {
                    score += q_slice[i] * k_slice[i].to_f32();
                }
                scores[t] = score;
                if score > max_score { max_score = score; }
            }

            let mut sum_exp = 0.0f32;
            for t in 0..scores.len() {
                scores[t] = (scores[t] - max_score).exp();
                sum_exp += scores[t];
            }
            let inv_sum = 1.0 / (sum_exp + 1e-9);

            let mut head_out = vec![0.0f32; head_dim];
            for t in 0..scores.len() {
                let weight = scores[t] * inv_sum;
                let v_slice = &v_cache[t][kv_h * head_dim .. (kv_h + 1) * head_dim];
                for i in 0..head_dim {
                    head_out[i] += weight * v_slice[i].to_f32();
                }
            }
            head_out
        }).collect();

        Ok(attn_out)
    }

    pub fn clear_cache(&mut self) -> PyResult<()> {
        self.k_cache.clear();
        self.v_cache.clear();
        Ok(())
    }
}

#[pyclass]
pub struct GenomicSwiGLU {
    pub w_gate: Vec<u8>, pub w_up: Vec<u8>, pub centroids: Vec<f16>,
    pub out_features: usize, pub in_features: usize, pub block_size: usize, pub stride: usize,
}

#[pymethods]
impl GenomicSwiGLU {
    #[new]
    pub fn new(w_gate: Vec<u8>, w_up: Vec<u8>, centroids: Vec<f32>, out_features: usize, in_features: usize, block_size: usize) -> Self {
        let centroids = centroids.into_iter().map(f16::from_f32).collect();
        GenomicSwiGLU { w_gate, w_up, centroids, out_features, in_features, block_size, stride: block_size / 4 }
    }
    pub fn forward(&self, input: Vec<f32>) -> PyResult<Vec<f32>> {
        let n_blocks = self.in_features / self.block_size;
        let silu = |x: f32| x / (1.0 + (-x).exp());

        let results: Vec<f32> = (0..self.out_features).into_par_iter().map(|i| {
            let mut gate_sum = 0.0f32;
            let mut up_sum = 0.0f32;
            let row_offset = i * n_blocks * self.stride;
            for j in 0..n_blocks {
                let block_start = row_offset + j * self.stride;
                let g_weights = &self.w_gate[block_start..block_start + self.stride];
                let u_weights = &self.w_up[block_start..block_start + self.stride];
                let input_block = &input[j * self.block_size .. (j + 1) * self.block_size];
                let c_offset = (i * n_blocks + j) * 4;
                let c = &self.centroids[c_offset..c_offset + 4];

                let mut dims = 0;
                for k in 0..self.stride {
                    let g_byte = g_weights[k];
                    let u_byte = u_weights[k];
                    for s in 0..4 {
                        let shift = (3 - s) * 2;
                        let g_bits = (g_byte >> shift) & 0b11;
                        let u_bits = (u_byte >> shift) & 0b11;
                        let g_val = match g_bits { 0b00 => c[0].to_f32(), 0b01 => c[1].to_f32(), 0b11 => c[2].to_f32(), 0b10 => c[3].to_f32(), _ => 0.0 };
                        let u_val = match u_bits { 0b00 => c[0].to_f32(), 0b01 => c[1].to_f32(), 0b11 => c[2].to_f32(), 0b10 => c[3].to_f32(), _ => 0.0 };
                        gate_sum += input_block[dims] * g_val;
                        up_sum += input_block[dims] * u_val;
                        dims += 1;
                    }
                }
            }
            silu(gate_sum) * up_sum
        }).collect();
        Ok(results)
    }
}

