use pyo3::prelude::*;
use rayon::prelude::*;
use crate::kernels::*;
use crate::utils::*;

#[pyclass]
pub struct GenomicLinear {
    #[pyo3(get)]
    pub database: Vec<u8>,
    pub epigenetic_database: Vec<u8>,
    pub triplet_database: Vec<u8>,
    #[pyo3(get)]
    pub anchors: Vec<f32>,
    #[pyo3(get, set)]
    pub centroids: Vec<f32>,
    pub epigenetic_centroids: Vec<f32>,
    pub triplet_centroids: Vec<f32>,
    #[pyo3(get)]
    pub out_features: usize,
    #[pyo3(get)]
    pub in_features: usize,
    #[pyo3(get)]
    pub block_size: usize,
    #[pyo3(get, set)]
    pub rmsnorm_weight: Vec<f32>,
    #[pyo3(get, set)]
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
        GenomicLinear { database, epigenetic_database, triplet_database, anchors, centroids, epigenetic_centroids, triplet_centroids, out_features, in_features, block_size, rmsnorm_weight, eps, precision_mask, stride }
    }
    pub fn forward(&self, mut input_vector: Vec<f32>) -> PyResult<Vec<f32>> {
        if !self.rmsnorm_weight.is_empty() { input_vector = unsafe { rms_norm_neon(&input_vector, &self.rmsnorm_weight, self.eps) }; }
        let n_blocks_per_row = self.in_features / self.block_size;
        let has_anchors = !self.anchors.is_empty();
        let database_len = self.database.len();
        let has_mask = !self.precision_mask.is_empty();
        let has_epi = !self.epigenetic_database.is_empty();
        let has_tri = !self.triplet_database.is_empty();

        let results: Vec<f32> = (0..self.out_features).into_par_iter().map(|i| {
            let mut row_sum = 0.0f32;
            let row_offset = i * n_blocks_per_row * self.stride;
            if row_offset + n_blocks_per_row * self.stride <= database_len {
                for j in 0..n_blocks_per_row {
                    let block_start = row_offset + j * self.stride;
                    let block_weights = &self.database[block_start..block_start + self.stride];
                    let input_block = &input_vector[j * self.block_size .. (j + 1) * self.block_size];
                    let c_offset = (i * n_blocks_per_row + j) * 4;
                    let c = &self.centroids[c_offset..c_offset + 4];
                    
                    let mut dims = 0;
                    for k in 0..self.stride {
                        let mode = if has_mask { self.precision_mask[j * self.stride + k] } else { 0 };
                        let byte = block_weights[k];
                        
                        for s in 0..4 {
                            let shift = (3 - s) * 2;
                            let bits = (byte >> shift) & 0b11;
                            let mut val = match bits { 0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0 };
                            
                            if mode >= 1 && has_epi {
                                let epi_byte = self.epigenetic_database[block_start + k];
                                let eb_bits = (epi_byte >> shift) & 0b11;
                                let ce = &self.epigenetic_centroids[c_offset..c_offset + 4];
                                val += match eb_bits { 0b00 => ce[0], 0b01 => ce[1], 0b11 => ce[2], 0b10 => ce[3], _ => 0.0 };
                            }
                            
                            if mode >= 2 && has_tri {
                                let tri_byte = self.triplet_database[block_start + k];
                                let tb_bits = (tri_byte >> shift) & 0b11;
                                let ct = &self.triplet_centroids[c_offset..c_offset + 4];
                                val += match tb_bits { 0b00 => ct[0], 0b01 => ct[1], 0b11 => ct[2], 0b10 => ct[3], _ => 0.0 };
                            }

                            row_sum += input_block[dims] * val;
                            dims += 1;
                        }
                    }
                }
            }
            if has_anchors && (i + 1) * self.in_features <= self.anchors.len() {
                let anchor_row = &self.anchors[i * self.in_features .. (i + 1) * self.in_features];
                row_sum += unsafe { dot_product_neon(anchor_row, &input_vector) };
            }
            row_sum.clamp(-100.0, 100.0)
        }).collect();
        Ok(results)
    }
    pub fn refine_centroids(&mut self, input_vector: Vec<f32>, target_output: Vec<f32>, lr: f32) -> PyResult<()> {
        let n_blocks_per_row = self.in_features / self.block_size;
        let mut activations = input_vector.clone();
        if !self.rmsnorm_weight.is_empty() { activations = unsafe { rms_norm_neon(&activations, &self.rmsnorm_weight, self.eps) }; }
        let current_output = self.forward(input_vector)?;
        let block_scale = 1.0 / self.block_size as f32;
        for i in 0..self.out_features {
            let error = current_output[i] - target_output[i];
            let row_offset = i * n_blocks_per_row * self.stride;
            for j in 0..n_blocks_per_row {
                let block_start = row_offset + j * self.stride;
                let block_weights = &self.database[block_start..block_start + self.stride];
                let input_block = &activations[j * self.block_size .. (j + 1) * self.block_size];
                let c_offset = (i * n_blocks_per_row + j) * 4;
                let mut dims = 0;
                for k in 0..self.stride {
                    let byte = block_weights[k];
                    for s in 0..4 {
                        let shift = (3 - s) * 2;
                        let bits = (byte >> shift) & 0b11;
                        let c_idx = match bits { 0b00 => 0, 0b01 => 1, 0b11 => 2, 0b10 => 3, _ => 0 };
                        let grad = error * input_block[dims] * block_scale;
                        self.centroids[c_offset + c_idx] -= lr * grad;
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
    pub q_database: Vec<u8>, pub k_database: Vec<u8>, pub v_database: Vec<u8>,
    pub centroids: Vec<f32>, pub stride: usize, pub n_head: usize, pub n_head_kv: usize,
    pub head_dim: usize, pub k_cache: Vec<Vec<u8>>, pub v_cache: Vec<Vec<u8>>,
    pub precision_mask: Vec<u8>,
}

#[pymethods]
impl GenomicAttention {
    #[new]
    #[pyo3(signature = (q, k, v, centroids, stride, n_head, n_head_kv, head_dim, rmsnorm_weight=Vec::new(), eps=1e-6))]
    pub fn new(q: Vec<u8>, k: Vec<u8>, v: Vec<u8>, centroids: Vec<f32>, stride: usize, n_head: usize, n_head_kv: usize, head_dim: usize, rmsnorm_weight: Vec<f32>, eps: f32) -> Self {
        GenomicAttention { q_database: q, k_database: k, v_database: v, centroids, stride, n_head, n_head_kv, head_dim, k_cache: Vec::new(), v_cache: Vec::new(), precision_mask: Vec::new() }
    }
    pub fn forward(&mut self, input_vector: Vec<f32>, _pos: usize) -> PyResult<Vec<f32>> {
        let n_embd = input_vector.len();
        let q_len = self.n_head * self.head_dim;
        let mut query = vec![0.0f32; q_len];
        let c_base = &self.centroids;
        let n_blocks_per_row = n_embd / 32;

        for h in 0..self.n_head {
            let row_offset = h * n_blocks_per_row * self.stride;
            if row_offset + n_blocks_per_row * self.stride > self.q_database.len() { continue; }

            let q_start = h * self.head_dim;
            for j in 0..n_blocks_per_row {
                let block_start = row_offset + j * self.stride;
                let q_dna = &self.q_database[block_start .. block_start + self.stride];
                let input_block = &input_vector[j * 32 .. (j + 1) * 32];
                let c_offset = (h * n_blocks_per_row + j) * 4;
                
                if c_offset + 4 <= c_base.len() {
                    let c = &c_base[c_offset..c_offset + 4];
                    let mut dims = 0;
                    for &byte in q_dna {
                        for s in 0..4 {
                            let shift = (3 - s) * 2;
                            let bits = (byte >> shift) & 0b11;
                            let val = match bits { 0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0 };
                            if q_start + dims < q_len {
                                query[q_start + dims] = unsafe { dot_product_neon(&input_vector, &vec![val; n_embd]) };
                            }
                            dims += 1;
                        }
                    }
                }
            }
        }
        Ok(query)
    }
    pub fn clear_cache(&mut self) -> PyResult<()> {
        self.k_cache.clear();
        self.v_cache.clear();
        Ok(())
    }
}

#[pyclass]
pub struct GenomicSwiGLU {
    pub w_gate: Vec<u8>, pub w_up: Vec<u8>, pub centroids: Vec<f32>,
    pub out_features: usize, pub in_features: usize, pub block_size: usize, pub stride: usize,
    pub precision_mask: Vec<u8>,
}

#[pymethods]
impl GenomicSwiGLU {
    #[new]
    pub fn new(w_gate: Vec<u8>, w_up: Vec<u8>, centroids: Vec<f32>, out_features: usize, in_features: usize, block_size: usize) -> Self {
        GenomicSwiGLU { w_gate, w_up, centroids, out_features, in_features, block_size, stride: block_size / 4, precision_mask: Vec::new() }
    }
    pub fn forward(&self, input_vector: Vec<f32>) -> PyResult<Vec<f32>> {
        let n_blocks_per_row = self.in_features / self.block_size;
        let silu = |x: f32| x / (1.0 + (-x).exp());
        let results: Vec<f32> = (0..self.out_features).into_par_iter().map(|i| {
            let mut gate_sum = 0.0f32;
            let mut up_sum = 0.0f32;
            let row_offset = i * n_blocks_per_row * self.stride;
            for j in 0..n_blocks_per_row {
                let block_start = row_offset + j * self.stride;
                let g_weights = &self.w_gate[block_start..block_start + self.stride];
                let u_weights = &self.w_up[block_start..block_start + self.stride];
                let input_block = &input_vector[j * self.block_size .. (j + 1) * self.block_size];
                let c_offset = (i * n_blocks_per_row + j) * 4;
                let c = &self.centroids[c_offset..c_offset + 4];
                let mut dims = 0;
                for k in 0..self.stride {
                    let g_byte = g_weights[k];
                    let u_byte = u_weights[k];
                    for s in 0..4 {
                        let shift = (3 - s) * 2;
                        let g_bits = (g_byte >> shift) & 0b11;
                        let u_bits = (u_byte >> shift) & 0b11;
                        let g_val = match g_bits { 0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0 };
                        let u_val = match u_bits { 0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0 };
                        let inp = input_block[dims];
                        gate_sum += inp * g_val; up_sum += inp * u_val; dims += 1;
                    }
                }
            }
            silu(gate_sum) * up_sum
        }).collect();
        Ok(results)
    }
}
