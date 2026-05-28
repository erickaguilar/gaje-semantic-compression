use crate::compute::kernels::*;
use half::f16;
use rayon::prelude::*;
use rand::Rng;
use std::sync::Arc;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Núcleo de Capa Lineal Genómica (Pure Rust)
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct GenomicLinear {
    pub database: Arc<Vec<u8>>,
    pub epi_strands: Arc<Vec<u8>>,
    pub tri_strands: Arc<Vec<u8>>,
    pub epi_cols: Arc<Vec<(usize, usize)>>,
    pub tri_cols: Arc<Vec<(usize, usize)>>,
    pub anchor_indices: Arc<Vec<u32>>,
    pub anchor_values: Arc<Vec<f16>>,
    pub anchor_row_ptrs: Arc<Vec<usize>>,
    pub centroids: Vec<f32>,
    pub epigenetic_centroids: Vec<f32>,
    pub triplet_centroids: Vec<f32>,
    pub out_features: usize,
    pub in_features: usize,
    pub block_size: usize,
    pub rmsnorm_weight: Vec<f32>,
    pub eps: f32,
    pub bias: Vec<f32>,
    pub stride: usize,
}

impl GenomicLinear {
    pub fn new(database: Vec<u8>, anchors_u8: Vec<u8>, centroids: Vec<f32>, out_features: usize, in_features: usize, block_size: usize, rmsnorm_weight: Vec<f32>, eps: f32, precision_mask: Vec<u8>, epigenetic_database: Vec<u8>, epigenetic_centroids: Vec<f32>, triplet_database: Vec<u8>, triplet_centroids: Vec<f32>, bias: Vec<f32>) -> Self {
        let stride = block_size / 4;
        let (anchor_indices, anchor_values, anchor_row_ptrs) = if anchors_u8.is_empty() { (Vec::new(), Vec::new(), vec![0; out_features + 1]) }
        else if anchors_u8.len() >= 4 && &anchors_u8[0..4] == b"GAJE" {
            let count = u32::from_le_bytes(anchors_u8[4..8].try_into().unwrap()) as usize;
            let mut indices = Vec::with_capacity(count); let mut values = Vec::with_capacity(count); let mut row_ptrs = vec![0; out_features + 1];
            let idx_s = 8; let val_s = idx_s + count * 4; let ptr_s = val_s + count * 2;
            for i in 0..count {
                indices.push(u32::from_le_bytes(anchors_u8[idx_s + i * 4..idx_s + i * 4 + 4].try_into().unwrap()));
                values.push(f16::from_le_bytes(anchors_u8[val_s + i * 2..val_s + i * 2 + 2].try_into().unwrap()));
            }
            for i in 0..=out_features { row_ptrs[i] = u64::from_le_bytes(anchors_u8[ptr_s + i * 8..ptr_s + i * 8 + 8].try_into().unwrap()) as usize; }
            (indices, values, row_ptrs)
        } else { (Vec::new(), Vec::new(), vec![0; out_features + 1]) };
        let n_blocks = in_features / block_size;
        let mut epi_cols = Vec::new(); let mut tri_cols = Vec::new();
        if !precision_mask.is_empty() {
            for j in 0..n_blocks { for k in 0..stride {
                let m = precision_mask[j * stride + k];
                if m >= 1 { epi_cols.push((j, k)); }
                if m >= 2 { tri_cols.push((j, k)); }
            }}
        }
        let mut epi_strands = Vec::new(); let mut tri_strands = Vec::new();
        if !epigenetic_database.is_empty() && !epi_cols.is_empty() {
            for i in 0..out_features { 
                let off = i * n_blocks * stride; 
                for &(j, k) in &epi_cols { 
                    let idx = off + j * stride + k;
                    epi_strands.push(*epigenetic_database.get(idx).unwrap_or(&0)); 
                } 
            }
        }
        if !triplet_database.is_empty() && !tri_cols.is_empty() {
            for i in 0..out_features { 
                let off = i * n_blocks * stride; 
                for &(j, k) in &tri_cols { 
                    let idx = off + j * stride + k;
                    tri_strands.push(*triplet_database.get(idx).unwrap_or(&0)); 
                } 
            }
        }
        GenomicLinear {
            database: Arc::new(database), epi_strands: Arc::new(epi_strands), tri_strands: Arc::new(tri_strands), epi_cols: Arc::new(epi_cols), tri_cols: Arc::new(tri_cols),
            anchor_indices: Arc::new(anchor_indices), anchor_values: Arc::new(anchor_values), anchor_row_ptrs: Arc::new(anchor_row_ptrs),
            centroids, epigenetic_centroids, triplet_centroids, out_features, in_features, block_size, rmsnorm_weight, eps, bias, stride,
        }
    }

    pub fn forward_core(&self, mut input: Vec<f32>, modulation_factors: Option<[f32; 4]>) -> Result<Vec<f32>, String> {
        if !self.rmsnorm_weight.is_empty() { input = unsafe { rms_norm(&input, &self.rmsnorm_weight, self.eps) }; }
        let n_blocks = self.in_features / self.block_size;
        let has_bias = !self.bias.is_empty(); let has_epi = !self.epi_strands.is_empty(); let has_tri = !self.tri_strands.is_empty();
        
        let m_factors = modulation_factors.unwrap_or([1.0f32; 4]);

        let results: Vec<f32> = (0..self.out_features).into_par_iter().map(|i| {
            let row_off = i * n_blocks * self.stride;
            let weights = &self.database[row_off..row_off + n_blocks * self.stride];
            let row_centroids = &self.centroids[i * n_blocks * 4..(i + 1) * n_blocks * 4];
            
            // Aplicamos modulación granular directamente en el producto punto si se provee
            let mut sum = unsafe { genomic_dot_product(weights, &input, row_centroids, self.stride, n_blocks, &m_factors) };
            let a_s = self.anchor_row_ptrs[i]; let a_e = self.anchor_row_ptrs[i + 1];
            for k in a_s..a_e { sum += input[self.anchor_indices[k] as usize] * self.anchor_values[k].to_f32(); }
            if has_epi {
                let mut e_sum = 0.0f32; let r_epi_off = i * self.epi_cols.len();
                for (idx, &(j, k)) in self.epi_cols.iter().enumerate() {
                    let ce = &self.epigenetic_centroids[(i * n_blocks + j) * 4..(i * n_blocks + j) * 4 + 4];
                    let byte = self.epi_strands[r_epi_off + idx];
                    for s in 0..4 {
                        let eb = (byte >> ((3 - s) * 2)) & 0b11;
                        let val = match eb { 0b00 => ce[0], 0b01 => ce[1], 0b11 => ce[2], 0b10 => ce[3], _ => 0.0 };
                        e_sum += input[j * self.block_size + k * 4 + s] * val;
                    }
                }
                sum += e_sum;
            }
            if has_tri {
                let mut t_sum = 0.0f32; let r_tri_off = i * self.tri_cols.len();
                for (idx, &(j, k)) in self.tri_cols.iter().enumerate() {
                    let ct = &self.triplet_centroids[(i * n_blocks + j) * 4..(i * n_blocks + j) * 4 + 4];
                    let byte = self.tri_strands[r_tri_off + idx];
                    for s in 0..4 {
                        let tb = (byte >> ((3 - s) * 2)) & 0b11;
                        let val = match tb { 0b00 => ct[0], 0b01 => ct[1], 0b11 => ct[2], 0b10 => ct[3], _ => 0.0 };
                        t_sum += input[j * self.block_size + k * 4 + s] * val;
                    }
                }
                sum += t_sum;
            }
            if has_bias { sum += self.bias[i]; }
            sum
        }).collect();
        Ok(results)
    }

    pub fn get_row_core(&self, idx: usize) -> Result<Vec<f32>, String> {
        if idx >= self.out_features { return Err(format!("Index {} out of bounds", idx)); }
        let n_blocks = self.in_features / self.block_size;
        let mut res = vec![0.0f32; self.in_features];
        let row_start = idx * n_blocks * self.stride;
        let row_dna = &self.database[row_start..row_start + n_blocks * self.stride];
        for b in 0..n_blocks {
            let block_dna = &row_dna[b * self.stride..(b + 1) * self.stride];
            let c_off = (idx * n_blocks + b) * 4;
            let centroids = &self.centroids[c_off..c_off + 4];
            let decoded = crate::compute::math::dequantize_embedding_core(block_dna, self.block_size, Some(centroids))?;
            for i in 0..self.block_size { res[b * self.block_size + i] = decoded[i]; }
        }
        Ok(res)
    }

    pub fn backward_core(&self, d_output: Vec<f32>) -> Result<Vec<f32>, String> {
        let n_blocks = self.in_features / self.block_size;
        let mut d_input = vec![0.0f32; self.in_features];
        d_input.par_chunks_mut(self.block_size).enumerate().for_each(|(j, d_in_block)| {
            for i in 0..self.out_features {
                let d_out_val = d_output[i]; if d_out_val == 0.0 { continue; }
                let row_off = i * n_blocks * self.stride;
                let weights = &self.database[row_off + j * self.stride .. row_off + (j + 1) * self.stride];
                let c_off = (i * n_blocks + j) * 4;
                let row_centroids = &self.centroids[c_off..c_off + 4];
                for k in 0..self.stride {
                    let byte = weights[k];
                    for s in 0..4 {
                        let bits = (byte >> ((3 - s) * 2)) & 0b11;
                        let val = match bits { 0b00 => row_centroids[0], 0b01 => row_centroids[1], 0b11 => row_centroids[2], 0b10 => row_centroids[3], _ => 0.0 };
                        d_in_block[k * 4 + s] += d_out_val * val;
                    }
                }
            }
        });
        Ok(d_input)
    }

    pub fn refine_with_grads_core(&mut self, mut input: Vec<f32>, grads: Vec<f32>, lr: f32) -> Result<(), String> {
        if !self.rmsnorm_weight.is_empty() { input = unsafe { rms_norm(&input, &self.rmsnorm_weight, self.eps) }; }
        let n_blocks = self.in_features / self.block_size;
        self.centroids.par_chunks_mut(n_blocks * 4).enumerate().for_each(|(i, row_centroids)| {
            if i >= grads.len() { return; }
            let grad_scale = grads[i] * lr; if grad_scale.abs() > 1e-8 {
                let row_off = i * n_blocks * self.stride;
                for j in 0..n_blocks {
                    let weights = &self.database[row_off + j * self.stride..row_off + (j + 1) * self.stride];
                    let input_block = &input[j * self.block_size..(j + 1) * self.block_size];
                    for k in 0..self.stride {
                        let byte = weights[k];
                        for s in 0..4 {
                            let bits = (byte >> ((3 - s) * 2)) & 0b11;
                            let c_idx = match bits { 0b00 => 0, 0b01 => 1, 0b11 => 2, 0b10 => 3, _ => 4 };
                            if c_idx < 4 { row_centroids[j * 4 + c_idx] -= grad_scale * input_block[k * 4 + s]; }
                        }
                    }
                }
            }
        });
        Ok(())
    }

    pub fn apply_mutation_core(&mut self, delta_centroids: Vec<f32>, undo: bool) -> Result<(), String> {
        for (i, d) in self.centroids.iter_mut().zip(delta_centroids) { if undo { *i += d; } else { *i -= d; } }
        Ok(())
    }

    pub fn apply_weighted_mutation_core(&mut self, delta: Vec<f32>, weight: f32) -> Result<(), String> {
        for (c, d) in self.centroids.iter_mut().zip(delta) { *c += d * weight; }
        Ok(())
    }

    pub fn mutate_random_core(&mut self, scale: f32) -> Result<Vec<f32>, String> {
        let mut rng = rand::thread_rng(); let mut delta = Vec::with_capacity(self.centroids.len());
        for c in &mut self.centroids { let m = rng.gen_range(-scale..scale); *c += m; delta.push(m); }
        Ok(delta)
    }

    pub fn anchors_sparse_buffer(&self) -> Vec<u8> {
        let mut out = Vec::new(); out.extend_from_slice(b"GAJE");
        let count = self.anchor_indices.len(); out.extend_from_slice(&(count as u32).to_le_bytes());
        for &idx in self.anchor_indices.iter() { out.extend_from_slice(&idx.to_le_bytes()); }
        for &val in self.anchor_values.iter() { out.extend_from_slice(&val.to_le_bytes()); }
        for &ptr in self.anchor_row_ptrs.iter() { out.extend_from_slice(&(ptr as u64).to_le_bytes()); }
        out
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl GenomicLinear {
    #[new]
    #[pyo3(signature = (database, anchors_u8, centroids, out_features, in_features, block_size, rmsnorm_weight=Vec::new(), eps=1e-6, precision_mask=Vec::new(), epigenetic_database=Vec::new(), epigenetic_centroids=Vec::new(), triplet_database=Vec::new(), triplet_centroids=Vec::new(), bias=Vec::new()))]
    pub fn py_new(database: Vec<u8>, anchors_u8: Vec<u8>, centroids: Vec<f32>, out_features: usize, in_features: usize, block_size: usize, rmsnorm_weight: Vec<f32>, eps: f32, precision_mask: Vec<u8>, epigenetic_database: Vec<u8>, epigenetic_centroids: Vec<f32>, triplet_database: Vec<u8>, triplet_centroids: Vec<f32>, bias: Vec<f32>) -> Self {
        GenomicLinear::new(database, anchors_u8, centroids, out_features, in_features, block_size, rmsnorm_weight, eps, precision_mask, epigenetic_database, epigenetic_centroids, triplet_database, triplet_centroids, bias)
    }
    pub fn forward(&self, input: Vec<f32>) -> PyResult<Vec<f32>> { self.forward_core(input, None).map_err(pyo3::exceptions::PyValueError::new_err) }
    pub fn get_row(&self, idx: usize) -> PyResult<Vec<f32>> { self.get_row_core(idx).map_err(pyo3::exceptions::PyValueError::new_err) }
    pub fn backward(&self, d_output: Vec<f32>) -> PyResult<Vec<f32>> { self.backward_core(d_output).map_err(pyo3::exceptions::PyValueError::new_err) }
    pub fn refine_with_grads(&mut self, input: Vec<f32>, grads: Vec<f32>, lr: f32) -> PyResult<()> { self.refine_with_grads_core(input, grads, lr).map_err(pyo3::exceptions::PyValueError::new_err) }
}
