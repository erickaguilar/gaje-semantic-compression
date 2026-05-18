use crate::compute::kernels::*;
use half::f16;
use pyo3::prelude::*;
use rayon::prelude::*;

use std::sync::Arc;

#[pyclass]
#[derive(Clone)]
pub struct GenomicLinear {
    pub database: Arc<Vec<u8>>,
    pub epi_strands: Arc<Vec<u8>>,
    pub tri_strands: Arc<Vec<u8>>,
    pub epi_cols: Arc<Vec<(usize, usize)>>, // (j, k) block and stride offsets
    pub tri_cols: Arc<Vec<(usize, usize)>>,
    pub anchors: Arc<Vec<f16>>,
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
    pub bias: Vec<f32>,
    #[pyo3(get)]
    pub stride: usize,
    
    // Kept to not break Python compatibility on legacy scripts
    pub epigenetic_database: Arc<Vec<u8>>,
    pub triplet_database: Arc<Vec<u8>>,
    pub precision_mask: Arc<Vec<u8>>,
}

#[pymethods]
impl GenomicLinear {
    #[new]
    #[pyo3(signature = (database, anchors_u8, centroids, out_features, in_features, block_size, rmsnorm_weight=Vec::new(), eps=1e-6, precision_mask=Vec::new(), epigenetic_database=Vec::new(), epigenetic_centroids=Vec::new(), triplet_database=Vec::new(), triplet_centroids=Vec::new(), bias=Vec::new()))]
    pub fn new(
        database: Vec<u8>,
        anchors_u8: Vec<u8>,
        centroids: Vec<f32>,
        out_features: usize,
        in_features: usize,
        block_size: usize,
        rmsnorm_weight: Vec<f32>,
        eps: f32,
        precision_mask: Vec<u8>,
        epigenetic_database: Vec<u8>,
        epigenetic_centroids: Vec<f32>,
        triplet_database: Vec<u8>,
        triplet_centroids: Vec<f32>,
        bias: Vec<f32>,
    ) -> Self {
        let stride = block_size / 4;
        
        let anchors = if anchors_u8.is_empty() {
            Vec::new()
        } else {
            unsafe {
                std::slice::from_raw_parts(
                    anchors_u8.as_ptr() as *const f16,
                    anchors_u8.len() / 2
                ).to_vec()
            }
        };

        let n_blocks = in_features / block_size;

        let mut epi_cols = Vec::new();
        let mut tri_cols = Vec::new();

        if !precision_mask.is_empty() {
            // Mask is identical for all output rows, just read the first row
            for j in 0..n_blocks {
                for k in 0..stride {
                    let mode = precision_mask[j * stride + k];
                    if mode >= 1 { epi_cols.push((j, k)); }
                    if mode >= 2 { tri_cols.push((j, k)); }
                }
            }
        }

        let has_epi = !epigenetic_database.is_empty() && !epi_cols.is_empty();
        let has_tri = !triplet_database.is_empty() && !tri_cols.is_empty();

        let mut epi_strands = Vec::new();
        let mut tri_strands = Vec::new();

        if has_epi {
            epi_strands.reserve(out_features * epi_cols.len());
            for i in 0..out_features {
                let row_offset = i * n_blocks * stride;
                for &(j, k) in &epi_cols {
                    epi_strands.push(epigenetic_database[row_offset + j * stride + k]);
                }
            }
        }

        if has_tri {
            tri_strands.reserve(out_features * tri_cols.len());
            for i in 0..out_features {
                let row_offset = i * n_blocks * stride;
                for &(j, k) in &tri_cols {
                    tri_strands.push(triplet_database[row_offset + j * stride + k]);
                }
            }
        }

        GenomicLinear {
            database: Arc::new(database),
            epi_strands: Arc::new(epi_strands),
            tri_strands: Arc::new(tri_strands),
            epi_cols: Arc::new(epi_cols),
            tri_cols: Arc::new(tri_cols),
            anchors: Arc::new(anchors),
            centroids,
            epigenetic_centroids,
            triplet_centroids,
            out_features,
            in_features,
            block_size,
            rmsnorm_weight,
            eps,
            bias,
            stride,
            epigenetic_database: Arc::new(epigenetic_database),
            triplet_database: Arc::new(triplet_database),
            precision_mask: Arc::new(precision_mask),
        }
    }

    #[getter]
    pub fn database<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, pyo3::types::PyBytes>> {
        Ok(pyo3::types::PyBytes::new(py, &self.database))
    }

    #[getter]
    pub fn epigenetic_database<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, pyo3::types::PyBytes>> {
        Ok(pyo3::types::PyBytes::new(py, &self.epigenetic_database))
    }

    #[getter]
    pub fn triplet_database<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, pyo3::types::PyBytes>> {
        Ok(pyo3::types::PyBytes::new(py, &self.triplet_database))
    }

    #[getter]
    pub fn precision_mask<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, pyo3::types::PyBytes>> {
        Ok(pyo3::types::PyBytes::new(py, &self.precision_mask))
    }

    #[getter]
    pub fn anchors(&self) -> Vec<f32> {
        self.anchors.iter().map(|&x| x.to_f32()).collect()
    }

    pub fn forward(&self, mut input: Vec<f32>) -> PyResult<Vec<f32>> {
        if !self.rmsnorm_weight.is_empty() {
            input = unsafe { rms_norm(&input, &self.rmsnorm_weight, self.eps) };
        }

        let n_blocks = self.in_features / self.block_size;
        let has_anchors = !self.anchors.is_empty();
        let has_bias = !self.bias.is_empty();
        
        let has_epi = !self.epi_strands.is_empty();
        let has_tri = !self.tri_strands.is_empty();

        let results: Vec<f32> = (0..self.out_features)
            .into_par_iter()
            .map(|i| {
                let row_offset = i * n_blocks * self.stride;

                // Phase 1: Pure SIMD MatMul for Base 2-bit Strands (Always runs, no branching)
                let row_weights = &self.database[row_offset..row_offset + n_blocks * self.stride];
                let row_centroids = &self.centroids[i * n_blocks * 4..(i + 1) * n_blocks * 4];
                let mut row_sum = unsafe {
                    genomic_dot_product(
                        row_weights,
                        &input,
                        row_centroids,
                        self.stride,
                        n_blocks,
                    )
                };

                // Phase 2: Epigenetic Correction (4-bit) Scatter-Add
                if has_epi {
                    let mut epi_sum = 0.0f32;
                    let row_epi_offset = i * self.epi_cols.len();
                    for (idx, &(j, k)) in self.epi_cols.iter().enumerate() {
                        let c_offset = (i * n_blocks + j) * 4;
                        let ce = &self.epigenetic_centroids[c_offset..c_offset + 4];
                        let input_block = &input[j * self.block_size..(j + 1) * self.block_size];
                        
                        let byte = self.epi_strands[row_epi_offset + idx];
                        for s in 0..4 {
                            let shift = (3 - s) * 2;
                            let eb = (byte >> shift) & 0b11;
                            let val = match eb {
                                0b00 => ce[0],
                                0b01 => ce[1],
                                0b11 => ce[2],
                                0b10 => ce[3],
                                _ => 0.0,
                            };
                            epi_sum += input_block[k * 4 + s] * val;
                        }
                    }
                    row_sum += epi_sum;
                }

                // Phase 3: Triplet Refinement (6-bit) Scatter-Add
                if has_tri {
                    let mut tri_sum = 0.0f32;
                    let row_tri_offset = i * self.tri_cols.len();
                    for (idx, &(j, k)) in self.tri_cols.iter().enumerate() {
                        let c_offset = (i * n_blocks + j) * 4;
                        let ct = &self.triplet_centroids[c_offset..c_offset + 4];
                        let input_block = &input[j * self.block_size..(j + 1) * self.block_size];
                        
                        let byte = self.tri_strands[row_tri_offset + idx];
                        for s in 0..4 {
                            let shift = (3 - s) * 2;
                            let tb = (byte >> shift) & 0b11;
                            let val = match tb {
                                0b00 => ct[0],
                                0b01 => ct[1],
                                0b11 => ct[2],
                                0b10 => ct[3],
                                _ => 0.0,
                            };
                            tri_sum += input_block[k * 4 + s] * val;
                        }
                    }
                    row_sum += tri_sum;
                }

                if has_anchors {
                    let anchor_row = &self.anchors[i * self.in_features..(i + 1) * self.in_features];
                    let mut a_sum = 0.0f32;
                    for j in 0..self.in_features {
                        a_sum += anchor_row[j].to_f32() * input[j];
                    }
                    row_sum += a_sum;
                }

                if has_bias {
                    row_sum += self.bias[i];
                }
                
                row_sum
            })
            .collect();

        Ok(results)
    }

    pub fn get_row(&self, idx: usize) -> PyResult<Vec<f32>> {
        if idx >= self.out_features {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Index {} out of bounds for vocab size (out_features) {}",
                idx, self.out_features
            )));
        }
        let n_blocks = self.in_features / self.block_size;
        let mut res = vec![0.0f32; self.in_features];

        let row_start = idx * n_blocks * self.stride;
        let row_dna = &self.database[row_start..row_start + n_blocks * self.stride];

        for b in 0..n_blocks {
            let block_dna = &row_dna[b * self.stride..(b + 1) * self.stride];
            let c_offset = (idx * n_blocks + b) * 4;
            let centroids_f32 = &self.centroids[c_offset..c_offset + 4];

            let decoded = crate::compute::math::dequantize_embedding(
                block_dna.to_vec(),
                self.block_size,
                Some(centroids_f32.to_vec()),
            )?;
            for i in 0..self.block_size {
                res[b * self.block_size + i] = decoded[i];
            }
        }
        
        let has_epi = !self.epi_strands.is_empty();
        let has_tri = !self.tri_strands.is_empty();
        
        if has_epi {
            let row_epi_offset = idx * self.epi_cols.len();
            for (col_idx, &(j, k)) in self.epi_cols.iter().enumerate() {
                let c_offset = (idx * n_blocks + j) * 4;
                let ce = &self.epigenetic_centroids[c_offset..c_offset + 4];
                let byte = self.epi_strands[row_epi_offset + col_idx];
                for s in 0..4 {
                    let shift = (3 - s) * 2;
                    let eb = (byte >> shift) & 0b11;
                    let val = match eb {
                        0b00 => ce[0],
                        0b01 => ce[1],
                        0b11 => ce[2],
                        0b10 => ce[3],
                        _ => 0.0,
                    };
                    res[j * self.block_size + k * 4 + s] += val;
                }
            }
        }
        
        if has_tri {
            let row_tri_offset = idx * self.tri_cols.len();
            for (col_idx, &(j, k)) in self.tri_cols.iter().enumerate() {
                let c_offset = (idx * n_blocks + j) * 4;
                let ct = &self.triplet_centroids[c_offset..c_offset + 4];
                let byte = self.tri_strands[row_tri_offset + col_idx];
                for s in 0..4 {
                    let shift = (3 - s) * 2;
                    let tb = (byte >> shift) & 0b11;
                    let val = match tb {
                        0b00 => ct[0],
                        0b01 => ct[1],
                        0b11 => ct[2],
                        0b10 => ct[3],
                        _ => 0.0,
                    };
                    res[j * self.block_size + k * 4 + s] += val;
                }
            }
        }

        if !self.anchors.is_empty() {
            let anchor_row = &self.anchors[idx * self.in_features..(idx + 1) * self.in_features];
            for i in 0..self.in_features {
                res[i] += anchor_row[i].to_f32();
            }
        }

        Ok(res)
    }

    pub fn refine_centroids(
        &mut self,
        mut input: Vec<f32>,
        target: Vec<f32>,
        lr: f32,
    ) -> PyResult<()> {
        if !self.rmsnorm_weight.is_empty() {
            input = unsafe { rms_norm(&input, &self.rmsnorm_weight, self.eps) };
        }
        let n_blocks = self.in_features / self.block_size;
        let has_anchors = !self.anchors.is_empty();

        for i in 0..self.out_features {
            let mut row_sum = 0.0f32;
            let row_offset = i * n_blocks * self.stride;
            
            // Phase 1
            let row_weights = &self.database[row_offset..row_offset + n_blocks * self.stride];
            let row_centroids = &self.centroids[i * n_blocks * 4..(i + 1) * n_blocks * 4];
            row_sum += unsafe {
                genomic_dot_product(row_weights, &input, row_centroids, self.stride, n_blocks)
            };
            
            // Phase 2
            if !self.epi_strands.is_empty() {
                let mut epi_sum = 0.0f32;
                let row_epi_offset = i * self.epi_cols.len();
                for (idx, &(j, k)) in self.epi_cols.iter().enumerate() {
                    let c_offset = (i * n_blocks + j) * 4;
                    let ce = &self.epigenetic_centroids[c_offset..c_offset + 4];
                    let input_block = &input[j * self.block_size..(j + 1) * self.block_size];
                    let byte = self.epi_strands[row_epi_offset + idx];
                    for s in 0..4 {
                        let shift = (3 - s) * 2;
                        let val = match (byte >> shift) & 0b11 {
                            0b00 => ce[0], 0b01 => ce[1], 0b11 => ce[2], 0b10 => ce[3], _ => 0.0,
                        };
                        epi_sum += input_block[k * 4 + s] * val;
                    }
                }
                row_sum += epi_sum;
            }
            
            // Phase 3
            if !self.tri_strands.is_empty() {
                let mut tri_sum = 0.0f32;
                let row_tri_offset = i * self.tri_cols.len();
                for (idx, &(j, k)) in self.tri_cols.iter().enumerate() {
                    let c_offset = (i * n_blocks + j) * 4;
                    let ct = &self.triplet_centroids[c_offset..c_offset + 4];
                    let input_block = &input[j * self.block_size..(j + 1) * self.block_size];
                    let byte = self.tri_strands[row_tri_offset + idx];
                    for s in 0..4 {
                        let shift = (3 - s) * 2;
                        let val = match (byte >> shift) & 0b11 {
                            0b00 => ct[0], 0b01 => ct[1], 0b11 => ct[2], 0b10 => ct[3], _ => 0.0,
                        };
                        tri_sum += input_block[k * 4 + s] * val;
                    }
                }
                row_sum += tri_sum;
            }

            if has_anchors {
                let anchor_row = &self.anchors[i * self.in_features..(i + 1) * self.in_features];
                for (j, &a) in anchor_row.iter().enumerate() {
                    row_sum += a.to_f32() * input[j];
                }
            }

            let grad_scale = (row_sum - target[i]) * lr;
            self.apply_grad_to_row(i, &input, grad_scale, n_blocks);
        }

        Ok(())
    }

    pub fn refine_with_grads(
        &mut self,
        mut input: Vec<f32>,
        grads: Vec<f32>,
        lr: f32,
    ) -> PyResult<()> {
        if !self.rmsnorm_weight.is_empty() {
            input = unsafe { rms_norm(&input, &self.rmsnorm_weight, self.eps) };
        }
        let n_blocks = self.in_features / self.block_size;

        for i in 0..self.out_features {
            let grad_scale = grads[i] * lr;
            self.apply_grad_to_row(i, &input, grad_scale, n_blocks);
        }

        Ok(())
    }
}

impl GenomicLinear {
    pub fn apply_grad_to_row(
        &mut self,
        i: usize,
        input: &[f32],
        grad_scale: f32,
        n_blocks: usize,
    ) {
        self.apply_grad_to_row_with_delta(i, input, grad_scale, n_blocks, &mut None)
    }

    fn apply_grad_to_row_with_delta(
        &mut self,
        i: usize,
        input: &[f32],
        grad_scale: f32,
        n_blocks: usize,
        delta_buffer: &mut Option<&mut [f32]>,
    ) {
        let row_offset = i * n_blocks * self.stride;
        for j in 0..n_blocks {
            let block_start = row_offset + j * self.stride;
            let c_offset = (i * n_blocks + j) * 4;
            let input_block = &input[j * self.block_size..(j + 1) * self.block_size];
            let weights = &self.database[block_start..block_start + self.stride];
            let mut dims = 0;
            for k in 0..self.stride {
                let byte = weights[k];
                let mut mode = 0;
                if !self.precision_mask.is_empty() {
                    mode = self.precision_mask[j * self.stride + k];
                }

                for s in 0..4 {
                    let shift = (3 - s) * 2;
                    let bits = (byte >> shift) & 0b11;
                    let c_idx = match bits {
                        0b00 => 0,
                        0b01 => 1,
                        0b11 => 2,
                        0b10 => 3,
                        _ => 4,
                    };
                    if c_idx < 4 {
                        let delta = grad_scale * input_block[dims];
                        if let Some(ref mut buf) = delta_buffer {
                            buf[c_offset + c_idx] += delta;
                        }

                        let current = self.centroids[c_offset + c_idx];
                        self.centroids[c_offset + c_idx] = current - delta;

                        if mode >= 1 && !self.epi_strands.is_empty() {
                            let current_e = self.epigenetic_centroids[c_offset + c_idx];
                            self.epigenetic_centroids[c_offset + c_idx] =
                                current_e - delta * 0.5;
                        }
                        if mode >= 2 && !self.tri_strands.is_empty() {
                            let current_t = self.triplet_centroids[c_offset + c_idx];
                            self.triplet_centroids[c_offset + c_idx] =
                                current_t - delta * 0.25;
                        }
                    }
                    dims += 1;
                }
            }
        }
    }

    pub fn apply_mutation(&mut self, delta_centroids: Vec<f32>, undo: bool) -> PyResult<()> {
        if delta_centroids.len() != self.centroids.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Mutation delta size mismatch"));
        }
        for (i, d) in self.centroids.iter_mut().zip(delta_centroids) {
            if undo {
                *i += d; 
            } else {
                *i -= d;
            }
        }
        Ok(())
    }

    pub fn mutate_random(&mut self, scale: f32) -> PyResult<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut delta = Vec::with_capacity(self.centroids.len());
        for c in &mut self.centroids {
            let mutation = rng.gen_range(-scale..scale);
            *c += mutation;
            delta.push(mutation);
        }
        Ok(delta)
    }

    pub fn undo_delta(&mut self, delta: Vec<f32>) -> PyResult<()> {
        if delta.len() != self.centroids.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Delta size mismatch"));
        }
        for (c, d) in self.centroids.iter_mut().zip(delta) {
            *c -= d;
        }
        Ok(())
    }

    pub fn apply_weighted_mutation(&mut self, delta: Vec<f32>, weight: f32) -> PyResult<()> {
        if delta.len() != self.centroids.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Delta size mismatch"));
        }
        for (c, d) in self.centroids.iter_mut().zip(delta) {
            *c += d * weight;
        }
        Ok(())
    }
}