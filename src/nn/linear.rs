use crate::compute::kernels::*;
use half::f16;
use pyo3::prelude::*;
use rayon::prelude::*;
use rand::Rng;

use std::sync::Arc;

#[pyclass]
#[derive(Clone)]
pub struct GenomicLinear {
    pub database: Arc<Vec<u8>>,
    pub epi_strands: Arc<Vec<u8>>,
    pub tri_strands: Arc<Vec<u8>>,
    pub epi_cols: Arc<Vec<(usize, usize)>>, // (j, k) block and stride offsets
    pub tri_cols: Arc<Vec<(usize, usize)>>,
    pub anchor_indices: Arc<Vec<u32>>,      // Global indices within the weight matrix
    pub anchor_values: Arc<Vec<f16>>,       // f16 values of the anchors
    pub anchor_row_ptrs: Arc<Vec<usize>>,   // Pointers to start of each row in anchor_indices/values
    pub anchors: Arc<Vec<f16>>,             // Legacy dense anchors (now usually empty)
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

        // Detect if anchors_u8 is Sparse or Dense
        let (anchor_indices, anchor_values, anchor_row_ptrs, legacy_anchors) = if anchors_u8.is_empty() {
            (Vec::new(), Vec::new(), vec![0; out_features + 1], Vec::new())
        } else if (anchors_u8.len() > 8 && &anchors_u8[0..4] == b"GAJE") || (anchors_u8.len() > 6 && &anchors_u8[0..2] == b"GA") { 
            // Magic bytes for Sparse (GAJE = new, GA = legacy/stable)
            let is_gaje = &anchors_u8[0..4] == b"GAJE";
            let count_start = if is_gaje { 4 } else { 2 };
            let count = u32::from_le_bytes(anchors_u8[count_start..count_start+4].try_into().unwrap()) as usize;
            
            let mut indices = Vec::with_capacity(count);
            let mut values = Vec::with_capacity(count);
            let mut row_ptrs = vec![0; out_features + 1];

            let index_start = count_start + 4;
            let value_start = index_start + count * 4;
            let ptr_start = value_start + count * 2;

            for i in 0..count {
                let idx = u32::from_le_bytes(anchors_u8[index_start + i * 4..index_start + i * 4 + 4].try_into().unwrap());
                indices.push(idx);
                let val = f16::from_le_bytes(anchors_u8[value_start + i * 2..value_start + i * 2 + 2].try_into().unwrap());
                values.push(val);
            }

            for i in 0..=out_features {
                let p = u64::from_le_bytes(anchors_u8[ptr_start + i * 8..ptr_start + i * 8 + 8].try_into().unwrap()) as usize;
                row_ptrs[i] = p;
            }

            (indices, values, row_ptrs, Vec::new())
        } else {
            // Legacy Dense or Automatic Sparsification
            let dense_anchors = unsafe {
                std::slice::from_raw_parts(anchors_u8.as_ptr() as *const f16, anchors_u8.len() / 2)
            };
            
            let mut indices = Vec::new();
            let mut values = Vec::with_capacity(dense_anchors.len() / 100); // Guessing 1% sparsity
            let mut row_ptrs = vec![0; out_features + 1];

            for i in 0..out_features {
                row_ptrs[i] = indices.len();
                for j in 0..in_features {
                    let val = dense_anchors[i * in_features + j];
                    if val != f16::ZERO {
                        indices.push(j as u32);
                        values.push(val);
                    }
                }
            }
            row_ptrs[out_features] = indices.len();

            (indices, values, row_ptrs, Vec::new())
        };

        let n_blocks = in_features / block_size;

        let mut epi_cols = Vec::new();
        let mut tri_cols = Vec::new();

        if !precision_mask.is_empty() {
            // Mask is identical for all output rows, just read the first row
            for j in 0..n_blocks {
                for k in 0..stride {
                    let mode = precision_mask[j * stride + k];
                    if mode >= 1 {
                        epi_cols.push((j, k));
                    }
                    if mode >= 2 {
                        tri_cols.push((j, k));
                    }
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
            anchor_indices: Arc::new(anchor_indices),
            anchor_values: Arc::new(anchor_values),
            anchor_row_ptrs: Arc::new(anchor_row_ptrs),
            anchors: Arc::new(legacy_anchors),
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
    pub fn database<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        Ok(pyo3::types::PyBytes::new(py, &self.database))
    }

    #[getter]
    pub fn epi_strands<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        Ok(pyo3::types::PyBytes::new(py, &self.epi_strands))
    }

    #[getter]
    pub fn tri_strands<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        Ok(pyo3::types::PyBytes::new(py, &self.tri_strands))
    }

    #[getter]
    pub fn epi_cols(&self) -> Vec<(usize, usize)> {
        self.epi_cols.to_vec()
    }

    #[getter]
    pub fn tri_cols(&self) -> Vec<(usize, usize)> {
        self.tri_cols.to_vec()
    }

    pub fn anchors_sparse_buffer(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GAJE"); // Magic bytes
        let count = self.anchor_indices.len();
        out.extend_from_slice(&(count as u32).to_le_bytes());
        for &idx in self.anchor_indices.iter() {
            out.extend_from_slice(&idx.to_le_bytes());
        }
        for &val in self.anchor_values.iter() {
            out.extend_from_slice(&val.to_le_bytes());
        }
        for &ptr in self.anchor_row_ptrs.iter() {
            out.extend_from_slice(&(ptr as u64).to_le_bytes());
        }
        out
    }

    #[getter]
    pub fn anchors_raw<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let out = self.anchors_sparse_buffer();
        Ok(pyo3::types::PyBytes::new(py, &out))
    }

    #[getter]
    pub fn epigenetic_database<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        Ok(pyo3::types::PyBytes::new(py, &self.epigenetic_database))
    }

    #[getter]
    pub fn triplet_database<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        Ok(pyo3::types::PyBytes::new(py, &self.triplet_database))
    }

    #[getter]
    pub fn precision_mask<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        Ok(pyo3::types::PyBytes::new(py, &self.precision_mask))
    }

    #[getter]
    pub fn anchors(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.out_features * self.in_features];
        for i in 0..self.out_features {
            let start = self.anchor_row_ptrs[i];
            let end = self.anchor_row_ptrs[i+1];
            for k in start..end {
                let col = self.anchor_indices[k] as usize;
                let val = self.anchor_values[k].to_f32();
                out[i * self.in_features + col] = val;
            }
        }
        out
    }

    pub fn forward(&self, mut input: Vec<f32>) -> PyResult<Vec<f32>> {
        if !self.rmsnorm_weight.is_empty() {
            input = unsafe { rms_norm(&input, &self.rmsnorm_weight, self.eps) };
        }

        let n_blocks = self.in_features / self.block_size;
        let has_bias = !self.bias.is_empty();
        let has_epi = !self.epi_strands.is_empty();
        let has_tri = !self.tri_strands.is_empty();

        let results: Vec<f32> = (0..self.out_features)
            .into_par_iter()
            .map(|i| {
                let row_offset = i * n_blocks * self.stride;
                let row_weights = &self.database[row_offset..row_offset + n_blocks * self.stride];
                let row_centroids = &self.centroids[i * n_blocks * 4..(i + 1) * n_blocks * 4];

                let mut row_sum = unsafe {
                    genomic_dot_product(
                        row_weights,
                        &input,
                        row_centroids,
                        self.stride,
                        n_blocks,
                        &[],
                    )
                };

                let a_start = self.anchor_row_ptrs[i];
                let a_end = self.anchor_row_ptrs[i + 1];
                for k in a_start..a_end {
                    let col = self.anchor_indices[k] as usize;
                    let val = self.anchor_values[k].to_f32();
                    row_sum += input[col] * val;
                }

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

                if has_bias {
                    row_sum += self.bias[i];
                }
                row_sum
            })
            .collect();

        Ok(results)
    }

    pub fn spiking_forward(
        &self,
        input: Vec<f32>,
        steps: usize,
        threshold: f32,
        decay: f32,
    ) -> PyResult<Vec<f32>> {
        let n_out = self.out_features;
        let mut potentials = vec![0.0f32; n_out];
        let mut spike_counts = vec![0.0f32; n_out];
        let mut first_spike_time = vec![0.0f32; n_out];

        // Pre-calculate the excitation (synaptic current)
        let excitation = self.forward(input)?;

        // Simulate LIF dynamics
        for t in 0..steps {
            for i in 0..n_out {
                potentials[i] += excitation[i];
                if potentials[i] >= threshold {
                    if first_spike_time[i] == 0.0 {
                        // Temporal priority: earlier spikes get higher temporal score
                        first_spike_time[i] = (steps - t) as f32;
                    }
                    potentials[i] = 0.0; // Reset
                    spike_counts[i] += 1.0;
                } else if potentials[i] > 0.0 {
                    potentials[i] *= decay;
                }
            }
        }

        // Calculate Resonance Score: Frequency + Temporal Precision
        let mut resonance = vec![0.0f32; n_out];
        for i in 0..n_out {
            resonance[i] = spike_counts[i] * (steps as f32) + first_spike_time[i];
        }

        Ok(resonance)
    }

    pub fn get_row(&self, idx: usize) -> PyResult<Vec<f32>> {
        if idx >= self.out_features {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Index {} out of bounds", idx
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
                block_dna.to_vec(), self.block_size, Some(centroids_f32.to_vec()),
            )?;
            for i in 0..self.block_size {
                res[b * self.block_size + i] = decoded[i];
            }
        }

        if !self.epi_strands.is_empty() {
            let row_epi_offset = idx * self.epi_cols.len();
            for (col_idx, &(j, k)) in self.epi_cols.iter().enumerate() {
                let ce = &self.epigenetic_centroids[(idx * n_blocks + j) * 4..(idx * n_blocks + j) * 4 + 4];
                let byte = self.epi_strands[row_epi_offset + col_idx];
                for s in 0..4 {
                    let val = match (byte >> ((3 - s) * 2)) & 0b11 {
                        0b00 => ce[0], 0b01 => ce[1], 0b11 => ce[2], 0b10 => ce[3], _ => 0.0,
                    };
                    res[j * self.block_size + k * 4 + s] += val;
                }
            }
        }

        if !self.tri_strands.is_empty() {
            let row_tri_offset = idx * self.tri_cols.len();
            for (col_idx, &(j, k)) in self.tri_cols.iter().enumerate() {
                let ct = &self.triplet_centroids[(idx * n_blocks + j) * 4..(idx * n_blocks + j) * 4 + 4];
                let byte = self.tri_strands[row_tri_offset + col_idx];
                for s in 0..4 {
                    let val = match (byte >> ((3 - s) * 2)) & 0b11 {
                        0b00 => ct[0], 0b01 => ct[1], 0b11 => ct[2], 0b10 => ct[3], _ => 0.0,
                    };
                    res[j * self.block_size + k * 4 + s] += val;
                }
            }
        }

        let a_start = self.anchor_row_ptrs[idx];
        let a_end = self.anchor_row_ptrs[idx + 1];
        for k in a_start..a_end {
            res[self.anchor_indices[k] as usize] += self.anchor_values[k].to_f32();
        }

        Ok(res)
    }

    pub fn refine_centroids(&mut self, mut input: Vec<f32>, target: Vec<f32>, lr: f32) -> PyResult<()> {
        if !self.rmsnorm_weight.is_empty() {
            input = unsafe { rms_norm(&input, &self.rmsnorm_weight, self.eps) };
        }
        let n_blocks = self.in_features / self.block_size;
        let mut new_centroids = self.centroids.clone();

        new_centroids.par_chunks_mut(n_blocks * 4).enumerate().for_each(|(i, row_centroids)| {
            let row_offset = i * n_blocks * self.stride;
            let row_weights = &self.database[row_offset..row_offset + n_blocks * self.stride];
            let mut row_sum = unsafe { genomic_dot_product(row_weights, &input, row_centroids, self.stride, n_blocks, &[]) };

            let a_start = self.anchor_row_ptrs[i];
            let a_end = self.anchor_row_ptrs[i + 1];
            for k in a_start..a_end {
                row_sum += input[self.anchor_indices[k] as usize] * self.anchor_values[k].to_f32();
            }

            let grad_scale = (row_sum - target[i]) * lr;
            if grad_scale.abs() > 1e-8 {
                for j in 0..n_blocks {
                    let weights = &self.database[row_offset + j * self.stride..row_offset + (j + 1) * self.stride];
                    let input_block = &input[j * self.block_size..(j + 1) * self.block_size];
                    let mut dims = 0;
                    for k in 0..self.stride {
                        let byte = weights[k];
                        for s in 0..4 {
                            let bits = (byte >> ((3 - s) * 2)) & 0b11;
                            let c_idx = match bits { 0b00 => 0, 0b01 => 1, 0b11 => 2, 0b10 => 3, _ => 4 };
                            if c_idx < 4 {
                                row_centroids[j * 4 + c_idx] -= grad_scale * input_block[dims];
                            }
                            dims += 1;
                        }
                    }
                }
            }
        });
        self.centroids = new_centroids;
        Ok(())
    }

    pub fn refine_with_grads(&mut self, mut input: Vec<f32>, grads: Vec<f32>, lr: f32) -> PyResult<()> {
        if !self.rmsnorm_weight.is_empty() {
            input = unsafe { rms_norm(&input, &self.rmsnorm_weight, self.eps) };
        }
        let n_blocks = self.in_features / self.block_size;
        self.centroids.par_chunks_mut(n_blocks * 4).enumerate().for_each(|(i, row_centroids)| {
            if i >= grads.len() { return; }
            let grad_scale = grads[i] * lr;
            if grad_scale.abs() > 1e-8 {
                let row_offset = i * n_blocks * self.stride;
                for j in 0..n_blocks {
                    let weights = &self.database[row_offset + j * self.stride..row_offset + (j + 1) * self.stride];
                    let input_block = &input[j * self.block_size..(j + 1) * self.block_size];
                    let mut dims = 0;
                    for k in 0..self.stride {
                        let byte = weights[k];
                        for s in 0..4 {
                            let bits = (byte >> ((3 - s) * 2)) & 0b11;
                            let c_idx = match bits { 0b00 => 0, 0b01 => 1, 0b11 => 2, 0b10 => 3, _ => 4 };
                            if c_idx < 4 {
                                row_centroids[j * 4 + c_idx] -= grad_scale * input_block[dims];
                            }
                            dims += 1;
                        }
                    }
                }
            }
        });
        Ok(())
    }

    /// Propagación hacia atrás (Backward): Calcula el gradiente de la entrada.
    /// d_input = d_output * W^T
    pub fn backward(&self, d_output: Vec<f32>) -> PyResult<Vec<f32>> {
        let n_blocks = self.in_features / self.block_size;
        let mut d_input = vec![0.0f32; self.in_features];

        // Procesamos por bloques de entrada para paralelizar el acumulado
        d_input.par_chunks_mut(self.block_size).enumerate().for_each(|(j, d_in_block)| {
            for i in 0..self.out_features {
                let d_out_val = d_output[i];
                if d_out_val == 0.0 { continue; }

                let row_offset = i * n_blocks * self.stride;
                let weights = &self.database[row_offset + j * self.stride .. row_offset + (j + 1) * self.stride];
                let c_offset = (i * n_blocks + j) * 4;
                let row_centroids = &self.centroids[c_offset..c_offset + 4];

                let mut dims = 0;
                for k in 0..self.stride {
                    let byte = weights[k];
                    for s in 0..4 {
                        let bits = (byte >> ((3 - s) * 2)) & 0b11;
                        let val = match bits {
                            0b00 => row_centroids[0],
                            0b01 => row_centroids[1],
                            0b11 => row_centroids[2],
                            0b10 => row_centroids[3],
                            _ => 0.0,
                        };
                        d_in_block[dims] += d_out_val * val;
                        dims += 1;
                    }
                }
            }
            
            // Sumar impacto de las anclas (Sparse Anchors)
            // Nota: Aquí solo sumamos si el ancla está en este bloque j
        });

        // Impacto de anclas (se procesa de forma más eficiente por fila ya que son pocas)
        for i in 0..self.out_features {
            let d_out_val = d_output[i];
            if d_out_val == 0.0 { continue; }
            
            let start = self.anchor_row_ptrs[i];
            let end = self.anchor_row_ptrs[i+1];
            for k in start..end {
                let col = self.anchor_indices[k] as usize;
                let val = self.anchor_values[k].to_f32();
                d_input[col] += d_out_val * val;
            }
        }

        Ok(d_input)
    }

    pub fn monte_carlo_refine(&mut self, mut input: Vec<f32>, target: Vec<f32>, iterations: usize, noise_scale: f32) -> PyResult<()> {
        if !self.rmsnorm_weight.is_empty() {
            input = unsafe { rms_norm(&input, &self.rmsnorm_weight, self.eps) };
        }
        let n_blocks = self.in_features / self.block_size;
        let mut new_centroids = self.centroids.clone();

        new_centroids.par_chunks_mut(n_blocks * 4).enumerate().for_each(|(i, row_centroids)| {
            if i >= target.len() { return; }
            let row_offset = i * n_blocks * self.stride;
            let row_weights = &self.database[row_offset..row_offset + n_blocks * self.stride];
            
            let get_row_sum = |c_vals: &[f32]| -> f32 {
                let mut sum = unsafe { genomic_dot_product(row_weights, &input, c_vals, self.stride, n_blocks, &[]) };
                let a_start = self.anchor_row_ptrs[i];
                let a_end = self.anchor_row_ptrs[i + 1];
                for k in a_start..a_end {
                    sum += input[self.anchor_indices[k] as usize] * self.anchor_values[k].to_f32();
                }
                sum
            };

            let best_sum = get_row_sum(row_centroids);
            let mut best_error = (best_sum - target[i]).powi(2);
            let mut rng = rand::thread_rng();
            
            let mut candidate = row_centroids.to_vec();
            for _ in 0..iterations {
                for (j, val) in candidate.iter_mut().enumerate() {
                    *val = row_centroids[j] + (rng.gen::<f32>() * 2.0 - 1.0) * noise_scale;
                }
                
                let cand_sum = get_row_sum(&candidate);
                let cand_error = (cand_sum - target[i]).powi(2);
                
                if cand_error < best_error {
                    best_error = cand_error;
                    row_centroids.copy_from_slice(&candidate);
                }
            }
        });
        
        self.centroids = new_centroids;
        Ok(())
    }
}

impl GenomicLinear {
    pub fn apply_grad_to_row(&mut self, i: usize, input: &[f32], grad_scale: f32, n_blocks: usize) {
        let row_offset = i * n_blocks * self.stride;
        for j in 0..n_blocks {
            let c_offset = (i * n_blocks + j) * 4;
            let input_block = &input[j * self.block_size..(j + 1) * self.block_size];
            let weights = &self.database[row_offset + j * self.stride..row_offset + (j + 1) * self.stride];
            let mut dims = 0;
            for k in 0..self.stride {
                let byte = weights[k];
                for s in 0..4 {
                    let bits = (byte >> ((3 - s) * 2)) & 0b11;
                    let c_idx = match bits { 0b00 => 0, 0b01 => 1, 0b11 => 2, 0b10 => 3, _ => 4 };
                    if c_idx < 4 {
                        let delta = grad_scale * input_block[dims];
                        self.centroids[c_offset + c_idx] -= delta;
                    }
                    dims += 1;
                }
            }
        }
    }

    pub fn apply_mutation(&mut self, delta_centroids: Vec<f32>, undo: bool) -> PyResult<()> {
        for (i, d) in self.centroids.iter_mut().zip(delta_centroids) {
            if undo { *i += d; } else { *i -= d; }
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
        for (c, d) in self.centroids.iter_mut().zip(delta) { *c -= d; }
        Ok(())
    }

    pub fn apply_weighted_mutation(&mut self, delta: Vec<f32>, weight: f32) -> PyResult<()> {
        for (c, d) in self.centroids.iter_mut().zip(delta) { *c += d * weight; }
        Ok(())
    }
}
