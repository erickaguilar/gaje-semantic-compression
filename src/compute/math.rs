use half::f16;
use rayon::prelude::*;
use std::cmp::Ordering;
use rand::Rng;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyBytes;
#[cfg(feature = "python")]
use pyo3::exceptions::{PyTypeError, PyValueError};

#[cfg(not(feature = "python"))]
use crate::pyo3_shim::{PyResult, Python, PyObject, exceptions::{PyTypeError, PyValueError}};

// --- Lógica Interna Pura (Rust) ---

pub fn genomize_f32_core(
    f32_data: &[f32],
    block_size: usize,
    anchor_threshold: f32,
    custom_base_c: Option<[f32; 4]>,
) -> (Vec<u8>, Vec<f32>, Vec<u8>) {
    let n_elements = f32_data.len();
    let n_blocks = n_elements / block_size;

    let mut dna_database = Vec::with_capacity(n_elements / 4);
    let mut all_centroids = Vec::with_capacity(n_blocks * 4);
    let mut anchors = vec![half::f16::ZERO; n_elements];

    let base_c = custom_base_c.unwrap_or([-1.510f32, -0.4528, 0.4528, 1.510]);

    for i in 0..n_blocks {
        let start = i * block_size;
        let block_f32 = &f32_data[start..start + block_size];

        let mut sum = 0.0f32;
        for &val in block_f32 {
            sum += val;
        }
        let mean = sum / block_size as f32;

        let mut var_sum = 0.0f32;
        for &val in block_f32 {
            let diff = val - mean;
            var_sum += diff * diff;
        }
        let std = (var_sum / block_size as f32).sqrt() + 1e-6;

        let t = [mean - std, mean, mean + std];
        let c = [
            mean + base_c[0] * std,
            mean + base_c[1] * std,
            mean + base_c[2] * std,
            mean + base_c[3] * std,
        ];

        for k in 0..(block_size / 4) {
            let mut byte = 0u8;
            for s in 0..4 {
                let val = block_f32[k * 4 + s];
                let bits = if val < t[0] {
                    0b00
                } else if val < t[1] {
                    0b01
                } else if val < t[2] {
                    0b11
                } else {
                    0b10
                };

                let c_val = match bits {
                    0b00 => c[0],
                    0b01 => c[1],
                    0b11 => c[2],
                    0b10 => c[3],
                    _ => 0.0,
                };

                let residual = val - c_val;
                if residual.abs() > anchor_threshold {
                    anchors[start + k * 4 + s] = half::f16::from_f32(residual);
                }
                byte = (byte << 2) | bits;
            }
            dna_database.push(byte);
        }
        for &cv in &c {
            all_centroids.push(cv);
        }
    }

    let anchors_u8 = unsafe {
        std::slice::from_raw_parts(anchors.as_ptr() as *const u8, anchors.len() * 2).to_vec()
    };

    (dna_database, all_centroids, anchors_u8)
}

pub fn genomize_f16_core(
    f16_data: &[f16],
    block_size: usize,
    anchor_threshold: f32,
    custom_base_c: Option<[f32; 4]>,
) -> (Vec<u8>, Vec<f32>, Vec<u8>) {
    let n_elements = f16_data.len();
    let n_blocks = n_elements / block_size;
    let mut dna_database = Vec::with_capacity(n_elements / 4);
    let mut all_centroids = Vec::with_capacity(n_blocks * 4);
    let mut anchors = vec![half::f16::ZERO; n_elements];
    let base_c = custom_base_c.unwrap_or([-1.510f32, -0.4528, 0.4528, 1.510]);
    for i in 0..n_blocks {
        let start = i * block_size;
        let block_f16 = &f16_data[start..start + block_size];
        let mut block_f32 = vec![0.0f32; block_size];
        let mut sum = 0.0f32;
        for j in 0..block_size {
            let val = block_f16[j].to_f32();
            block_f32[j] = val;
            sum += val;
        }
        let mean = sum / block_size as f32;
        let mut var_sum = 0.0f32;
        for &val in &block_f32 {
            let diff = val - mean;
            var_sum += diff * diff;
        }
        let std = (var_sum / block_size as f32).sqrt() + 1e-6;
        let t = [mean - std, mean, mean + std];
        let c = [
            mean + base_c[0] * std,
            mean + base_c[1] * std,
            mean + base_c[2] * std,
            mean + base_c[3] * std,
        ];

        for k in 0..(block_size / 4) {
            let mut byte = 0u8;
            for s in 0..4 {
                let val = block_f32[k * 4 + s];
                let bits = if val < t[0] {
                    0b00
                } else if val < t[1] {
                    0b01
                } else if val < t[2] {
                    0b11
                } else {
                    0b10
                };

                let c_val = match bits {
                    0b00 => c[0],
                    0b01 => c[1],
                    0b11 => c[2],
                    0b10 => c[3],
                    _ => 0.0,
                };

                let residual = val - c_val;
                if residual.abs() > anchor_threshold {
                    anchors[start + k * 4 + s] = half::f16::from_f32(residual);
                }
                byte = (byte << 2) | bits;
            }
            dna_database.push(byte);
        }
        for &cv in &c {
            all_centroids.push(cv);
        }
    }

    let anchors_u8 = unsafe {
        std::slice::from_raw_parts(anchors.as_ptr() as *const u8, anchors.len() * 2).to_vec()
    };

    (dna_database, all_centroids, anchors_u8)
}

// --- Interfaz PyO3 (Python Wrappers) ---

pub fn dequantize_embedding_core(
    dna_packed: &[u8],
    dims: usize,
    centroids: Option<&[f32]>,
) -> Result<Vec<f32>, String> {
    let c = centroids.unwrap_or(&[-0.68, -0.17, 0.17, 0.68]);
    let mut rec = Vec::with_capacity(dims);
    let mut dp = 0;
    let is_multi = c.len() == dims * 4;
    for &byte in dna_packed {
        for j in 0..4 {
            if dp >= dims { break; }
            let s = (3 - j) * 2;
            let bits = (byte >> s) & 0b11;
            let cent = if is_multi {
                let b = dp * 4;
                match bits { 0b00 => c[b], 0b01 => c[b + 1], 0b11 => c[b + 2], 0b10 => c[b + 3], _ => 0.0 }
            } else {
                match bits { 0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0 }
            };
            rec.push(cent);
            dp += 1;
        }
    }
    Ok(rec)
}

#[cfg_attr(feature = "python", pyfunction)]
#[cfg_attr(feature = "python", pyo3(name = "dequantize_embedding", signature = (dna_packed, dims, centroids=None)))]
pub fn dequantize_embedding_py(
    dna_packed: Vec<u8>,
    dims: usize,
    centroids: Option<Vec<f32>>,
) -> PyResult<Vec<f32>> {
    dequantize_embedding_core(&dna_packed, dims, centroids.as_deref()).map_err(PyValueError::new_err)
}

#[cfg_attr(feature = "python", pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (vector, thresholds=None)))]
pub fn quantize_embedding(
    vector: Vec<f32>,
    thresholds: Option<Vec<f32>>,
    _py: Python<'_>,
) -> PyResult<PyObject> {
    let t = thresholds.unwrap_or_else(|| vec![-0.43, 0.0, 0.43]);
    let n = vector.len();
    let mut packed = Vec::with_capacity((n + 3) / 4);
    for i in (0..n).step_by(4) {
        let mut byte = 0u8;
        for j in 0..4 {
            if i + j < n {
                let val = vector[i + j];
                let bits = if val < t[0] {
                    0b00
                } else if val < t[1] {
                    0b01
                } else if val < t[2] {
                    0b11
                } else {
                    0b10
                };
                byte = (byte << 2) | bits;
            }
        }
        packed.push(byte);
    }
    #[cfg(feature = "python")]
    { Ok(PyBytes::new(_py, &packed).into()) }
    #[cfg(not(feature = "python"))]
    { Err("Python not enabled".to_string()) }
}

#[cfg_attr(feature = "python", pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (vector, thresholds=None)))]
pub fn quantize_pq(
    vector: Vec<f32>,
    thresholds: Option<Vec<f32>>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    quantize_embedding(vector, thresholds, py)
}

#[cfg_attr(feature = "python", pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (data_u8, block_size, anchor_threshold, custom_base_c=None)))]
pub fn genomize_f32_native(
    data_u8: Vec<u8>,
    block_size: usize,
    anchor_threshold: f32,
    custom_base_c: Option<Vec<f32>>,
    _py: Python<'_>,
) -> PyResult<(PyObject, Vec<f32>, PyObject)> {
    let f32_data: &[f32] =
        unsafe { std::slice::from_raw_parts(data_u8.as_ptr() as *const f32, data_u8.len() / 4) };

    let base_c_arr = if let Some(c) = custom_base_c {
        if c.len() != 4 {
            return Err(PyTypeError::new_err(
                "custom_base_c must have 4 elements",
            ));
        }
        Some([c[0], c[1], c[2], c[3]])
    } else {
        None
    };

    #[allow(unused_variables)]
    let (dna, centroids, anchors) =
        genomize_f32_core(f32_data, block_size, anchor_threshold, base_c_arr);

    #[cfg(feature = "python")]
    {
        let dna_py = PyBytes::new(_py, &dna).into();
        let anchors_py = PyBytes::new(_py, &anchors).into();
        Ok((dna_py, centroids, anchors_py))
    }
    #[cfg(not(feature = "python"))]
    { Err("Python not enabled".to_string()) }
}

#[cfg_attr(feature = "python", pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (data_u8, block_size, anchor_threshold, custom_base_c=None)))]
pub fn genomize_f16_native(
    data_u8: Vec<u8>,
    block_size: usize,
    anchor_threshold: f32,
    custom_base_c: Option<Vec<f32>>,
    _py: Python<'_>,
) -> PyResult<(PyObject, Vec<f32>, PyObject)> {
    let f16_data: &[f16] =
        unsafe { std::slice::from_raw_parts(data_u8.as_ptr() as *const f16, data_u8.len() / 2) };

    let base_c_arr = if let Some(c) = custom_base_c {
        if c.len() != 4 {
            return Err(PyTypeError::new_err(
                "custom_base_c must have 4 elements",
            ));
        }
        Some([c[0], c[1], c[2], c[3]])
    } else {
        None
    };

    #[allow(unused_variables)]
    let (dna, centroids, anchors) =
        genomize_f16_core(f16_data, block_size, anchor_threshold, base_c_arr);

    #[cfg(feature = "python")]
    {
        let dna_py = PyBytes::new(_py, &dna).into();
        let anchors_py = PyBytes::new(_py, &anchors).into();
        Ok((dna_py, centroids, anchors_py))
    }
    #[cfg(not(feature = "python"))]
    { Err("Python not enabled".to_string()) }
}

#[cfg_attr(feature = "python", pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (logits, temperature=1.0, top_p=0.9)))]
pub fn sample_top_p(logits: Vec<f32>, temperature: f32, top_p: f32) -> PyResult<usize> {
    if logits.is_empty() {
        return Ok(0);
    }
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, ((l - max_logit) / temperature).exp()))
        .collect();
    let sum_exp: f32 = probs.iter().map(|(_, p)| p).sum();
    for p in &mut probs {
        p.1 /= sum_exp;
    }
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let mut cumulative_prob = 0.0;
    let mut cutoff_idx = probs.len();
    for (i, &(_, p)) in probs.iter().enumerate() {
        cumulative_prob += p;
        if cumulative_prob > top_p {
            cutoff_idx = i + 1;
            break;
        }
    }
    probs.truncate(cutoff_idx);
    let final_sum: f32 = probs.iter().map(|(_, p)| p).sum();
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen::<f32>() * final_sum;
    let mut current_sum = 0.0;
    for &(id, p) in &probs {
        current_sum += p;
        if r <= current_sum {
            return Ok(id);
        }
    }
    Ok(probs[0].0)
}

#[cfg_attr(feature = "python", pyfunction)]
pub fn calculate_shannon_entropy(data_u8: Vec<u8>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
    if data_u8.is_empty() || rows == 0 || cols == 0 {
        return Ok(vec![]);
    }
    let f32_data: &[f32] =
        unsafe { std::slice::from_raw_parts(data_u8.as_ptr() as *const f32, data_u8.len() / 4) };
    let entropies: Vec<f32> = (0..cols)
        .into_par_iter()
        .map(|d_idx| {
            let mut values = Vec::with_capacity(rows);
            for r in 0..rows {
                values.push(f32_data[r * cols + d_idx]);
            }
            let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let range = max - min;
            if range < 1e-6 {
                return 0.0f32;
            }
            let n_bins = 64;
            let mut bins = vec![0usize; n_bins];
            for &v in &values {
                let bin_idx = (((v - min) / range) * (n_bins - 1) as f32) as usize;
                bins[bin_idx.min(n_bins - 1)] += 1;
            }
            let mut entropy = 0.0f32;
            for &count in &bins {
                if count > 0 {
                    let p = count as f32 / rows as f32;
                    entropy -= p * (p.ln() / 2.0f32.ln());
                }
            }
            entropy
        })
        .collect();
    Ok(entropies)
}

#[cfg_attr(feature = "python", pyfunction)]
pub fn dequantize_q8_0_native(
    data_u8: Vec<u8>,
    out_features: usize,
    in_features: usize,
) -> PyResult<Vec<f32>> {
    let n_blocks = in_features / 32;
    let block_size = 34;
    let mut results = vec![0.0f32; out_features * in_features];
    results
        .par_chunks_mut(in_features)
        .enumerate()
        .for_each(|(i, row)| {
            let row_offset = i * n_blocks * block_size;
            for b in 0..n_blocks {
                let offset = row_offset + b * block_size;
                if offset + 2 > data_u8.len() {
                    break;
                }
                let delta = f16::from_le_bytes([data_u8[offset], data_u8[offset + 1]]).to_f32();
                for j in 0..32 {
                    if offset + 2 + j >= data_u8.len() {
                        break;
                    }
                    row[b * 32 + j] = (data_u8[offset + 2 + j] as i8 as f32) * delta;
                }
            }
        });
    Ok(results)
}

#[cfg_attr(feature = "python", pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (query_vector, database, centroids=None, k=10)))]
pub fn dna_similarity_search_adc(
    query_vector: Vec<f32>,
    database: Vec<Vec<u8>>,
    centroids: Option<Vec<f32>>,
    k: usize,
) -> PyResult<Vec<(usize, f32)>> {
    let c = centroids.unwrap_or_else(|| vec![-0.68, -0.17, 0.17, 0.68]);
    let q_len = query_vector.len();
    let mut results: Vec<(usize, f32)> = database
        .par_iter()
        .enumerate()
        .map(|(idx, strand)| {
            let mut dist_sq = 0.0f32;
            let mut dims = 0;
            let is_multi = c.len() == q_len * 4;
            for &byte in strand {
                for j in 0..4 {
                    if dims >= q_len {
                        break;
                    }
                    let shift = (3 - j) * 2;
                    let bits = (byte >> shift) & 0b11;
                    let centroid = if is_multi {
                        let b = dims * 4;
                        match bits {
                            0b00 => c[b],
                            0b01 => c[b + 1],
                            0b11 => c[b + 2],
                            0b10 => c[b + 3],
                            _ => 0.0,
                        }
                    } else {
                        match bits {
                            0b00 => c[0],
                            0b01 => c[1],
                            0b11 => c[2],
                            0b10 => c[3],
                            _ => 0.0,
                        }
                    };
                    let diff = query_vector[dims] - centroid;
                    dist_sq += diff * diff;
                    dims += 1;
                }
            }
            (idx, dist_sq.sqrt())
        })
        .collect();
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    if k > 0 && k < results.len() {
        results.truncate(k);
    }
    Ok(results)
}

#[cfg_attr(feature = "python", pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (query, database, centroids=None, k=10)))]
#[allow(unused_variables)]
pub fn dna_similarity_search(
    query: PyObject,
    database: Vec<Vec<u8>>,
    centroids: Option<Vec<f32>>,
    k: usize,
    _py: Python<'_>,
) -> PyResult<Vec<(usize, f32)>> {
    #[cfg(feature = "python")]
    {
        if let Ok(qv) = query.extract::<Vec<f32>>(_py) {
            return dna_similarity_search_adc(qv, database, centroids, k);
        }
    }
    #[cfg(feature = "python")]
    {
        if let Ok(qd) = query.extract::<Vec<u8>>(_py) {
            let c = centroids.unwrap_or_else(|| vec![-0.68, -0.17, 0.17, 0.68]);
            let mut res: Vec<(usize, f32)> = database
                .par_iter()
                .enumerate()
                .map(|(idx, strand)| {
                    let mut d = 0.0f32;
                    for i in 0..std::cmp::min(qd.len(), strand.len()) {
                        let (b1, b2) = (qd[i], strand[i]);
                        for j in 0..4 {
                            let s = (3 - j) * 2;
                            let (v1b, v2b) = ((b1 >> s) & 0b11, (b2 >> s) & 0b11);
                            let v1 = match v1b { 0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0 };
                            let v2 = match v2b { 0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0 };
                            d += (v1 - v2).powi(2);
                        }
                    }
                    (idx, d.sqrt())
                })
                .collect();
            res.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            if k > 0 && k < res.len() { res.truncate(k); }
            return Ok(res);
        }
    }
    Err(PyTypeError::new_err("Query error"))
}

#[cfg_attr(feature = "python", pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (database, stride, active_dims)))]
pub fn prune_genomic_database(
    database: Vec<u8>,
    stride: usize,
    active_dims: Vec<usize>,
) -> PyResult<(Vec<u8>, usize)> {
    let n_strands = if stride == 0 { 0 } else { database.len() / stride };
    let new_dims = active_dims.len();
    let new_stride = (new_dims + 3) / 4;
    let mut new_database = Vec::with_capacity(n_strands * new_stride);
    for s_idx in 0..n_strands {
        let strand = &database[s_idx * stride..(s_idx + 1) * stride];
        let mut new_strand = vec![0u8; new_stride];
        for (new_d_idx, &old_d_idx) in active_dims.iter().enumerate() {
            let bits = (strand[old_d_idx / 4] >> ((3 - (old_d_idx % 4)) * 2)) & 0b11;
            new_strand[new_d_idx / 4] |= bits << ((3 - (new_d_idx % 4)) * 2);
        }
        new_database.extend(new_strand);
    }
    Ok((new_database, new_stride))
}

#[cfg_attr(feature = "python", pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (logits, repetition_penalty=1.2, last_tokens=None)))]
pub fn apply_repetition_penalty(
    logits: Vec<f32>,
    repetition_penalty: f32,
    last_tokens: Option<Vec<usize>>,
) -> PyResult<Vec<f32>> {
    let mut out = logits;
    if let Some(tokens) = last_tokens {
        for &tid in &tokens {
            if tid < out.len() {
                if out[tid] > 0.0 { out[tid] /= repetition_penalty; }
                else { out[tid] *= repetition_penalty; }
            }
        }
    }
    Ok(out)
}

fn quantile(data: &mut [f32], q: f32) -> f32 {
    if data.is_empty() { return 0.0; }
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let pos = (data.len() - 1) as f32 * q;
    let base = pos.floor() as usize;
    let rest = pos - base as f32;
    if base + 1 < data.len() { data[base] + rest * (data[base + 1] - data[base]) } else { data[base] }
}

#[cfg_attr(feature = "python", pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (entropy_per_dim, fidelity_level=0.8)))]
pub fn generate_precision_mask_native(
    entropy_per_dim: Vec<f32>,
    fidelity_level: f32,
) -> PyResult<Vec<u8>> {
    let mut data = entropy_per_dim.clone();
    let q_mid = quantile(&mut data, 1.0 - fidelity_level);
    let q_high = quantile(&mut data, 1.0 - (fidelity_level / 2.0));
    let mask: Vec<u8> = entropy_per_dim.iter().map(|&e| { if e > q_high { 2 } else if e > q_mid { 1 } else { 0 } }).collect();
    Ok(mask)
}

#[cfg_attr(feature = "python", pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (entropy_per_dim, threshold=0.01)))]
pub fn get_active_dimensions_native(
    entropy_per_dim: Vec<f32>,
    threshold: f32,
) -> PyResult<Vec<usize>> {
    let active_dims: Vec<usize> = entropy_per_dim.iter().enumerate().filter(|&(_, &e)| e > threshold).map(|(idx, _)| idx).collect();
    Ok(active_dims)
}

pub fn generate_random_dna(n_elements: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    let n_bytes = (n_elements + 3) / 4;
    let mut dna = vec![0u8; n_bytes];
    let mut state = rng.gen_range(-1.0..1.0f32);
    let momentum = 0.85f32;
    for i in 0..n_bytes {
        let mut byte = 0u8;
        for _ in 0..4 {
            let noise = rng.gen_range(-1.0..1.0f32);
            state = state * momentum + noise * (1.0 - momentum);
            let bits = if state < -0.4 { 0b00 } else if state < 0.0 { 0b01 } else if state < 0.4 { 0b11 } else { 0b10 };
            byte = (byte << 2) | bits;
        }
        dna[i] = byte;
    }
    dna
}

#[cfg_attr(feature = "python", pyfunction)]
pub fn calculate_genomic_mse(weights: Vec<f32>, centroids: Vec<f32>) -> f32 {
    if centroids.len() != 4 { return 1.0; }
    weights.par_iter().map(|&w| {
        let mut min_sq_diff = f32::MAX;
        for &c in &centroids { let diff = w - c; let sq_diff = diff * diff; if sq_diff < min_sq_diff { min_sq_diff = sq_diff; } }
        min_sq_diff
    }).sum::<f32>() / (weights.len() as f32).max(1.0)
}

#[cfg_attr(feature = "python", pyfunction)]
pub fn calculate_mse_native(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    if a.len() != b.len() { return Err(PyValueError::new_err("Vector length mismatch")); }
    let n = a.len(); if n == 0 { return Ok(0.0); }
    let sum_sq_diff: f32 = a.par_iter().zip(b.par_iter()).map(|(&va, &vb)| (va - vb).powi(2)).sum();
    Ok(sum_sq_diff / n as f32)
}

#[cfg_attr(feature = "python", pyfunction)]
pub fn calculate_cosine_similarity_native(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    if a.len() != b.len() { return Err(PyValueError::new_err("Vector length mismatch")); }
    let n = a.len(); if n == 0 { return Ok(0.0); }
    let dot: f32 = a.par_iter().zip(b.par_iter()).map(|(&va, &vb)| va * vb).sum();
    let norm_a: f32 = a.par_iter().map(|&v| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = b.par_iter().map(|&v| v * v).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return Ok(0.0); }
    Ok(dot / (norm_a * norm_b))
}

#[cfg_attr(feature = "python", pyfunction)]
pub fn calculate_distribution_entropy_native(probs: Vec<f32>) -> PyResult<f32> {
    let entropy: f32 = probs.par_iter().filter(|&&p| p > 1e-12).map(|&p| -p * p.ln() / 2.0f32.ln()).sum();
    Ok(entropy)
}

pub fn generate_default_centroids(n_blocks: usize) -> Vec<f32> {
    let mut centroids = Vec::with_capacity(n_blocks * 4);
    for _ in 0..n_blocks { centroids.push(-1.51); centroids.push(-0.45); centroids.push(0.45); centroids.push(1.51); }
    centroids
}
