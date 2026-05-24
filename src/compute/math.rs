use half::f16;
use rayon::prelude::*;
use std::cmp::Ordering;
use rand::Rng;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyBytes;
#[cfg(not(feature = "python"))]
use crate::pyo3_shim::*;

// --- Lógica Interna Pura (Rust) ---

pub fn genomize_f32_core(f32_data: &[f32], block_size: usize, anchor_threshold: f32, custom_base_c: Option<[f32; 4]>) -> (Vec<u8>, Vec<f32>, Vec<u8>) {
    let n_elements = f32_data.len(); let n_blocks = n_elements / block_size;
    let mut dna_database = Vec::with_capacity(n_elements / 4); let mut all_centroids = Vec::with_capacity(n_blocks * 4);
    let mut anchors = vec![half::f16::ZERO; n_elements]; let base_c = custom_base_c.unwrap_or([-1.510f32, -0.4528, 0.4528, 1.510]);
    for i in 0..n_blocks {
        let start = i * block_size; let block_f32 = &f32_data[start..start + block_size];
        let mut sum = 0.0f32; for &val in block_f32 { sum += val; }
        let mean = sum / block_size as f32;
        let mut var_sum = 0.0f32; for &val in block_f32 { let diff = val - mean; var_sum += diff * diff; }
        let std = (var_sum / block_size as f32).sqrt() + 1e-6;
        let t = [mean - std, mean, mean + std];
        let c = [mean + base_c[0] * std, mean + base_c[1] * std, mean + base_c[2] * std, mean + base_c[3] * std];
        for k in 0..(block_size / 4) {
            let mut byte = 0u8;
            for s in 0..4 {
                let val = block_f32[k * 4 + s];
                let bits = if val < t[0] { 0b00 } else if val < t[1] { 0b01 } else if val < t[2] { 0b11 } else { 0b10 };
                let c_val = match bits { 0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0 };
                let residual = val - c_val;
                if anchor_threshold >= 0.0 && residual.abs() > anchor_threshold { anchors[start + k * 4 + s] = half::f16::from_f32(residual); }
                byte = (byte << 2) | bits;
            }
            dna_database.push(byte);
        }
        for &cv in &c { all_centroids.push(cv); }
    }
    let anchors_u8 = unsafe { std::slice::from_raw_parts(anchors.as_ptr() as *const u8, anchors.len() * 2).to_vec() };
    (dna_database, all_centroids, anchors_u8)
}

pub fn dequantize_embedding_core(dna_packed: &[u8], dims: usize, centroids: Option<&[f32]>) -> Result<Vec<f32>, String> {
    let c = centroids.unwrap_or(&[-0.68, -0.17, 0.17, 0.68]);
    let mut rec = Vec::with_capacity(dims); let mut dp = 0; let is_multi = c.len() == dims * 4;
    for &byte in dna_packed {
        for j in 0..4 {
            if dp >= dims { break; }
            let s = (3 - j) * 2; let bits = (byte >> s) & 0b11;
            let cent = if is_multi { let b = dp * 4; match bits { 0b00 => c[b], 0b01 => c[b + 1], 0b11 => c[b + 2], 0b10 => c[b + 3], _ => 0.0 } }
            else { match bits { 0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0 } };
            rec.push(cent); dp += 1;
        }
    }
    Ok(rec)
}

pub fn dequantize_q8_0_core(data_u8: &[u8], out_features: usize, in_features: usize) -> Vec<f32> {
    let n_blocks = in_features / 32; let block_size = 34;
    let mut results = vec![0.0f32; out_features * in_features];
    results.par_chunks_mut(in_features).enumerate().for_each(|(i, row)| {
        let row_off = i * n_blocks * block_size;
        for b in 0..n_blocks {
            let off = row_off + b * block_size; if off + 2 > data_u8.len() { break; }
            let delta = f16::from_le_bytes([data_u8[off], data_u8[off + 1]]).to_f32();
            for j in 0..32 { if off + 2 + j >= data_u8.len() { break; } row[b * 32 + j] = (data_u8[off + 2 + j] as i8 as f32) * delta; }
        }
    });
    results
}

// --- Interfaz PyO3 (Python Wrappers) ---

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (dna_packed, dims, centroids=None))]
pub fn dequantize_embedding(dna_packed: Vec<u8>, dims: usize, centroids: Option<Vec<f32>>) -> PyResult<Vec<f32>> {
    dequantize_embedding_core(&dna_packed, dims, centroids.as_deref()).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

#[cfg(feature = "python")]
#[pyfunction]
pub fn genomize_f32_native(data_u8: Vec<u8>, block_size: usize, anchor_threshold: f32, py: Python<'_>) -> PyResult<(PyObject, Vec<f32>, PyObject)> {
    let f32_data: &[f32] = unsafe { std::slice::from_raw_parts(data_u8.as_ptr() as *const f32, data_u8.len() / 4) };
    let (dna, centroids, anchors) = genomize_f32_core(f32_data, block_size, anchor_threshold, None);
    Ok((PyBytes::new(py, &dna).into(), centroids, PyBytes::new(py, &anchors).into()))
}

#[cfg(feature = "python")]
#[pyfunction]
pub fn sample_top_p(logits: Vec<f32>, temperature: f32, top_p: f32) -> PyResult<usize> {
    if logits.is_empty() { return Ok(0); }
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut probs: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &l)| (i, ((l - max_logit) / temperature).exp())).collect();
    let sum_exp: f32 = probs.iter().map(|(_, p)| p).sum();
    for p in &mut probs { p.1 /= sum_exp; }
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let mut cumulative_prob = 0.0;
    let mut cutoff_idx = probs.len();
    for (i, &(_, p)) in probs.iter().enumerate() { cumulative_prob += p; if cumulative_prob > top_p { cutoff_idx = i + 1; break; } }
    probs.truncate(cutoff_idx);
    let final_sum: f32 = probs.iter().map(|(_, p)| p).sum();
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen::<f32>() * final_sum;
    let mut current_sum = 0.0;
    for &(id, p) in &probs { current_sum += p; if r <= current_sum { return Ok(id); } }
    Ok(probs[0].0)
}
