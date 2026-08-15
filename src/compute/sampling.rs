//! # 🎲 Muestreo Autoregresivo y Utilidades de Generación

use rand::Rng;
use std::cmp::Ordering;

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(not(feature = "python"))]
use crate::pyo3_shim::{exceptions::PyValueError, PyResult};

pub fn sample_top_p_core(logits: Vec<f32>, temperature: f32, top_p: f32) -> Result<usize, String> {
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
    if sum_exp <= 0.0 {
        return Ok(0);
    }
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
#[cfg_attr(feature = "python", pyo3(signature = (logits, temperature=1.0, top_p=0.9)))]
pub fn sample_top_p(logits: Vec<f32>, temperature: f32, top_p: f32) -> PyResult<usize> {
    sample_top_p_core(logits, temperature, top_p).map_err(PyValueError::new_err)
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
                if out[tid] > 0.0 {
                    out[tid] /= repetition_penalty;
                } else {
                    out[tid] *= repetition_penalty;
                }
            }
        }
    }
    Ok(out)
}

fn quantile(data: &mut [f32], q: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let pos = (data.len() - 1) as f32 * q;
    let base = pos.floor() as usize;
    let rest = pos - base as f32;
    if base + 1 < data.len() {
        data[base] + rest * (data[base + 1] - data[base])
    } else {
        data[base]
    }
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
    let mask: Vec<u8> = entropy_per_dim
        .iter()
        .map(|&e| {
            if e > q_high {
                2
            } else if e > q_mid {
                1
            } else {
                0
            }
        })
        .collect();
    Ok(mask)
}

#[cfg_attr(feature = "python", pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (entropy_per_dim, threshold=0.01)))]
pub fn get_active_dimensions_native(
    entropy_per_dim: Vec<f32>,
    threshold: f32,
) -> PyResult<Vec<usize>> {
    let active_dims: Vec<usize> = entropy_per_dim
        .iter()
        .enumerate()
        .filter(|&(_, &e)| e > threshold)
        .map(|(idx, _)| idx)
        .collect();
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
            let bits = if state < -0.4 {
                0b00
            } else if state < 0.0 {
                0b01
            } else if state < 0.4 {
                0b11
            } else {
                0b10
            };
            byte = (byte << 2) | bits;
        }
        dna[i] = byte;
    }
    dna
}
