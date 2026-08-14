//! # 📊 Métricas y Análisis de Densidad Informativa

use rayon::prelude::*;

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(not(feature = "python"))]
use crate::pyo3_shim::{exceptions::PyValueError, PyResult};

#[cfg_attr(feature = "python", pyfunction)]
pub fn calculate_genomic_mse(weights: Vec<f32>, centroids: Vec<f32>) -> f32 {
    if centroids.len() != 4 {
        return 1.0;
    }
    weights
        .par_iter()
        .map(|&w| {
            let mut min_sq_diff = f32::MAX;
            for &c in &centroids {
                let diff = w - c;
                let sq_diff = diff * diff;
                if sq_diff < min_sq_diff {
                    min_sq_diff = sq_diff;
                }
            }
            min_sq_diff
        })
        .sum::<f32>()
        / (weights.len() as f32).max(1.0)
}

#[cfg_attr(feature = "python", pyfunction)]
pub fn calculate_mse_native(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("Vector length mismatch"));
    }
    let n = a.len();
    if n == 0 {
        return Ok(0.0);
    }
    let sum_sq_diff: f32 = a
        .par_iter()
        .zip(b.par_iter())
        .map(|(&va, &vb)| (va - vb).powi(2))
        .sum();
    Ok(sum_sq_diff / n as f32)
}

#[cfg_attr(feature = "python", pyfunction)]
pub fn calculate_cosine_similarity_native(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("Vector length mismatch"));
    }
    let n = a.len();
    if n == 0 {
        return Ok(0.0);
    }
    let dot: f32 = a
        .par_iter()
        .zip(b.par_iter())
        .map(|(&va, &vb)| va * vb)
        .sum();
    let norm_a: f32 = a.par_iter().map(|&v| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = b.par_iter().map(|&v| v * v).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }
    Ok(dot / (norm_a * norm_b))
}

#[cfg_attr(feature = "python", pyfunction)]
pub fn calculate_distribution_entropy_native(probs: Vec<f32>) -> PyResult<f32> {
    let entropy: f32 = probs
        .par_iter()
        .filter(|&&p| p > 1e-12)
        .map(|&p| -p * p.ln() / 2.0f32.ln())
        .sum();
    Ok(entropy)
}

/// # Detección de Incertidumbre Semántica
///
/// Calcula una aproximación rápida de la entropía de un vector de activaciones.
/// Se utiliza para decidir si activar las hebras de ARN (precisión adaptativa).
pub fn calculate_activation_entropy(input: &[f32]) -> f32 {
    if input.is_empty() {
        return 0.0;
    }
    let n = input.len() as f32;
    let mean = input.iter().sum::<f32>() / n;
    let variance = input.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;

    // Usamos la varianza como proxy de entropía para activaciones normalizadas (RMSNorm)
    // Alta varianza -> Alta incertidumbre/Complejidad
    variance.sqrt()
}

/// Decide si se deben activar las hebras de ARN basándose en un umbral de entropía.
pub fn should_activate_rna(entropy: f32, threshold: f32) -> bool {
    entropy > threshold
}

/// # Analizador de Entropía de Shannon (Sovereign Information Density)
///
/// Calcula la entropía informativa por dimensión para un tensor dado.
/// Esto permite identificar qué dimensiones son ricas en información (alta entropía)
/// y cuáles son ruido o redundantes (baja entropía).
pub fn calculate_shannon_entropy_core(
    data: &[f32],
    rows: usize,
    cols: usize,
    bins: usize,
) -> Vec<f32> {
    (0..cols)
        .into_par_iter()
        .map(|col_idx| {
            let mut col_data = Vec::with_capacity(rows);
            for r in 0..rows {
                col_data.push(data[r * cols + col_idx]);
            }

            let min = col_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = col_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let range = max - min;

            if range < 1e-7 {
                return 0.0;
            }

            let mut histogram = vec![0usize; bins];
            for &val in &col_data {
                let bin_idx = (((val - min) / range) * (bins - 1) as f32) as usize;
                histogram[bin_idx.min(bins - 1)] += 1;
            }

            let mut entropy = 0.0;
            let n = rows as f32;
            for &count in &histogram {
                if count > 0 {
                    let p = count as f32 / n;
                    entropy -= p * p.log2();
                }
            }
            entropy
        })
        .collect()
}

#[cfg_attr(feature = "python", pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (data_u8, rows, cols, bins=64)))]
pub fn calculate_shannon_entropy(
    data_u8: Vec<u8>,
    rows: usize,
    cols: usize,
    bins: usize,
) -> PyResult<Vec<f32>> {
    let data_f32: &[f32] =
        unsafe { std::slice::from_raw_parts(data_u8.as_ptr() as *const f32, data_u8.len() / 4) };
    Ok(calculate_shannon_entropy_core(data_f32, rows, cols, bins))
}

/// # Entropía Genómica (2-bit Shannon Entropy)
///
/// Calcula la entropía de Shannon directamente sobre los pesos empaquetados de 2 bits.
/// Útil para medir la densidad informativa de una capa de ADN sin de-cuantizar.
pub fn calculate_genomic_entropy_core(dna_packed: &[u8]) -> f32 {
    if dna_packed.is_empty() {
        return 0.0;
    }
    let mut counts = [0usize; 4];
    for &byte in dna_packed {
        counts[((byte >> 6) & 0b11) as usize] += 1;
        counts[((byte >> 4) & 0b11) as usize] += 1;
        counts[((byte >> 2) & 0b11) as usize] += 1;
        counts[(byte & 0b11) as usize] += 1;
    }
    let total = (dna_packed.len() * 4) as f32;
    let mut entropy = 0.0f32;
    for &count in &counts {
        if count > 0 {
            let p = count as f32 / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}

#[cfg_attr(feature = "python", pyfunction)]
pub fn calculate_genomic_entropy(dna_packed: Vec<u8>) -> PyResult<f32> {
    Ok(calculate_genomic_entropy_core(&dna_packed))
}

#[cfg_attr(feature = "python", pyfunction)]
pub fn rms_norm_py(input: Vec<f32>, weight: Vec<f32>, eps: f32) -> PyResult<Vec<f32>> {
    Ok(unsafe { crate::compute::kernels::rms_norm(&input, &weight, eps) })
}