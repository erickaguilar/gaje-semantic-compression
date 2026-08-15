//! # 🧬 Espacio de Fase Toroidal (Complejo)
//!
//! Operaciones sobre el cuerpo ciclotómico $\mathbb{Q}(\zeta_{16})$: las neuronas actúan
//! como osciladores que interfieren constructiva o destructivamente, preservando la
//! densidad semántica en una fracción del espacio original.

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyBytes;

#[cfg(not(feature = "python"))]
use crate::pyo3_shim::{exceptions::PyValueError, PyObject, PyResult, Python};

// --- Look-Up Tables para Optimización ARM ---
lazy_static::lazy_static! {
    static ref SIN_LUT: [f32; 256] = {
        let mut lut = [0.0f32; 256];
        for i in 0..256 {
            let angle = (i as f32 / 256.0) * 2.0 * std::f32::consts::PI;
            lut[i] = angle.sin();
        }
        lut
    };
    static ref COS_LUT: [f32; 256] = {
        let mut lut = [0.0f32; 256];
        for i in 0..256 {
            let angle = (i as f32 / 256.0) * 2.0 * std::f32::consts::PI;
            lut[i] = angle.cos();
        }
        lut
    };
}

/// Obtiene el seno rápido desde la LUT
pub fn fast_sin(phase: f32) -> f32 {
    let normalized = (phase / (2.0 * std::f32::consts::PI)).rem_euclid(1.0);
    let idx = (normalized * 255.0) as usize;
    SIN_LUT[idx]
}

/// Obtiene el coseno rápido desde la LUT
pub fn fast_cos(phase: f32) -> f32 {
    let normalized = (phase / (2.0 * std::f32::consts::PI)).rem_euclid(1.0);
    let idx = (normalized * 255.0) as usize;
    COS_LUT[idx]
}

pub fn quantize_phase_core(real: &[f32], imag: &[f32]) -> Vec<u8> {
    let n = real.len();
    let mut packed = Vec::with_capacity((n + 3) / 4);
    for i in (0..n).step_by(4) {
        let mut byte = 0u8;
        for j in 0..4 {
            if i + j < n {
                let r = real[i + j];
                let im = imag[i + j];
                // atan2 returns values in (-PI, PI]
                let angle = im.atan2(r);

                let bits = if (0.0..std::f32::consts::FRAC_PI_2).contains(&angle) {
                    0b00 // Quadrant I: 0 to 90 deg (A)
                } else if (std::f32::consts::FRAC_PI_2..=std::f32::consts::PI).contains(&angle) {
                    0b01 // Quadrant II: 90 to 180 deg (C)
                } else if (-std::f32::consts::PI..-std::f32::consts::FRAC_PI_2).contains(&angle) {
                    0b11 // Quadrant III: 180 to 270 deg (G)
                } else {
                    0b10 // Quadrant IV: 270 to 360 deg (T)
                };
                byte = (byte << 2) | bits;
            }
        }
        packed.push(byte);
    }
    packed
}

#[cfg_attr(feature = "python", pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (real, imag)))]
pub fn quantize_phase_native(
    real: Vec<f32>,
    imag: Vec<f32>,
    _py: Python<'_>,
) -> PyResult<PyObject> {
    if real.len() != imag.len() {
        return Err(PyValueError::new_err(
            "Real and Imaginary parts must have the same length",
        ));
    }
    let _packed = quantize_phase_core(&real, &imag);
    #[cfg(feature = "python")]
    {
        Ok(PyBytes::new(_py, &_packed).into())
    }
    #[cfg(not(feature = "python"))]
    {
        Err("Python not enabled".to_string())
    }
}

pub fn dequantize_phase_core(dna_packed: &[u8], dims: usize) -> (Vec<f32>, Vec<f32>) {
    let mut real = Vec::with_capacity(dims);
    let mut imag = Vec::with_capacity(dims);
    let mut dp = 0;
    for &byte in dna_packed {
        for j in 0..4 {
            if dp >= dims {
                break;
            }
            let s = (3 - j) * 2;
            let bits = (byte >> s) & 0b11;
            let (r, im) = match bits {
                0b00 => (1.0, 0.0),  // A: 0 deg
                0b01 => (0.0, 1.0),  // C: 90 deg
                0b11 => (-1.0, 0.0), // G: 180 deg
                0b10 => (0.0, -1.0), // T: 270 deg
                _ => (0.0, 0.0),
            };
            real.push(r);
            imag.push(im);
            dp += 1;
        }
    }
    (real, imag)
}

#[inline(always)]
pub fn complex_add(r1: f32, i1: f32, r2: f32, i2: f32) -> (f32, f32) {
    (r1 + r2, i1 + i2)
}

#[inline(always)]
pub fn complex_mul(r1: f32, i1: f32, r2: f32, i2: f32) -> (f32, f32) {
    (r1 * r2 - i1 * i2, r1 * i2 + i1 * r2)
}

#[cfg_attr(feature = "python", pyfunction)]
pub fn dequantize_phase_native(dna_packed: Vec<u8>, dims: usize) -> PyResult<(Vec<f32>, Vec<f32>)> {
    Ok(dequantize_phase_core(&dna_packed, dims))
}
