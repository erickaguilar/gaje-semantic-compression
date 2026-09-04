// =============================================================================
// python — Wrappers #[pyfunction] de cuantización/genomización (feature `python`)
// =============================================================================
use half::f16;
use rayon::prelude::*;

#[cfg(feature = "python")]
use pyo3::exceptions::{PyTypeError, PyValueError};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyBytes;

#[cfg(not(feature = "python"))]
use crate::pyo3_shim::{
    exceptions::{PyTypeError, PyValueError},
    PyObject, PyResult, Python,
};

use crate::compute::quantize::dequantize::dequantize_embedding_core;
use crate::compute::quantize::genomize::{
    genomize_4bit_core, genomize_f16_core, genomize_f32_core,
};

#[cfg_attr(feature = "python", pyfunction)]
#[cfg_attr(feature = "python", pyo3(name = "dequantize_embedding", signature = (dna_packed, dims, centroids=None)))]
pub fn dequantize_embedding_py(
    dna_packed: Vec<u8>,
    dims: usize,
    centroids: Option<Vec<f32>>,
) -> PyResult<Vec<f32>> {
    dequantize_embedding_core(&dna_packed, dims, centroids.as_deref())
        .map_err(PyValueError::new_err)
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
    {
        Ok(PyBytes::new(_py, &packed).into())
    }
    #[cfg(not(feature = "python"))]
    {
        Err("Python not enabled".to_string())
    }
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
#[cfg_attr(feature = "python", pyo3(signature = (data_u8, block_size, anchor_threshold, bit_depth=2, custom_base_c=None)))]
pub fn genomize_f32_native(
    data_u8: Vec<u8>,
    block_size: usize,
    anchor_threshold: f32,
    bit_depth: u8,
    custom_base_c: Option<Vec<f32>>,
    _py: Python<'_>,
) -> PyResult<(PyObject, Vec<f32>, PyObject)> {
    let f32_data: &[f32] =
        unsafe { std::slice::from_raw_parts(data_u8.as_ptr() as *const f32, data_u8.len() / 4) };

    if bit_depth == 32 {
        #[cfg(feature = "python")]
        {
            use pyo3::types::PyBytes;
            let dna_py = PyBytes::new(_py, &data_u8).into();
            let anchors_py = PyBytes::new(_py, &[]).into();
            Ok((dna_py, vec![], anchors_py))
        }
        #[cfg(not(feature = "python"))]
        {
            Err("Python not enabled".to_string())
        }
    } else if bit_depth == 4 {
        let (_dna, _centroids, _anchors) =
            genomize_4bit_core(f32_data, block_size, anchor_threshold);
        #[cfg(feature = "python")]
        {
            use pyo3::types::PyBytes;
            let dna_py = PyBytes::new(_py, &_dna).into();
            let anchors_py = PyBytes::new(_py, &_anchors).into();
            Ok((dna_py, _centroids, anchors_py))
        }
        #[cfg(not(feature = "python"))]
        {
            Err("Python not enabled".to_string())
        }
    } else {
        let base_c_arr = if let Some(c) = custom_base_c {
            if c.len() != 4 {
                return Err(PyTypeError::new_err("custom_base_c must have 4 elements"));
            }
            Some([c[0], c[1], c[2], c[3]])
        } else {
            None
        };

        let (_dna, _centroids, _anchors) =
            genomize_f32_core(f32_data, block_size, anchor_threshold, base_c_arr);

        #[cfg(feature = "python")]
        {
            use pyo3::types::PyBytes;
            let dna_py = PyBytes::new(_py, &_dna).into();
            let anchors_py = PyBytes::new(_py, &_anchors).into();
            Ok((dna_py, _centroids, anchors_py))
        }
        #[cfg(not(feature = "python"))]
        {
            Err("Python not enabled".to_string())
        }
    }
}

#[cfg_attr(feature = "python", pyfunction)]
pub fn quantize_q4_0_native(data_u8: Vec<u8>, _py: Python<'_>) -> PyResult<PyObject> {
    let f32_data: &[f32] =
        unsafe { std::slice::from_raw_parts(data_u8.as_ptr() as *const f32, data_u8.len() / 4) };

    let n_elements = f32_data.len();
    if n_elements % 32 != 0 {
        return Err(PyTypeError::new_err(
            "Weights length must be divisible by 32",
        ));
    }

    let n_blocks = n_elements / 32;

    let out_blocks: Vec<crate::io::header::Q4_0Block> = _py.allow_threads(|| {
        (0..n_blocks)
            .into_par_iter()
            .map(|i| {
                let start = i * 32;
                let block_f32 = &f32_data[start..start + 32];

                let mut min_val = f32::MAX;
                let mut max_val = f32::MIN;
                for &v in block_f32 {
                    if v < min_val {
                        min_val = v;
                    }
                    if v > max_val {
                        max_val = v;
                    }
                }

                let scale = (max_val - min_val) / 15.0;
                let inv_scale = if scale > 1e-7 { 1.0 / scale } else { 0.0 };

                let mut qs = [0u8; 16];
                for k in 0..16 {
                    let q0 = if scale > 1e-7 {
                        (((block_f32[k * 2] - min_val) * inv_scale)
                            .round()
                            .clamp(0.0, 15.0)) as u8
                    } else {
                        0
                    };
                    let q1 = if scale > 1e-7 {
                        (((block_f32[k * 2 + 1] - min_val) * inv_scale)
                            .round()
                            .clamp(0.0, 15.0)) as u8
                    } else {
                        0
                    };
                    qs[k] = q0 | (q1 << 4);
                }

                crate::io::header::Q4_0Block {
                    scale: half::f16::from_f32(scale),
                    min: half::f16::from_f32(min_val),
                    qs,
                }
            })
            .collect()
    });

    // Convert out_blocks slice to raw bytes
    let _raw_bytes = unsafe {
        std::slice::from_raw_parts(
            out_blocks.as_ptr() as *const u8,
            out_blocks.len() * std::mem::size_of::<crate::io::header::Q4_0Block>(),
        )
    };

    #[cfg(feature = "python")]
    {
        use pyo3::types::PyBytes;
        let bytes_py = PyBytes::new(_py, raw_bytes).into();
        Ok(bytes_py)
    }
    #[cfg(not(feature = "python"))]
    {
        Err(PyTypeError::new_err("Python not enabled"))
    }
}

#[cfg_attr(feature = "python", pyfunction)]
pub fn quantize_q8_0_native(data_u8: Vec<u8>, _py: Python<'_>) -> PyResult<PyObject> {
    let f32_data: &[f32] =
        unsafe { std::slice::from_raw_parts(data_u8.as_ptr() as *const f32, data_u8.len() / 4) };

    let n_elements = f32_data.len();
    if n_elements % 32 != 0 {
        return Err(PyTypeError::new_err(
            "Weights length must be divisible by 32",
        ));
    }

    let n_blocks = n_elements / 32;

    let out_blocks: Vec<crate::io::header::Q8_0Block> = _py.allow_threads(|| {
        (0..n_blocks)
            .into_par_iter()
            .map(|i| {
                let start = i * 32;
                let block_f32 = &f32_data[start..start + 32];

                let mut max_abs = 0.0f32;
                for &v in block_f32 {
                    let abs_v = v.abs();
                    if abs_v > max_abs {
                        max_abs = abs_v;
                    }
                }

                let scale = max_abs / 127.0;
                let inv_scale = if scale > 1e-7 { 1.0 / scale } else { 0.0 };

                let mut qs = [0i8; 32];
                for k in 0..32 {
                    let q = if scale > 1e-7 {
                        (block_f32[k] * inv_scale).round().clamp(-128.0, 127.0) as i8
                    } else {
                        0
                    };
                    qs[k] = q;
                }

                crate::io::header::Q8_0Block {
                    scale: half::f16::from_f32(scale),
                    qs,
                }
            })
            .collect()
    });

    // Convert out_blocks slice to raw bytes
    let _raw_bytes = unsafe {
        std::slice::from_raw_parts(
            out_blocks.as_ptr() as *const u8,
            out_blocks.len() * std::mem::size_of::<crate::io::header::Q8_0Block>(),
        )
    };

    #[cfg(feature = "python")]
    {
        use pyo3::types::PyBytes;
        let bytes_py = PyBytes::new(_py, raw_bytes).into();
        Ok(bytes_py)
    }
    #[cfg(not(feature = "python"))]
    {
        Err(PyTypeError::new_err("Python not enabled"))
    }
}

/// Cuantiza un tensor f32 (raw bytes, len%32==0) a bloques Q2_0 (2 bits/peso +
/// scale/min por bloque de 32). Devuelve los bytes crudos de los bloques.
#[cfg_attr(feature = "python", pyfunction)]
pub fn quantize_q2_0_native(data_u8: Vec<u8>, _py: Python<'_>) -> PyResult<PyObject> {
    let f32_data: &[f32] =
        unsafe { std::slice::from_raw_parts(data_u8.as_ptr() as *const f32, data_u8.len() / 4) };

    let n_elements = f32_data.len();
    if n_elements % 32 != 0 {
        return Err(PyTypeError::new_err(
            "Weights length must be divisible by 32",
        ));
    }

    let n_blocks = n_elements / 32;

    let out_blocks: Vec<crate::io::header::Q2_0Block> = _py.allow_threads(|| {
        (0..n_blocks)
            .into_par_iter()
            .map(|i| {
                let start = i * 32;
                let block_f32 = &f32_data[start..start + 32];

                let mut min_val = f32::MAX;
                let mut max_val = f32::MIN;
                for &v in block_f32 {
                    if v < min_val {
                        min_val = v;
                    }
                    if v > max_val {
                        max_val = v;
                    }
                }

                let scale = (max_val - min_val) / 3.0; // 4 niveles (2 bits)
                let inv_scale = if scale > 1e-7 { 1.0 / scale } else { 0.0 };

                let mut qs = [0u8; 8];
                for k in 0..8 {
                    let mut byte = 0u8;
                    for j in 0..4 {
                        let q = if scale > 1e-7 {
                            (((block_f32[k * 4 + j] - min_val) * inv_scale)
                                .round()
                                .clamp(0.0, 3.0)) as u8
                        } else {
                            0
                        };
                        byte |= q << (j * 2);
                    }
                    qs[k] = byte;
                }

                crate::io::header::Q2_0Block {
                    scale: half::f16::from_f32(scale),
                    min: half::f16::from_f32(min_val),
                    qs,
                }
            })
            .collect()
    });

    // Convert out_blocks slice to raw bytes
    let _raw_bytes = unsafe {
        std::slice::from_raw_parts(
            out_blocks.as_ptr() as *const u8,
            out_blocks.len() * std::mem::size_of::<crate::io::header::Q2_0Block>(),
        )
    };

    #[cfg(feature = "python")]
    {
        use pyo3::types::PyBytes;
        let bytes_py = PyBytes::new(_py, raw_bytes).into();
        Ok(bytes_py)
    }
    #[cfg(not(feature = "python"))]
    {
        Err(PyTypeError::new_err("Python not enabled"))
    }
}

#[cfg_attr(feature = "python", pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (data_u8, block_size, anchor_threshold, bit_depth=2, custom_base_c=None)))]
pub fn genomize_f16_native(
    data_u8: Vec<u8>,
    block_size: usize,
    anchor_threshold: f32,
    bit_depth: u8,
    custom_base_c: Option<Vec<f32>>,
    _py: Python<'_>,
) -> PyResult<(PyObject, Vec<f32>, PyObject)> {
    let f16_data: &[f16] =
        unsafe { std::slice::from_raw_parts(data_u8.as_ptr() as *const f16, data_u8.len() / 2) };

    let base_c_arr = if let Some(c) = custom_base_c {
        if c.len() != 4 {
            return Err(PyTypeError::new_err("custom_base_c must have 4 elements"));
        }
        Some([c[0], c[1], c[2], c[3]])
    } else {
        None
    };

    if bit_depth == 4 {
        // Convertir F16 a F32 para procesar con el core de 4 bits existente
        let f32_data: Vec<f32> = f16_data.iter().map(|v| v.to_f32()).collect();
        let (_dna, _centroids, _anchors) = genomize_4bit_core(&f32_data, block_size, anchor_threshold);
        #[cfg(feature = "python")]
        {
            use pyo3::types::PyBytes;
            let dna_py = PyBytes::new(_py, &dna).into();
            let anchors_py = PyBytes::new(_py, &anchors).into();
            Ok((dna_py, centroids, anchors_py))
        }
        #[cfg(not(feature = "python"))]
        {
            Err("Python not enabled".into())
        }
    } else {
        let (_dna, _centroids, _anchors) =
            genomize_f16_core(f16_data, block_size, anchor_threshold, base_c_arr);

        #[cfg(feature = "python")]
        {
            let dna_py = PyBytes::new(_py, &dna).into();
            let anchors_py = PyBytes::new(_py, &anchors).into();
            Ok((dna_py, centroids, anchors_py))
        }
        #[cfg(not(feature = "python"))]
        {
            Err("Python not enabled".to_string())
        }
    }
}

#[cfg(feature = "python")]
#[pyfunction]
pub fn dequantize_q8_0_native(
    data_u8: Vec<u8>,
    out_features: usize,
    in_features: usize,
) -> PyResult<Vec<f32>> {
    Ok(crate::compute::quantize::dequantize::dequantize_q8_0_core(
        &data_u8,
        out_features,
        in_features,
    ))
}
