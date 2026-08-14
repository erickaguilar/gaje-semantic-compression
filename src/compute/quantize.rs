use half::f16;
use rayon::prelude::*;
use std::cmp::Ordering;

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

    // Si anchor_threshold > 0 y < 1.0, lo interpretamos como densidad (ej: 0.01 = 1%)
    let mut actual_threshold = anchor_threshold;
    if anchor_threshold > 0.0 && anchor_threshold < 1.0 {
        let mut abs_vals: Vec<f32> = f32_data.iter().map(|v| v.abs()).collect();
        abs_vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let top_idx = (n_elements as f32 * anchor_threshold) as usize;
        actual_threshold = if top_idx < n_elements {
            abs_vals[top_idx]
        } else {
            0.0
        };
    }

    let mut anchor_indices = Vec::new();
    let mut anchor_values = Vec::new();
    let _anchor_row_ptrs = [0u64; 1]; // Temporal, se ajustará después si es multidimensional

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
                let idx = k * 4 + s;
                let val = block_f32[idx];
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
                if anchor_threshold >= 0.0 && val.abs() >= actual_threshold {
                    anchor_indices.push((start + idx) as u32);
                    anchor_values.push(half::f16::from_f32(residual));
                }
                byte = (byte << 2) | bits;
            }
            dna_database.push(byte);
        }
        for &cv in &c {
            all_centroids.push(cv);
        }
    }

    // Empaquetar en formato "GAJE"
    let mut anchors_u8 = Vec::new();
    if !anchor_indices.is_empty() {
        anchors_u8.extend_from_slice(b"GAJE");
        let count = anchor_indices.len() as u32;
        anchors_u8.extend_from_slice(&count.to_le_bytes());
        for &idx in &anchor_indices {
            anchors_u8.extend_from_slice(&idx.to_le_bytes());
        }
        for &val in &anchor_values {
            anchors_u8.extend_from_slice(&val.to_le_bytes());
        }
        // Para genomize simple, asumimos una sola fila virtual o dejamos row_ptrs para el llamador
        // NOTA: GenomicLinear::new espera row_ptrs de tamaño out_features + 1.
        // Aquí genomize_f32_core no conoce out_features directamente (solo n_elements).
        // Por ahora, pondremos [0, count] asumiendo una fila, pero lo ideal es que el llamador lo gestione.
        anchors_u8.extend_from_slice(&0u64.to_le_bytes());
        anchors_u8.extend_from_slice(&(count as u64).to_le_bytes());
    }

    (dna_database, all_centroids, anchors_u8)
}

/// # 🧬 Cuantización Toroidal (Fase Compleja)
///
/// Proyecta tensores f32 en el cuerpo ciclotómico Q(zeta_16), tratando los
/// valores como ángulos en un toroide semántico.
pub fn quantize_toroidal_core(data: &[f32], block_size: usize) -> (Vec<u8>, Vec<f32>) {
    let n_elements = data.len();
    let n_blocks = n_elements / block_size;
    let mut dna = Vec::with_capacity(n_elements / 4);
    let mut phase_centroids = Vec::with_capacity(n_blocks * 4);

    for i in 0..n_blocks {
        let block = &data[i * block_size..(i + 1) * block_size];

        // En la topología toroidal, los centroides representan los "polos" de resonancia
        // de la fase compleja. Usamos 4 polos cardinales por bloque.
        let mut sum = 0.0f32;
        for &v in block {
            sum += v;
        }
        let mean = sum / block_size as f32;

        // Determinamos la amplitud del toroide local (dispersión)
        let mut var = 0.0f32;
        for &v in block {
            var += (v - mean).powi(2);
        }
        let std = (var / block_size as f32).sqrt() + 1e-6;

        // Polos cardinales: N, S, E, W en el plano complejo semántico
        let c = [mean - std, mean + std, mean - std * 0.5, mean + std * 0.5];
        for &val in &c {
            phase_centroids.push(val);
        }

        for k in 0..(block_size / 4) {
            let mut byte = 0u8;
            for s in 0..4 {
                let val = block[k * 4 + s];
                // Mapeo a fase (2 bits)
                let bits = if val < c[0] {
                    0b00
                } else if val < c[2] {
                    0b01
                } else if val < c[3] {
                    0b11
                } else {
                    0b10
                };
                byte = (byte << 2) | bits;
            }
            dna.push(byte);
        }
    }
    (dna, phase_centroids)
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

    let mut actual_threshold = anchor_threshold;
    if anchor_threshold > 0.0 && anchor_threshold < 1.0 {
        let mut abs_vals: Vec<f32> = f16_data.iter().map(|v| v.to_f32().abs()).collect();
        abs_vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let top_idx = (n_elements as f32 * anchor_threshold) as usize;
        actual_threshold = if top_idx < n_elements {
            abs_vals[top_idx]
        } else {
            0.0
        };
    }

    let mut anchor_indices = Vec::new();
    let mut anchor_values = Vec::new();

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
                let idx = k * 4 + s;
                let val = block_f32[idx];
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
                if anchor_threshold >= 0.0 && val.abs() >= actual_threshold {
                    anchor_indices.push((start + idx) as u32);
                    anchor_values.push(half::f16::from_f32(residual));
                }
                byte = (byte << 2) | bits;
            }
            dna_database.push(byte);
        }
        for &cv in &c {
            all_centroids.push(cv);
        }
    }

    let mut anchors_u8 = Vec::new();
    if !anchor_indices.is_empty() {
        anchors_u8.extend_from_slice(b"GAJE");
        let count = anchor_indices.len() as u32;
        anchors_u8.extend_from_slice(&count.to_le_bytes());
        for &idx in &anchor_indices {
            anchors_u8.extend_from_slice(&idx.to_le_bytes());
        }
        for &val in &anchor_values {
            anchors_u8.extend_from_slice(&val.to_le_bytes());
        }
        anchors_u8.extend_from_slice(&0u64.to_le_bytes());
        anchors_u8.extend_from_slice(&(count as u64).to_le_bytes());
    }

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
            if dp >= dims {
                break;
            }
            let s = (3 - j) * 2;
            let bits = (byte >> s) & 0b11;
            let cent = if is_multi {
                let b = dp * 4;
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
    let raw_bytes = unsafe {
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
    let raw_bytes = unsafe {
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

pub fn genomize_4bit_core(
    f32_data: &[f32],
    block_size: usize,
    anchor_threshold: f32,
) -> (Vec<u8>, Vec<f32>, Vec<u8>) {
    let n_elements = f32_data.len();
    let n_blocks = n_elements / block_size;
    let mut dna_database = Vec::with_capacity(n_elements / 2);
    let mut all_centroids = Vec::with_capacity(n_blocks * 16);

    let mut actual_threshold = anchor_threshold;
    if anchor_threshold > 0.0 && anchor_threshold < 1.0 {
        let mut abs_vals: Vec<f32> = f32_data.iter().map(|v| v.abs()).collect();
        abs_vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let top_idx = (n_elements as f32 * anchor_threshold) as usize;
        actual_threshold = if top_idx < n_elements {
            abs_vals[top_idx]
        } else {
            0.0
        };
    }

    let mut anchor_indices = Vec::new();
    let mut anchor_values = Vec::new();

    for i in 0..n_blocks {
        let start = i * block_size;
        let block_f32 = &f32_data[start..start + block_size];

        // 16 centroides lineales para 4 bits en el rango del bloque
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

        let mut c = [0.0f32; 16];
        let step = (max_val - min_val) / 15.0;
        for j in 0..16 {
            c[j] = min_val + j as f32 * step;
            all_centroids.push(c[j]);
        }

        for k in 0..(block_size / 2) {
            let mut byte = 0u8;
            for s in 0..2 {
                let idx = k * 2 + s;
                let val = block_f32[idx];

                // Cuantización 4-bit (Centroide más cercano)
                let mut best_idx = 0;
                let mut min_dist = f32::MAX;
                for j in 0..16 {
                    let dist = (val - c[j]).abs();
                    if dist < min_dist {
                        min_dist = dist;
                        best_idx = j;
                    }
                }

                if s == 0 {
                    byte |= (best_idx as u8) << 4;
                } else {
                    byte |= best_idx as u8;
                }

                if anchor_threshold >= 0.0 && val.abs() >= actual_threshold {
                    anchor_indices.push((start + idx) as u32);
                    anchor_values.push(f16::from_f32(val));
                }
            }
            dna_database.push(byte);
        }
    }

    // Empaquetar anclas en formato GAJE
    let mut anchors_buf = Vec::new();
    anchors_buf.extend_from_slice(b"GAJE");
    anchors_buf.extend_from_slice(&(anchor_indices.len() as u32).to_le_bytes());
    for &idx in &anchor_indices {
        anchors_buf.extend_from_slice(&idx.to_le_bytes());
    }
    for &val in &anchor_values {
        anchors_buf.extend_from_slice(&val.to_le_bytes());
    }
    let row_ptrs = [0u64, anchor_indices.len() as u64]; // Simplificación para exportación de 1 layer
    for &ptr in &row_ptrs {
        anchors_buf.extend_from_slice(&ptr.to_le_bytes());
    }

    (dna_database, all_centroids, anchors_buf)
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
        let (dna, centroids, anchors) = genomize_4bit_core(&f32_data, block_size, anchor_threshold);
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
        let (dna, centroids, anchors) =
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

pub fn dequantize_q8_0_core(data_u8: &[u8], out_features: usize, in_features: usize) -> Vec<f32> {
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
                let delta =
                    half::f16::from_le_bytes([data_u8[offset], data_u8[offset + 1]]).to_f32();
                for j in 0..32 {
                    if offset + 2 + j >= data_u8.len() {
                        break;
                    }
                    row[b * 32 + j] = (data_u8[offset + 2 + j] as i8 as f32) * delta;
                }
            }
        });
    results
}

#[cfg(feature = "python")]
#[pyfunction]
pub fn dequantize_q8_0_native(
    data_u8: Vec<u8>,
    out_features: usize,
    in_features: usize,
) -> PyResult<Vec<f32>> {
    Ok(dequantize_q8_0_core(&data_u8, out_features, in_features))
}

pub fn generate_default_centroids(n_blocks: usize) -> Vec<f32> {
    let mut centroids = Vec::with_capacity(n_blocks * 4);
    for _ in 0..n_blocks {
        centroids.push(-1.51);
        centroids.push(-0.45);
        centroids.push(0.45);
        centroids.push(1.51);
    }
    centroids
}