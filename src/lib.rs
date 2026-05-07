use pyo3::prelude::*;
use rayon::prelude::*;

/// Cuantización Genómica de 2 bits con soporte para múltiples umbrales
#[pyfunction]
#[pyo3(signature = (vector, thresholds=None))]
pub fn quantize_embedding(vector: Vec<f32>, thresholds: Option<Vec<f32>>) -> PyResult<Vec<u8>> {
    let t = thresholds.unwrap_or_else(|| vec![-0.34, 0.0, 0.34]);
    let sub_vector_size = 4;
    let num_sub_vectors = vector.len() / sub_vector_size;
    let mut packed = Vec::with_capacity(num_sub_vectors);
    let is_multi = t.len() == vector.len() * 3;

    for i in 0..num_sub_vectors {
        let start = i * sub_vector_size;
        let mut current_byte = 0u8;
        for j in 0..sub_vector_size {
            let dim_idx = start + j;
            let val = vector[dim_idx];
            let (t0, t1, t2) = if is_multi {
                (t[dim_idx * 3], t[dim_idx * 3 + 1], t[dim_idx * 3 + 2])
            } else {
                (t[0], t[1], t[2])
            };
            let bits = match val {
                v if v < t0 => 0b00, // A
                v if v < t1 => 0b01, // C
                v if v < t2 => 0b11, // G
                _ => 0b10,           // T
            };
            current_byte = (current_byte << 2) | bits;
        }
        packed.push(current_byte);
    }
    Ok(packed)
}

#[pyfunction]
#[pyo3(signature = (vector, thresholds=None))]
pub fn quantize_pq(vector: Vec<f32>, thresholds: Option<Vec<f32>>) -> PyResult<Vec<u8>> {
    quantize_embedding(vector, thresholds)
}

/// Búsqueda Asimétrica (ADC) Paralelizada con Rayon
#[pyfunction]
#[pyo3(signature = (query_vector, database, centroids=None))]
pub fn dna_similarity_search_adc(
    query_vector: Vec<f32>,
    database: Vec<Vec<u8>>,
    centroids: Option<Vec<f32>>,
) -> PyResult<Vec<(usize, f32)>> {
    let c = centroids.unwrap_or_else(|| vec![-0.68, -0.17, 0.17, 0.68]);
    let is_multi = c.len() == query_vector.len() * 4;
    let q_len = query_vector.len();

    let mut results: Vec<(usize, f32)> = database
        .par_iter()
        .enumerate()
        .map(|(idx, strand)| {
            let mut squared_distance = 0.0f32;
            let mut dims_processed = 0;
            for &byte in strand {
                for j in 0..4 {
                    if dims_processed >= q_len {
                        break;
                    }
                    let shift = (3 - j) * 2;
                    let bits = (byte >> shift) & 0b11;
                    let centroid = if is_multi {
                        let base_idx = dims_processed * 4;
                        match bits {
                            0b00 => c[base_idx],
                            0b01 => c[base_idx + 1],
                            0b11 => c[base_idx + 2],
                            0b10 => c[base_idx + 3],
                            _ => unreachable!(),
                        }
                    } else {
                        match bits {
                            0b00 => c[0],
                            0b01 => c[1],
                            0b11 => c[2],
                            0b10 => c[3],
                            _ => unreachable!(),
                        }
                    };
                    let diff = query_vector[dims_processed] - centroid;
                    squared_distance += diff * diff;
                    dims_processed += 1;
                }
            }
            (idx, squared_distance.sqrt())
        })
        .collect();

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(results)
}

#[pyfunction]
#[pyo3(signature = (query, database, centroids=None))]
pub fn dna_similarity_search(
    query: PyObject,
    database: Vec<Vec<u8>>,
    centroids: Option<Vec<f32>>,
    py: Python<'_>,
) -> PyResult<Vec<(usize, f32)>> {
    if let Ok(query_vector) = query.extract::<Vec<f32>>(py) {
        return dna_similarity_search_adc(query_vector, database, centroids);
    }
    if let Ok(query_dna) = query.extract::<Vec<u8>>(py) {
        let c = centroids.unwrap_or_else(|| vec![-0.68, -0.17, 0.17, 0.68]);
        let mut results: Vec<(usize, f32)> = database
            .par_iter()
            .enumerate()
            .map(|(idx, strand)| {
                let mut dist = 0.0f32;
                for i in 0..std::cmp::min(query_dna.len(), strand.len()) {
                    let b1 = query_dna[i];
                    let b2 = strand[i];
                    for j in 0..4 {
                        let shift = (3 - j) * 2;
                        let v1_bits = (b1 >> shift) & 0b11;
                        let v2_bits = (b2 >> shift) & 0b11;
                        let v1 = match v1_bits {
                            0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0,
                        };
                        let v2 = match v2_bits {
                            0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0,
                        };
                        let diff = v1 - v2;
                        dist += diff * diff;
                    }
                }
                (idx, dist.sqrt())
            })
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        return Ok(results);
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Query debe ser Vec<f32> o Vec<u8>"))
}

#[pyfunction]
#[pyo3(signature = (dna_packed, dims, centroids=None))]
pub fn dequantize_embedding(
    dna_packed: Vec<u8>,
    dims: usize,
    centroids: Option<Vec<f32>>,
) -> PyResult<Vec<f32>> {
    let c = centroids.unwrap_or_else(|| vec![-0.68, -0.17, 0.17, 0.68]);
    let is_multi = c.len() == dims * 4;
    let mut reconstructed = Vec::with_capacity(dims);
    let mut dims_processed = 0;
    for &byte in &dna_packed {
        for j in 0..4 {
            if dims_processed >= dims { break; }
            let shift = (3 - j) * 2;
            let bits = (byte >> shift) & 0b11;
            let centroid = if is_multi {
                let base_idx = dims_processed * 4;
                match bits {
                    0b00 => c[base_idx], 0b01 => c[base_idx + 1], 0b11 => c[base_idx + 2], 0b10 => c[base_idx + 3], _ => unreachable!(),
                }
            } else {
                match bits {
                    0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => unreachable!(),
                }
            };
            reconstructed.push(centroid);
            dims_processed += 1;
        }
    }
    Ok(reconstructed)
}

#[pymodule]
fn dna_semantic_compression(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quantize_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_pq, m)?)?;
    m.add_function(wrap_pyfunction!(dna_similarity_search_adc, m)?)?;
    m.add_function(wrap_pyfunction!(dna_similarity_search, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_embedding, m)?)?;
    Ok(())
}
