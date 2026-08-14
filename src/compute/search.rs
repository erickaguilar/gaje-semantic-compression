//! # 🔍 Búsqueda de Similitud y Mantenimiento de Bases Genómicas

use rayon::prelude::*;
use std::cmp::Ordering;

#[cfg(feature = "python")]
use pyo3::exceptions::PyTypeError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(not(feature = "python"))]
use crate::pyo3_shim::{exceptions::PyTypeError, PyObject, PyResult, Python};

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
                            let v1 = match v1b {
                                0b00 => c[0],
                                0b01 => c[1],
                                0b11 => c[2],
                                0b10 => c[3],
                                _ => 0.0,
                            };
                            let v2 = match v2b {
                                0b00 => c[0],
                                0b01 => c[1],
                                0b11 => c[2],
                                0b10 => c[3],
                                _ => 0.0,
                            };
                            d += (v1 - v2).powi(2);
                        }
                    }
                    (idx, d.sqrt())
                })
                .collect();
            res.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            if k > 0 && k < res.len() {
                res.truncate(k);
            }
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
    let n_strands = if stride == 0 {
        0
    } else {
        database.len() / stride
    };
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