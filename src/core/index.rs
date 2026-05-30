#[cfg(not(feature = "python"))]
use crate::pyo3_shim::*;
#[cfg(feature = "python")]
use pyo3::prelude::*;

use rayon::prelude::*;
use std::cmp::Ordering;

#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct GajeIndex {
    pub database: Vec<Vec<u8>>,
    pub centroids: Vec<f32>,
    pub dims: usize,
}

#[cfg_attr(feature = "python", pymethods)]
impl GajeIndex {
    #[cfg(feature = "python")]
    #[new]
    pub fn new(dims: usize, centroids: Vec<f32>) -> Self {
        GajeIndex {
            database: Vec::new(),
            centroids,
            dims,
        }
    }

    pub fn add_batch(&mut self, strands: Vec<Vec<u8>>) -> PyResult<()> {
        self.database.extend(strands);
        Ok(())
    }

    pub fn flat_search(&self, query_vector: Vec<f32>, k: usize) -> PyResult<Vec<(usize, f32)>> {
        let q_len = query_vector.len();
        let c = &self.centroids;
        let mut results: Vec<(usize, f32)> = self
            .database
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
                        let s = (3 - j) * 2;
                        let bits = (byte >> s) & 0b11;
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
}
