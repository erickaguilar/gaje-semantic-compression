use crate::compute::math::*;
use pyo3::prelude::*;
use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
#[derive(Copy, Clone, PartialEq)]
pub struct Neighbor {
    pub idx: usize,
    pub distance: f32,
}
impl Eq for Neighbor {}
impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance.partial_cmp(&self.distance)
    }
}
impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}
#[pyclass]
pub struct GajeIndex {
    #[pyo3(get)]
    pub database: Vec<u8>,
    #[pyo3(get, set)]
    pub centroids: Vec<f32>,
    #[pyo3(get)]
    pub epigenetic_database: Vec<u8>,
    #[pyo3(get, set)]
    pub epigenetic_centroids: Vec<f32>,
    #[pyo3(get)]
    pub triplet_database: Vec<u8>,
    #[pyo3(get, set)]
    pub triplet_centroids: Vec<f32>,
    #[pyo3(get, set)]
    pub precision_mask: Vec<u8>,
    #[pyo3(get)]
    pub stride: usize,
    pub layers: Vec<Vec<Vec<usize>>>,
    pub max_level: i32,
    pub entry_point: Option<usize>,
    pub ef_construction: usize,
    pub m: usize,
    pub level_mult: f64,
}
impl GajeIndex {
    pub fn get_strand(&self, idx: usize) -> &[u8] {
        let start = idx * self.stride;
        &self.database[start..start + self.stride]
    }
    pub fn get_epigenetic_strand(&self, idx: usize) -> &[u8] {
        let start = idx * self.stride;
        &self.epigenetic_database[start..start + self.stride]
    }
    pub fn get_triplet_strand(&self, idx: usize) -> &[u8] {
        let start = idx * self.stride;
        &self.triplet_database[start..start + self.stride]
    }
    pub fn calculate_distance_lut(
        &self,
        lut_base: &[f32],
        lut_epi: &[f32],
        lut_tri: &[f32],
        target_idx: usize,
    ) -> f32 {
        let strand_start = target_idx * self.stride;
        let strand_end = strand_start + self.stride;
        if strand_end > self.database.len() {
            return f32::MAX;
        }
        let strand = &self.database[strand_start..strand_end];
        let n_dims = lut_base.len() / 4;
        let has_mask = !self.precision_mask.is_empty();
        let has_epi_data =
            !self.epigenetic_database.is_empty() && self.epigenetic_database.len() >= strand_end;
        let has_tri_data =
            !self.triplet_database.is_empty() && self.triplet_database.len() >= strand_end;
        let default_mode = if has_tri_data {
            2u8
        } else if has_epi_data {
            1u8
        } else {
            0u8
        };
        #[cfg(target_arch = "aarch64")]
        {
            let epi_strand_actual = if has_epi_data {
                &self.epigenetic_database[strand_start..strand_end]
            } else {
                &[]
            };
            let tri_strand_actual = if has_tri_data {
                &self.triplet_database[strand_start..strand_end]
            } else {
                &[]
            };
            let dummy_mask;
            let mask = if has_mask {
                &self.precision_mask
            } else {
                dummy_mask = vec![default_mode; self.stride];
                &dummy_mask
            };
            let dummy_e;
            let e_s = if epi_strand_actual.is_empty() {
                dummy_e = vec![0u8; self.stride];
                &dummy_e
            } else {
                epi_strand_actual
            };
            let dummy_t;
            let t_s = if tri_strand_actual.is_empty() {
                dummy_t = vec![0u8; self.stride];
                &dummy_t
            } else {
                tri_strand_actual
            };
            return unsafe {
                calculate_distance_lut(
                    lut_base, lut_epi, lut_tri, strand, e_s, t_s, mask, n_dims,
                )
            };
        }
        #[allow(unreachable_code)]
        let mut dist_sq = 0.0f32;
        let mut dims = 0;
        for i in 0..self.stride {
            let mut mode = if has_mask {
                self.precision_mask[i]
            } else {
                default_mode
            };
            if mode == 2 && !has_tri_data {
                mode = if has_epi_data { 1 } else { 0 };
            }
            if mode == 1 && !has_epi_data {
                mode = 0;
            }
            let byte = strand[i];
            let offset = strand_start + i;
            for j in 0..4 {
                if dims >= n_dims {
                    break;
                }
                let shift = (3 - j) * 2;
                let bb = (byte >> shift) & 0b11;
                let b_idx = (bb ^ (bb >> 1)) as usize;
                if mode == 0 {
                    dist_sq += lut_base[dims * 4 + b_idx];
                } else if mode == 1 {
                    let eb_byte = *self.epigenetic_database.get(offset).unwrap_or(&0);
                    let eb = (eb_byte >> shift) & 0b11;
                    let e_idx = (eb ^ (eb >> 1)) as usize;
                    dist_sq += lut_epi[dims * 16 + (b_idx << 2 | e_idx)];
                } else {
                    let eb_byte = *self.epigenetic_database.get(offset).unwrap_or(&0);
                    let tb_byte = *self.triplet_database.get(offset).unwrap_or(&0);
                    let eb = (eb_byte >> shift) & 0b11;
                    let tb = (tb_byte >> shift) & 0b11;
                    let e_idx = (eb ^ (eb >> 1)) as usize;
                    let t_idx = (tb ^ (tb >> 1)) as usize;
                    dist_sq += lut_tri[dims * 64 + (b_idx << 4 | e_idx << 2 | t_idx)];
                }
                dims += 1;
            }
        }
        dist_sq.sqrt()
    }
    pub fn search_layer_lut(
        &self,
        lut_base: &[f32],
        lut_epi: &[f32],
        lut_tri: &[f32],
        ep: usize,
        ef: usize,
        level: usize,
    ) -> BinaryHeap<Neighbor> {
        let mut visited = std::collections::HashSet::new();
        let mut candidates = BinaryHeap::new();
        let d_ep = self.calculate_distance_lut(lut_base, lut_epi, lut_tri, ep);
        let ep_neigh = Neighbor {
            idx: ep,
            distance: d_ep,
        };
        visited.insert(ep);
        candidates.push(ep_neigh);
        let mut found_neighbors = BinaryHeap::new();
        found_neighbors.push(std::cmp::Reverse(ep_neigh));
        while let Some(c) = candidates.pop() {
            let furthest_dist = found_neighbors
                .peek()
                .map(|f| f.0.distance)
                .unwrap_or(f32::MAX);
            if c.distance > furthest_dist && found_neighbors.len() >= ef {
                break;
            }
            for &neighbor_idx in &self.layers[level][c.idx] {
                if !visited.contains(&neighbor_idx) {
                    visited.insert(neighbor_idx);
                    let d = self.calculate_distance_lut(lut_base, lut_epi, lut_tri, neighbor_idx);
                    let n = Neighbor {
                        idx: neighbor_idx,
                        distance: d,
                    };
                    if d < furthest_dist || found_neighbors.len() < ef {
                        candidates.push(n);
                        found_neighbors.push(std::cmp::Reverse(n));
                        if found_neighbors.len() > ef {
                            found_neighbors.pop();
                        }
                    }
                }
            }
        }
        found_neighbors.into_iter().map(|r| r.0).collect()
    }
    fn insert_node(&mut self, idx: usize) {
        let mut rng = rand::thread_rng();
        let level = (-rng.gen::<f64>().ln() * self.level_mult) as i32;
        let n = if self.stride == 0 {
            0
        } else {
            self.database.len() / self.stride
        };
        if self.entry_point.is_none() {
            self.max_level = level;
            self.entry_point = Some(idx);
            self.layers = vec![vec![vec![]; n]; (level + 1) as usize];
            return;
        }

        let strand_start = idx * self.stride;
        let q = dequantize_embedding(
            self.database[strand_start..strand_start + self.stride].to_vec(),
            self.stride * 4,
            Some(self.centroids.clone()),
        )
        .unwrap();
        let mut lut_base = Vec::with_capacity(q.len() * 4);
        let c = &self.centroids;
        let is_multi = c.len() == q.len() * 4;
        for (d_idx, &val) in q.iter().enumerate() {
            for b in 0..4 {
                let diff = val - (if is_multi { c[d_idx * 4 + b] } else { c[b] });
                lut_base.push(diff * diff);
            }
        }
        let mut lut_epi = Vec::new();
        if !self.epigenetic_centroids.is_empty() {
            lut_epi.reserve(q.len() * 16);
            let _c_base = &self.centroids;
            let _c_epi = &self.epigenetic_centroids;
            for (d_idx, &val) in q.iter().enumerate() {
                let cb = if is_multi {
                    &self.centroids[d_idx * 4..(d_idx + 1) * 4]
                } else {
                    &self.centroids[0..4]
                };
                let ce = if self.epigenetic_centroids.len() == q.len() * 4 {
                    &self.epigenetic_centroids[d_idx * 4..(d_idx + 1) * 4]
                } else {
                    &self.epigenetic_centroids[0..4]
                };
                for &b_val in cb {
                    for &e_val in ce {
                        let diff = val - (b_val + e_val);
                        lut_epi.push(diff * diff);
                    }
                }
            }
        }
        let mut lut_tri = Vec::new();
        if !self.triplet_centroids.is_empty() {
            lut_tri.reserve(q.len() * 64);
            let c_tri = &self.triplet_centroids;
            for (d_idx, &val) in q.iter().enumerate() {
                let cb = if is_multi {
                    &self.centroids[d_idx * 4..(d_idx + 1) * 4]
                } else {
                    &self.centroids[0..4]
                };
                let ce = if !self.epigenetic_centroids.is_empty() {
                    if self.epigenetic_centroids.len() == q.len() * 4 {
                        &self.epigenetic_centroids[d_idx * 4..(d_idx + 1) * 4]
                    } else {
                        &self.epigenetic_centroids[0..4]
                    }
                } else {
                    &[0.0; 4]
                };
                let ct = if c_tri.len() == q.len() * 4 {
                    &c_tri[d_idx * 4..(d_idx + 1) * 4]
                } else {
                    &c_tri[0..4]
                };
                for &b_val in cb {
                    for &e_val in ce {
                        for &t_val in ct {
                            let diff = val - (b_val + e_val + t_val);
                            lut_tri.push(diff * diff);
                        }
                    }
                }
            }
        }
        let mut curr_ep = self.entry_point.unwrap();
        for l in (level + 1..=self.max_level).rev() {
            let res = self.search_layer_lut(&lut_base, &lut_epi, &lut_tri, curr_ep, 1, l as usize);
            if let Some(best) = res.peek() {
                curr_ep = best.idx;
            }
        }
        for l in (0..=std::cmp::min(level, self.max_level)).rev() {
            let neighbors = self.search_layer_lut(
                &lut_base,
                &lut_epi,
                &lut_tri,
                curr_ep,
                self.ef_construction,
                l as usize,
            );
            let mut neighbors_vec: Vec<Neighbor> = neighbors.into_vec();
            neighbors_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            let m_limit = if l == 0 { self.m * 2 } else { self.m };
            for n in neighbors_vec.iter().take(m_limit) {
                if !self.layers[l as usize][idx].contains(&n.idx) {
                    self.layers[l as usize][idx].push(n.idx);
                }
                if !self.layers[l as usize][n.idx].contains(&idx) {
                    self.layers[l as usize][n.idx].push(idx);
                }
            }
            if let Some(best) = neighbors_vec.first() {
                curr_ep = best.idx;
            }
        }
        if level > self.max_level {
            for _ in self.max_level..level {
                self.layers.push(vec![vec![]; n]);
            }
            self.max_level = level;
            self.entry_point = Some(idx);
        }
    }
}
#[pymethods]
impl GajeIndex {
    #[new]
    #[pyo3(signature = (database, centroids, epigenetic_database=Vec::new(), epigenetic_centroids=Vec::new(), triplet_database=Vec::new(), triplet_centroids=Vec::new(), precision_mask=Vec::new(), m=32, ef_construction=200))]
    pub fn new(
        database: Vec<Vec<u8>>,
        centroids: Vec<f32>,
        epigenetic_database: Vec<Vec<u8>>,
        epigenetic_centroids: Vec<f32>,
        triplet_database: Vec<Vec<u8>>,
        triplet_centroids: Vec<f32>,
        precision_mask: Vec<u8>,
        m: usize,
        ef_construction: usize,
    ) -> Self {
        let stride = if database.is_empty() {
            0
        } else {
            database[0].len()
        };
        let mut flat_db = Vec::with_capacity(database.len() * stride);
        for s in database {
            flat_db.extend(s);
        }
        let mut flat_epi_db = Vec::with_capacity(epigenetic_database.len() * stride);
        for s in epigenetic_database {
            flat_epi_db.extend(s);
        }
        let mut flat_tri_db = Vec::with_capacity(triplet_database.len() * stride);
        for s in triplet_database {
            flat_tri_db.extend(s);
        }
        GajeIndex {
            database: flat_db,
            centroids,
            epigenetic_database: flat_epi_db,
            epigenetic_centroids,
            triplet_database: flat_tri_db,
            triplet_centroids,
            precision_mask,
            stride,
            layers: Vec::new(),
            max_level: -1,
            entry_point: None,
            ef_construction,
            m,
            level_mult: 1.0 / (m as f64).ln(),
        }
    }
    pub fn build(&mut self) -> PyResult<()> {
        let n = if self.stride == 0 {
            0
        } else {
            self.database.len() / self.stride
        };
        println!(
            "[*] Building Optimized DNA Graph (N={}, M={}, ef_c={})...",
            n, self.m, self.ef_construction
        );
        self.layers = vec![vec![vec![]; n]; 1];
        for i in 0..n {
            self.insert_node(i);
        }
        Ok(())
    }
    pub fn add_batch(&mut self, strands: Vec<Vec<u8>>) -> PyResult<()> {
        if self.stride == 0 && !strands.is_empty() {
            self.stride = strands[0].len();
        }
        for s in strands {
            self.database.extend(s);
        }
        Ok(())
    }
    pub fn flat_search(&self, query_vector: Vec<f32>, k: usize) -> PyResult<Vec<(usize, f32)>> {
        let c = &self.centroids;
        let q_len = query_vector.len();
        let n = if self.stride == 0 {
            0
        } else {
            self.database.len() / self.stride
        };
        let mut results: Vec<(usize, f32)> = (0..n)
            .into_par_iter()
            .map(|idx| {
                let strand_start = idx * self.stride;
                let strand = &self.database[strand_start..strand_start + self.stride];
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
                        let c_idx = (bits ^ (bits >> 1)) as usize;
                        let centroid = if is_multi {
                            c[dims * 4 + c_idx]
                        } else {
                            c[c_idx]
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
    #[pyo3(signature = (query_vector, k=10, ef=None))]
    pub fn search(
        &self,
        query_vector: Vec<f32>,
        k: usize,
        ef: Option<usize>,
    ) -> PyResult<Vec<(usize, f32)>> {
        if self.entry_point.is_none() {
            return self.flat_search(query_vector, k);
        }
        let ef_val = ef.unwrap_or(std::cmp::max(k, 50));
        let c_base = &self.centroids;
        let is_multi = c_base.len() == query_vector.len() * 4;
        let mut lut_base = Vec::with_capacity(query_vector.len() * 4);
        for (d_idx, &val) in query_vector.iter().enumerate() {
            let cd = if is_multi {
                &c_base[d_idx * 4..(d_idx + 1) * 4]
            } else {
                &c_base[0..4]
            };
            for b in 0..4 {
                let diff = val - cd[b];
                lut_base.push(diff * diff);
            }
        }
        let mut lut_epi = Vec::new();
        if !self.epigenetic_centroids.is_empty() {
            lut_epi.reserve(query_vector.len() * 16);
            let _c_epi = &self.epigenetic_centroids;
            for (d_idx, &val) in query_vector.iter().enumerate() {
                let cb = if is_multi {
                    &self.centroids[d_idx * 4..(d_idx + 1) * 4]
                } else {
                    &self.centroids[0..4]
                };
                let ce = if self.epigenetic_centroids.len() == query_vector.len() * 4 {
                    &self.epigenetic_centroids[d_idx * 4..(d_idx + 1) * 4]
                } else {
                    &self.epigenetic_centroids[0..4]
                };
                for &b_val in cb {
                    for &e_val in ce {
                        let diff = val - (b_val + e_val);
                        lut_epi.push(diff * diff);
                    }
                }
            }
        }
        let mut lut_tri = Vec::new();
        if !self.triplet_centroids.is_empty() {
            lut_tri.reserve(query_vector.len() * 64);
            let c_tri = &self.triplet_centroids;
            for (d_idx, &val) in query_vector.iter().enumerate() {
                let cb = if is_multi {
                    &self.centroids[d_idx * 4..(d_idx + 1) * 4]
                } else {
                    &self.centroids[0..4]
                };
                let ce = if !self.epigenetic_centroids.is_empty() {
                    if self.epigenetic_centroids.len() == query_vector.len() * 4 {
                        &self.epigenetic_centroids[d_idx * 4..(d_idx + 1) * 4]
                    } else {
                        &self.epigenetic_centroids[0..4]
                    }
                } else {
                    &[0.0; 4]
                };
                let ct = if c_tri.len() == query_vector.len() * 4 {
                    &c_tri[d_idx * 4..(d_idx + 1) * 4]
                } else {
                    &c_tri[0..4]
                };
                for &b_val in cb {
                    for &e_val in ce {
                        for &t_val in ct {
                            let diff = val - (b_val + e_val + t_val);
                            lut_tri.push(diff * diff);
                        }
                    }
                }
            }
        }
        let mut curr_ep = self.entry_point.unwrap();
        for l in (1..=self.max_level).rev() {
            let res = self.search_layer_lut(&lut_base, &lut_epi, &lut_tri, curr_ep, 1, l as usize);
            if let Some(best) = res.peek() {
                curr_ep = best.idx;
            }
        }
        let final_neighbors =
            self.search_layer_lut(&lut_base, &lut_epi, &lut_tri, curr_ep, ef_val, 0);
        let mut results: Vec<(usize, f32)> = final_neighbors
            .into_iter()
            .map(|n| (n.idx, n.distance))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        if k > 0 && k < results.len() {
            results.truncate(k);
        }
        Ok(results)
    }
}
