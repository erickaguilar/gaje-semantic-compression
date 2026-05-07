use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use rand::Rng;

#[derive(Clone, Copy)]
struct Neighbor {
    idx: usize,
    distance: f32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
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
    pub database: Vec<Vec<u8>>,
    #[pyo3(get)]
    pub centroids: Vec<f32>,
    layers: Vec<Vec<Vec<usize>>>,
    max_level: i32,
    entry_point: Option<usize>,
    ef_construction: usize,
    m: usize,
    level_mult: f64,
}

impl GajeIndex {
    fn calculate_distance_lut(&self, lut: &[f32], target_idx: usize) -> f32 {
        let strand = &self.database[target_idx];
        let mut dist_sq = 0.0f32;
        let mut dims = 0;
        let lut_len = lut.len() / 4;

        for &byte in strand {
            for j in 0..4 {
                if dims >= lut_len { break; }
                let shift = (3 - j) * 2;
                let bits = (byte >> shift) & 0b11;
                let lut_idx = match bits {
                    0b00 => 0, 0b01 => 1, 0b11 => 2, 0b10 => 3, _ => 0,
                };
                dist_sq += lut[dims * 4 + lut_idx];
                dims += 1;
            }
        }
        dist_sq.sqrt()
    }

    fn search_layer_lut(&self, lut: &[f32], ep: usize, ef: usize, level: usize) -> BinaryHeap<Neighbor> {
        let mut visited = std::collections::HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut found_neighbors = BinaryHeap::new();

        let d_ep = self.calculate_distance_lut(lut, ep);
        let ep_neigh = Neighbor { idx: ep, distance: d_ep };
        
        visited.insert(ep);
        candidates.push(ep_neigh);
        found_neighbors.push(ep_neigh);

        while let Some(c) = candidates.pop() {
            let furthest_dist = found_neighbors.peek().map(|f| f.distance).unwrap_or(f32::MAX);
            if c.distance > furthest_dist && found_neighbors.len() >= ef {
                break;
            }

            for &neighbor_idx in &self.layers[level][c.idx] {
                if !visited.contains(&neighbor_idx) {
                    visited.insert(neighbor_idx);
                    let d = self.calculate_distance_lut(lut, neighbor_idx);
                    let n = Neighbor { idx: neighbor_idx, distance: d };
                    
                    if d < furthest_dist || found_neighbors.len() < ef {
                        candidates.push(n);
                        found_neighbors.push(n);
                        if found_neighbors.len() > ef {
                            found_neighbors.pop();
                        }
                    }
                }
            }
        }
        found_neighbors
    }
}

#[pymethods]
impl GajeIndex {
    #[new]
    #[pyo3(signature = (database, centroids, m=32, ef_construction=200))]
    pub fn new(database: Vec<Vec<u8>>, centroids: Vec<f32>, m: usize, ef_construction: usize) -> Self {
        GajeIndex {
            database,
            centroids,
            layers: Vec::new(),
            max_level: -1,
            entry_point: None,
            ef_construction,
            m,
            level_mult: 1.0 / (m as f64).ln(),
        }
    }

    pub fn build(&mut self) -> PyResult<()> {
        println!("[*] Building Optimized DNA Graph (M={}, ef_c={})...", self.m, self.ef_construction);
        let n = self.database.len();
        self.layers = vec![vec![vec![]; n]; 1]; 
        for i in 0..n {
            self.insert_node(i);
            if i % 1000 == 0 && i > 0 { println!("[*] Indexed {}/{} strands...", i, n); }
        }
        Ok(())
    }

    fn insert_node(&mut self, idx: usize) {
        let mut rng = rand::thread_rng();
        let level = (-rng.gen::<f64>().ln() * self.level_mult) as i32;
        
        if self.entry_point.is_none() {
            self.max_level = level;
            self.entry_point = Some(idx);
            self.layers = vec![vec![vec![]; self.database.len()]; (level + 1) as usize];
            return;
        }

        let q = dequantize_embedding(self.database[idx].clone(), 768, Some(self.centroids.clone())).unwrap();
        let mut lut = Vec::with_capacity(q.len() * 4);
        let c = &self.centroids;
        let is_multi = c.len() == q.len() * 4;
        for (d_idx, &val) in q.iter().enumerate() {
            for b in 0..4 {
                let centroid = if is_multi {
                    let b_idx = d_idx * 4;
                    match b { 0 => c[b_idx], 1 => c[b_idx+1], 2 => c[b_idx+2], 3 => c[b_idx+3], _ => 0.0 }
                } else {
                    match b { 0 => c[0], 1 => c[1], 2 => c[2], 3 => c[3], _ => 0.0 }
                };
                let diff = val - centroid;
                lut.push(diff * diff);
            }
        }

        let mut curr_ep = self.entry_point.unwrap();
        for l in (level + 1..=self.max_level).rev() {
            let res = self.search_layer_lut(&lut, curr_ep, 1, l as usize);
            if let Some(best) = res.peek() { curr_ep = best.idx; }
        }

        for l in (0..=std::cmp::min(level, self.max_level)).rev() {
            let neighbors = self.search_layer_lut(&lut, curr_ep, self.ef_construction, l as usize);
            let mut neighbors_vec: Vec<Neighbor> = neighbors.into_vec();
            neighbors_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            
            let m_limit = if l == 0 { self.m * 2 } else { self.m };
            for n in neighbors_vec.iter().take(m_limit) {
                if !self.layers[l as usize][idx].contains(&n.idx) { self.layers[l as usize][idx].push(n.idx); }
                if !self.layers[l as usize][n.idx].contains(&idx) { self.layers[l as usize][n.idx].push(idx); }
            }
            if let Some(best) = neighbors_vec.first() { curr_ep = best.idx; }
        }

        if level > self.max_level {
            for _ in self.max_level..level { self.layers.push(vec![vec![]; self.database.len()]); }
            self.max_level = level;
            self.entry_point = Some(idx);
        }
    }

    pub fn search(&self, query_vector: Vec<f32>, ef: usize) -> PyResult<Vec<(usize, f32)>> {
        if self.entry_point.is_none() {
            return dna_similarity_search_adc(query_vector, self.database.clone(), Some(self.centroids.clone()));
        }

        let mut lut = Vec::with_capacity(query_vector.len() * 4);
        let c = &self.centroids;
        let is_multi = c.len() == query_vector.len() * 4;
        for (d_idx, &val) in query_vector.iter().enumerate() {
            for b in 0..4 {
                let centroid = if is_multi {
                    let b_idx = d_idx * 4;
                    match b { 0 => c[b_idx], 1 => c[b_idx+1], 2 => c[b_idx+2], 3 => c[b_idx+3], _ => 0.0 }
                } else {
                    match b { 0 => c[0], 1 => c[1], 2 => c[2], 3 => c[3], _ => 0.0 }
                };
                let diff = val - centroid;
                lut.push(diff * diff);
            }
        }

        let mut curr_ep = self.entry_point.unwrap();
        for l in (1..=self.max_level).rev() {
            let res = self.search_layer_lut(&lut, curr_ep, 1, l as usize);
            if let Some(best) = res.peek() { curr_ep = best.idx; }
        }

        let final_neighbors = self.search_layer_lut(&lut, curr_ep, ef, 0);
        let mut results: Vec<(usize, f32)> = final_neighbors.into_iter().map(|n| (n.idx, n.distance)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(results)
    }
}

#[pyfunction]
#[pyo3(signature = (vector, thresholds=None))]
pub fn quantize_embedding(vector: Vec<f32>, thresholds: Option<Vec<f32>>) -> PyResult<Vec<u8>> {
    let t = thresholds.unwrap_or_else(|| vec![-0.34, 0.0, 0.34]);
    let sub_size = 4;
    let mut packed = Vec::with_capacity(vector.len() / sub_size);
    let is_multi = t.len() == vector.len() * 3;
    for i in 0..(vector.len() / sub_size) {
        let mut byte = 0u8;
        for j in 0..4 {
            let idx = i * 4 + j;
            let val = vector[idx];
            let (t0, t1, t2) = if is_multi { (t[idx*3], t[idx*3+1], t[idx*3+2]) } else { (t[0], t[1], t[2]) };
            let bits = if val < t0 { 0b00 } else if val < t1 { 0b01 } else if val < t2 { 0b11 } else { 0b10 };
            byte = (byte << 2) | bits;
        }
        packed.push(byte);
    }
    Ok(packed)
}

#[pyfunction]
#[pyo3(signature = (vector, thresholds=None))]
pub fn quantize_pq(vector: Vec<f32>, thresholds: Option<Vec<f32>>) -> PyResult<Vec<u8>> {
    quantize_embedding(vector, thresholds)
}

#[pyfunction]
#[pyo3(signature = (query_vector, database, centroids=None))]
pub fn dna_similarity_search_adc(query_vector: Vec<f32>, database: Vec<Vec<u8>>, centroids: Option<Vec<f32>>) -> PyResult<Vec<(usize, f32)>> {
    let c = centroids.unwrap_or_else(|| vec![-0.68, -0.17, 0.17, 0.68]);
    let q_len = query_vector.len();
    let mut results: Vec<(usize, f32)> = database.par_iter().enumerate().map(|(idx, strand)| {
        let mut dist_sq = 0.0f32;
        let mut dims = 0;
        let is_multi = c.len() == q_len * 4;
        for &byte in strand {
            for j in 0..4 {
                if dims >= q_len { break; }
                let shift = (3 - j) * 2;
                let bits = (byte >> shift) & 0b11;
                let centroid = if is_multi {
                    let b = dims * 4; match bits { 0b00=>c[b], 0b01=>c[b+1], 0b11=>c[b+2], 0b10=>c[b+3], _=>0.0 }
                } else {
                    match bits { 0b00=>c[0], 0b01=>c[1], 0b11=>c[2], 0b10=>c[3], _=>0.0 }
                };
                let diff = query_vector[dims] - centroid;
                dist_sq += diff * diff;
                dims += 1;
            }
        }
        (idx, dist_sq.sqrt())
    }).collect();
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(results)
}

#[pyfunction]
#[pyo3(signature = (query, database, centroids=None))]
pub fn dna_similarity_search(query: PyObject, database: Vec<Vec<u8>>, centroids: Option<Vec<f32>>, py: Python<'_>) -> PyResult<Vec<(usize, f32)>> {
    if let Ok(qv) = query.extract::<Vec<f32>>(py) { return dna_similarity_search_adc(qv, database, centroids); }
    if let Ok(qd) = query.extract::<Vec<u8>>(py) {
        let c = centroids.unwrap_or_else(|| vec![-0.68, -0.17, 0.17, 0.68]);
        let mut res: Vec<(usize, f32)> = database.par_iter().enumerate().map(|(idx, strand)| {
            let mut d = 0.0f32;
            for i in 0..std::cmp::min(qd.len(), strand.len()) {
                let (b1, b2) = (qd[i], strand[i]);
                for j in 0..4 {
                    let s = (3 - j) * 2;
                    let (v1b, v2b) = ((b1 >> s) & 0b11, (b2 >> s) & 0b11);
                    let v1 = match v1b { 0b00=>c[0], 0b01=>c[1], 0b11=>c[2], 0b10=>c[3], _=>0.0 };
                    let v2 = match v2b { 0b00=>c[0], 0b01=>c[1], 0b11=>c[2], 0b10=>c[3], _=>0.0 };
                    let diff = v1 - v2; d += diff * diff;
                }
            }
            (idx, d.sqrt())
        }).collect();
        res.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        return Ok(res);
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Query error"))
}

#[pyfunction]
#[pyo3(signature = (dna_packed, dims, centroids=None))]
pub fn dequantize_embedding(dna_packed: Vec<u8>, dims: usize, centroids: Option<Vec<f32>>) -> PyResult<Vec<f32>> {
    let c = centroids.unwrap_or_else(|| vec![-0.68, -0.17, 0.17, 0.68]);
    let mut rec = Vec::with_capacity(dims);
    let mut dp = 0;
    let is_multi = c.len() == dims * 4;
    for &byte in &dna_packed {
        for j in 0..4 {
            if dp >= dims { break; }
            let s = (3 - j) * 2;
            let bits = (byte >> s) & 0b11;
            let cent = if is_multi {
                let b = dp * 4; match bits { 0b00=>c[b], 0b01=>c[b+1], 0b11=>c[b+2], 0b10=>c[b+3], _=>0.0 }
            } else {
                match bits { 0b00=>c[0], 0b01=>c[1], 0b11=>c[2], 0b10=>c[3], _=>0.0 }
            };
            rec.push(cent); dp += 1;
        }
    }
    Ok(rec)
}

#[pymodule]
fn dna_semantic_compression(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GajeIndex>()?;
    m.add_function(wrap_pyfunction!(quantize_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_pq, m)?)?;
    m.add_function(wrap_pyfunction!(dna_similarity_search_adc, m)?)?;
    m.add_function(wrap_pyfunction!(dna_similarity_search, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_embedding, m)?)?;
    Ok(())
}
