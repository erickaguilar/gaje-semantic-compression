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
        // Min-heap behavior: we want smaller distances at the top
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
    // layers[level][node_idx] = Vec<Neighbor>
    layers: Vec<Vec<Vec<usize>>>,
    max_level: i32,
    entry_point: Option<usize>,
    ef_construction: usize,
    m: usize,
    level_mult: f64,
}

impl GajeIndex {
    fn calculate_distance(&self, q: &[f32], target_idx: usize) -> f32 {
        let strand = &self.database[target_idx];
        let c = &self.centroids;
        let is_multi = c.len() == q.len() * 4;
        let mut dist_sq = 0.0f32;
        let mut dims = 0;

        for &byte in strand {
            for j in 0..4 {
                if dims >= q.len() { break; }
                let shift = (3 - j) * 2;
                let bits = (byte >> shift) & 0b11;
                let centroid = if is_multi {
                    let base = dims * 4;
                    match bits { 0b00 => c[base], 0b01 => c[base+1], 0b11 => c[base+2], 0b10 => c[base+3], _ => 0.0 }
                } else {
                    match bits { 0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0 }
                };
                let diff = q[dims] - centroid;
                dist_sq += diff * diff;
                dims += 1;
            }
        }
        dist_sq.sqrt()
    }

    fn search_layer(&self, q: &[f32], ep: usize, ef: usize, level: usize) -> BinaryHeap<Neighbor> {
        let mut visited = std::collections::HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut found_neighbors = BinaryHeap::new();

        let d_ep = self.calculate_distance(q, ep);
        let ep_neigh = Neighbor { idx: ep, distance: d_ep };
        
        visited.insert(ep);
        candidates.push(ep_neigh);
        found_neighbors.push(ep_neigh);

        while let Some(c) = candidates.pop() {
            let furthest_dist = if let Some(f) = found_neighbors.peek() {
                f.distance
            } else {
                f32::MAX
            };

            if c.distance > furthest_dist && found_neighbors.len() >= ef {
                break;
            }

            for &neighbor_idx in &self.layers[level][c.idx] {
                if !visited.contains(&neighbor_idx) {
                    visited.insert(neighbor_idx);
                    let d = self.calculate_distance(q, neighbor_idx);
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
    #[pyo3(signature = (database, centroids, m=16, ef_construction=100))]
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
        println!("[*] Building Hierarchical DNA Graph...");
        let n = self.database.len();
        self.layers = vec![vec![vec![]; n]; 1]; // Start with base layer
        
        for i in 0..n {
            self.insert_node(i);
            if i % 1000 == 0 && i > 0 {
                println!("[*] Indexed {}/{} strands...", i, n);
            }
        }
        Ok(())
    }

    fn insert_node(&mut self, idx: usize) {
        let mut rng = rand::thread_rng();
        let level = ((-rng.gen::<f64>().ln() * self.level_mult) as i32);
        
        if self.layers.is_empty() || self.entry_point.is_none() {
            self.max_level = level;
            self.entry_point = Some(idx);
            for _ in 0..=level {
                self.layers.push(vec![vec![]; self.database.len() + 100]); 
            }
            return;
        }

        // 1. Find entry point for insertion level
        let mut curr_ep = self.entry_point.unwrap();
        let q = dequantize_embedding(self.database[idx].clone(), 768, Some(self.centroids.clone())).unwrap(); // Simplified dims
        
        for l in (level + 1..=self.max_level).rev() {
            let res = self.search_layer(&q, curr_ep, 1, l as usize);
            if let Some(best) = res.peek() {
                curr_ep = best.idx;
            }
        }

        // 2. Insert into layers
        for l in (0..=std::cmp::min(level, self.max_level)).rev() {
            let neighbors = self.search_layer(&q, curr_ep, self.ef_construction, l as usize);
            let mut neighbors_vec: Vec<Neighbor> = neighbors.into_vec();
            neighbors_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            
            // Connect
            let m_limit = if l == 0 { self.m * 2 } else { self.m };
            for n in neighbors_vec.iter().take(m_limit) {
                self.layers[l as usize][idx].push(n.idx);
                self.layers[l as usize][n.idx].push(idx);
            }
            if let Some(best) = neighbors_vec.first() {
                curr_ep = best.idx;
            }
        }

        if level > self.max_level {
            for _ in self.max_level..level {
                self.layers.push(vec![vec![]; self.database.len() + 1000]);
            }
            self.max_level = level;
            self.entry_point = Some(idx);
        }
    }

    pub fn search(&self, query_vector: Vec<f32>, ef: usize) -> PyResult<Vec<(usize, f32)>> {
        if self.entry_point.is_none() {
            return dna_similarity_search_adc(query_vector, self.database.clone(), Some(self.centroids.clone()));
        }

        let mut curr_ep = self.entry_point.unwrap();
        for l in (1..=self.max_level).rev() {
            let res = self.search_layer(&query_vector, curr_ep, 1, l as usize);
            if let Some(best) = res.peek() {
                curr_ep = best.idx;
            }
        }

        let final_neighbors = self.search_layer(&query_vector, curr_ep, ef, 0);
        let mut results: Vec<(usize, f32)> = final_neighbors.into_iter()
            .map(|n| (n.idx, n.distance))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(results)
    }
}

/// Funciones de utilidad (Mantenidas para compatibilidad)
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
            let (t0, t1, t2) = if is_multi { (t[dim_idx * 3], t[dim_idx * 3 + 1], t[dim_idx * 3 + 2]) } else { (t[0], t[1], t[2]) };
            let bits = match val { v if v < t0 => 0b00, v if v < t1 => 0b01, v if v < t2 => 0b11, _ => 0b10 };
            current_byte = (current_byte << 2) | bits;
        }
        packed.push(current_byte);
    }
    Ok(packed)
}

#[pyfunction]
pub fn quantize_pq(vector: Vec<f32>, thresholds: Option<Vec<f32>>) -> PyResult<Vec<u8>> {
    quantize_embedding(vector, thresholds)
}

#[pyfunction]
pub fn dna_similarity_search_adc(query_vector: Vec<f32>, database: Vec<Vec<u8>>, centroids: Option<Vec<f32>>) -> PyResult<Vec<(usize, f32)>> {
    let c = centroids.unwrap_or_else(|| vec![-0.68, -0.17, 0.17, 0.68]);
    let is_multi = c.len() == query_vector.len() * 4;
    let q_len = query_vector.len();
    let mut results: Vec<(usize, f32)> = database.par_iter().enumerate().map(|(idx, strand)| {
        let mut squared_distance = 0.0f32;
        let mut dims_processed = 0;
        for &byte in strand {
            for j in 0..4 {
                if dims_processed >= q_len { break; }
                let shift = (3 - j) * 2;
                let bits = (byte >> shift) & 0b11;
                let centroid = if is_multi {
                    let base_idx = dims_processed * 4;
                    match bits { 0b00 => c[base_idx], 0b01 => c[base_idx + 1], 0b11 => c[base_idx + 2], 0b10 => c[base_idx + 3], _ => 0.0 }
                } else {
                    match bits { 0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0 }
                };
                let diff = query_vector[dims_processed] - centroid;
                squared_distance += diff * diff;
                dims_processed += 1;
            }
        }
        (idx, squared_distance.sqrt())
    }).collect();
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(results)
}

#[pyfunction]
pub fn dna_similarity_search(query: PyObject, database: Vec<Vec<u8>>, centroids: Option<Vec<f32>>, py: Python<'_>) -> PyResult<Vec<(usize, f32)>> {
    if let Ok(query_vector) = query.extract::<Vec<f32>>(py) { return dna_similarity_search_adc(query_vector, database, centroids); }
    if let Ok(query_dna) = query.extract::<Vec<u8>>(py) {
        let c = centroids.unwrap_or_else(|| vec![-0.68, -0.17, 0.17, 0.68]);
        let mut results: Vec<(usize, f32)> = database.par_iter().enumerate().map(|(idx, strand)| {
            let mut dist = 0.0f32;
            for i in 0..std::cmp::min(query_dna.len(), strand.len()) {
                let b1 = query_dna[i]; let b2 = strand[i];
                for j in 0..4 {
                    let shift = (3 - j) * 2; let v1_bits = (b1 >> shift) & 0b11; let v2_bits = (b2 >> shift) & 0b11;
                    let v1 = match v1_bits { 0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0 };
                    let v2 = match v2_bits { 0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0 };
                    let diff = v1 - v2; dist += diff * diff;
                }
            }
            (idx, dist.sqrt())
        }).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        return Ok(results);
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Query debe ser Vec<f32> o Vec<u8>"))
}

#[pyfunction]
#[pyo3(signature = (dna_packed, dims, centroids=None))]
pub fn dequantize_embedding(dna_packed: Vec<u8>, dims: usize, centroids: Option<Vec<f32>>) -> PyResult<Vec<f32>> {
    let c = centroids.unwrap_or_else(|| vec![-0.68, -0.17, 0.17, 0.68]);
    let is_multi = c.len() == dims * 4;
    let mut reconstructed = Vec::with_capacity(dims);
    let mut dims_processed = 0;
    for &byte in &dna_packed {
        for j in 0..4 {
            if dims_processed >= dims { break; }
            let shift = (3 - j) * 2; let bits = (byte >> shift) & 0b11;
            let centroid = if is_multi {
                let base_idx = dims_processed * 4;
                match bits { 0b00 => c[base_idx], 0b01 => c[base_idx + 1], 0b11 => c[base_idx + 2], 0b10 => c[base_idx + 3], _ => 0.0 }
            } else {
                match bits { 0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0 }
            };
            reconstructed.push(centroid); dims_processed += 1;
        }
    }
    Ok(reconstructed)
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
