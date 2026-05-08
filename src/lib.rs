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
    pub database: Vec<u8>,
    #[pyo3(get)]
    pub centroids: Vec<f32>,
    #[pyo3(get)]
    pub stride: usize,
    layers: Vec<Vec<Vec<usize>>>,
    max_level: i32,
    entry_point: Option<usize>,
    ef_construction: usize,
    m: usize,
    level_mult: f64,
}

impl GajeIndex {
    fn get_strand(&self, idx: usize) -> &[u8] {
        let start = idx * self.stride;
        &self.database[start..start + self.stride]
    }

    fn calculate_distance_lut(&self, lut: &[f32], target_idx: usize) -> f32 {
        let strand = self.get_strand(target_idx);
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
        // found_neighbors debe ser un MAX-HEAP de distancias para poder sacar el más lejano.
        // Como Neighbor es un MIN-HEAP (pop el menor), usamos un heap de distancias invertidas o simplemente cambiamos la lógica.
        // Pero para mantener la consistencia con HNSW, necesitamos sacar el más lejano cuando excedemos ef.
        
        let d_ep = self.calculate_distance_lut(lut, ep);
        let ep_neigh = Neighbor { idx: ep, distance: d_ep };
        
        visited.insert(ep);
        candidates.push(ep_neigh);
        
        // Usaremos un BinaryHeap de Neighbor normal para candidatos (pop el más cercano).
        // Y para found_neighbors, usaremos un vector y lo mantendremos como un Max-Heap manualmente o usaremos otra técnica.
        // En Rust, la forma más fácil es usar un struct diferente.
        
        let mut found_neighbors = BinaryHeap::new(); // Este será un MIN-HEAP por defecto con Neighbor
        found_neighbors.push(std::cmp::Reverse(ep_neigh)); // Reverse lo convierte en Max-Heap

        while let Some(c) = candidates.pop() {
            let furthest_dist = found_neighbors.peek().map(|f| f.0.distance).unwrap_or(f32::MAX);
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
                        found_neighbors.push(std::cmp::Reverse(n));
                        if found_neighbors.len() > ef {
                            found_neighbors.pop();
                        }
                    }
                }
            }
        }
        // Devolver un heap de Neighbor normal
        found_neighbors.into_iter().map(|r| r.0).collect()
    }
}

#[pymethods]
impl GajeIndex {
    #[new]
    #[pyo3(signature = (database, centroids, m=32, ef_construction=200))]
    pub fn new(database: Vec<Vec<u8>>, centroids: Vec<f32>, m: usize, ef_construction: usize) -> Self {
        let stride = if database.is_empty() { 0 } else { database[0].len() };
        let mut flat_db = Vec::with_capacity(database.len() * stride);
        for s in database {
            flat_db.extend(s);
        }
        GajeIndex {
            database: flat_db,
            centroids,
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
        let n = if self.stride == 0 { 0 } else { self.database.len() / self.stride };
        println!("[*] Building Optimized DNA Graph (N={}, M={}, ef_c={})...", n, self.m, self.ef_construction);
        self.layers = vec![vec![vec![]; n]; 1]; 
        for i in 0..n {
            self.insert_node(i);
            if i % 1000 == 0 && i > 0 { println!("[*] Indexed {}/{} strands...", i, n); }
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
        let n = if self.stride == 0 { 0 } else { self.database.len() / self.stride };
        
        let mut results: Vec<(usize, f32)> = (0..n).into_par_iter().map(|idx| {
            let strand = self.get_strand(idx);
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
        
        if k > 0 && k < results.len() {
            results.truncate(k);
        }
        
        Ok(results)
    }

    fn insert_node(&mut self, idx: usize) {
        let mut rng = rand::thread_rng();
        let level = (-rng.gen::<f64>().ln() * self.level_mult) as i32;
        let n = if self.stride == 0 { 0 } else { self.database.len() / self.stride };
        
        if self.entry_point.is_none() {
            self.max_level = level;
            self.entry_point = Some(idx);
            self.layers = vec![vec![vec![]; n]; (level + 1) as usize];
            return;
        }

        let q = dequantize_embedding(self.get_strand(idx).to_vec(), self.stride * 4, Some(self.centroids.clone())).unwrap();
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
            for _ in self.max_level..level { self.layers.push(vec![vec![]; n]); }
            self.max_level = level;
            self.entry_point = Some(idx);
        }
    }

    #[pyo3(signature = (query_vector, k=10, ef=None))]
    pub fn search(&self, query_vector: Vec<f32>, k: usize, ef: Option<usize>) -> PyResult<Vec<(usize, f32)>> {
        if self.entry_point.is_none() {
            return self.flat_search(query_vector, k);
        }

        let ef_val = ef.unwrap_or(std::cmp::max(k, 50));
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

        let final_neighbors = self.search_layer_lut(&lut, curr_ep, ef_val, 0);
        let mut results: Vec<(usize, f32)> = final_neighbors.into_iter().map(|n| (n.idx, n.distance)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        if k > 0 && k < results.len() {
            results.truncate(k);
        }
        Ok(results)
    }

    #[pyo3(signature = (input_vector))]
    pub fn genomic_linear_forward(&self, input_vector: Vec<f32>) -> PyResult<Vec<f32>> {
        let q_len = input_vector.len();
        let c_all = &self.centroids;
        let is_multi = c_all.len() > 4;
        
        // 2. Perform Forward Pass (Parallel MatMul over 2-bit weights)
        let n_neurons = self.database.len() / self.stride;
        let results: Vec<f32> = (0..n_neurons).into_par_iter().map(|neuron_idx| {
            let weights = self.get_strand(neuron_idx);
            let c = if is_multi {
                let offset = neuron_idx * 4;
                &c_all[offset..offset + 4]
            } else {
                &c_all[0..4]
            };

            let mut sum = 0.0f32;
            let mut dims = 0;
            for &byte in weights {
                for j in 0..4 {
                    if dims >= q_len { break; }
                    let shift = (3 - j) * 2;
                    let bits = (byte >> shift) & 0b11;
                    let val = match bits {
                        0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0
                    };
                    sum += input_vector[dims] * val;
                    dims += 1;
                }
            }
            sum
        }).collect();

        Ok(results)
    }
}

#[pyfunction]
#[pyo3(signature = (vector, thresholds=None))]
pub fn quantize_embedding(vector: Vec<f32>, thresholds: Option<Vec<f32>>, py: Python<'_>) -> PyResult<PyObject> {
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
    Ok(pyo3::types::PyBytes::new_bound(py, &packed).into())
}

#[pyfunction]
#[pyo3(signature = (vector, thresholds=None))]
pub fn quantize_pq(vector: Vec<f32>, thresholds: Option<Vec<f32>>, py: Python<'_>) -> PyResult<PyObject> {
    quantize_embedding(vector, thresholds, py)
}

#[pyfunction]
#[pyo3(signature = (query_vector, database, centroids=None, k=10))]
pub fn dna_similarity_search_adc(query_vector: Vec<f32>, database: Vec<Vec<u8>>, centroids: Option<Vec<f32>>, k: usize) -> PyResult<Vec<(usize, f32)>> {
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
    if k > 0 && k < results.len() {
        results.truncate(k);
    }
    Ok(results)
}

#[pyfunction]
#[pyo3(signature = (query, database, centroids=None, k=10))]
pub fn dna_similarity_search(query: PyObject, database: Vec<Vec<u8>>, centroids: Option<Vec<f32>>, k: usize, py: Python<'_>) -> PyResult<Vec<(usize, f32)>> {
    if let Ok(qv) = query.extract::<Vec<f32>>(py) { return dna_similarity_search_adc(qv, database, centroids, k); }
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
        if k > 0 && k < res.len() {
            res.truncate(k);
        }
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

#[pyclass]
pub struct GenomicAttention {
    pub w_q: Vec<u8>,
    pub w_k: Vec<u8>,
    pub w_v: Vec<u8>,
    pub centroids: Vec<f32>, // Ahora puede ser un array plano de [n_neurons * 4]
    pub stride: usize,
    pub n_heads_q: usize,
    pub n_heads_kv: usize,
    pub head_dim: usize,
    pub k_cache: Vec<Vec<f32>>,
    pub v_cache: Vec<Vec<f32>>,
}

#[pymethods]
impl GenomicAttention {
    #[new]
    pub fn new(w_q: Vec<u8>, w_k: Vec<u8>, w_v: Vec<u8>, centroids: Vec<f32>, stride: usize, n_heads_q: usize, n_heads_kv: usize) -> Self {
        let head_dim = (w_q.len() / stride) / n_heads_q;
        
        GenomicAttention {
            w_q, w_k, w_v, centroids, stride, n_heads_q, n_heads_kv, head_dim,
            k_cache: Vec::new(),
            v_cache: Vec::new(),
        }
    }

    pub fn forward(&mut self, input_vector: Vec<f32>, pos: usize) -> PyResult<Vec<f32>> {
        let q_len = input_vector.len();
        let c_all = &self.centroids;
        let is_multi = c_all.len() > 4; // Check if we have per-neuron centroids
        
        // 1. Projection function con soporte para Multi-Centroids (Block-Quant)
        let project = |weights: &[u8], n_outputs: usize, c_offset_base: usize| -> Vec<f32> {
            (0..n_outputs).into_par_iter().map(|idx| {
                let start = idx * self.stride;
                let neuron_weights = &weights[start..start + self.stride];
                
                // Si es multi-centroide, cada neurona tiene sus propios 4 centroides
                let c = if is_multi {
                    let offset = (c_offset_base + idx) * 4;
                    &c_all[offset..offset + 4]
                } else {
                    &c_all[0..4]
                };

                let mut sum = 0.0f32;
                let mut dims = 0;
                for &byte in neuron_weights {
                    for j in 0..4 {
                        if dims >= q_len { break; }
                        let shift = (3 - j) * 2;
                        let bits = (byte >> shift) & 0b11;
                        let val = match bits {
                            0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0
                        };
                        sum += input_vector[dims] * val;
                        dims += 1;
                    }
                }
                sum
            }).collect()
        };

        // Calculamos offsets para los centroides de Q, K, V en el array plano
        let q_rows = self.n_heads_q * self.head_dim;
        let k_rows = self.n_heads_kv * self.head_dim;
        
        let mut q = project(&self.w_q, q_rows, 0);
        let mut k = project(&self.w_k, k_rows, q_rows);
        let v = project(&self.w_v, k_rows, q_rows + k_rows);

        // 2. Apply RoPE (Rotary Positional Embeddings)
        let apply_rope = |vec: &mut [f32], n_heads: usize, head_dim: usize, p: usize| {
            for h in 0..n_heads {
                let h_start = h * head_dim;
                for i in 0..(head_dim / 2) {
                    let theta = (p as f32) / (10000.0f32.powf((2 * i) as f32 / head_dim as f32));
                    let cos = theta.cos();
                    let sin = theta.sin();
                    
                    let v0 = vec[h_start + i];
                    let v1 = vec[h_start + i + head_dim / 2];
                    
                    vec[h_start + i] = v0 * cos - v1 * sin;
                    vec[h_start + i + head_dim / 2] = v0 * sin + v1 * cos;
                }
            }
        };

        apply_rope(&mut q, self.n_heads_q, self.head_dim, pos);
        apply_rope(&mut k, self.n_heads_kv, self.head_dim, pos);

        self.k_cache.push(k);
        self.v_cache.push(v);

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut output = vec![0.0f32; q.len()];
        let group_size = self.n_heads_q / self.n_heads_kv;

        for h_q in 0..self.n_heads_q {
            let h_kv = h_q / group_size;
            let q_start = h_q * self.head_dim;
            let kv_start = h_kv * self.head_dim;
            
            let q_head = &q[q_start..q_start + self.head_dim];
            
            let mut scores = Vec::with_capacity(self.k_cache.len());
            for k_full in &self.k_cache {
                let k_head = &k_full[kv_start..kv_start + self.head_dim];
                let score = q_head.iter().zip(k_head.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
                scores.push(score);
            }

            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            
            for (t, exp_s) in exp_scores.iter().enumerate() {
                let weight = exp_s / sum_exp;
                let v_head = &self.v_cache[t][kv_start..kv_start + self.head_dim];
                for i in 0..self.head_dim {
                    output[q_start + i] += weight * v_head[i];
                }
            }
        }

        Ok(output)
    }

    pub fn clear_cache(&mut self) {
        self.k_cache.clear();
        self.v_cache.clear();
    }
}

#[pymodule]
fn dna_semantic_compression(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GajeIndex>()?;
    m.add_class::<GenomicAttention>()?;
    m.add_function(wrap_pyfunction!(quantize_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_pq, m)?)?;
    m.add_function(wrap_pyfunction!(dna_similarity_search_adc, m)?)?;
    m.add_function(wrap_pyfunction!(dna_similarity_search, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_embedding, m)?)?;
    Ok(())
}
