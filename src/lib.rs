use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use rand::Rng;
use half::f16;
use std::cmp::Ordering;

#[derive(Copy, Clone, PartialEq)]
struct Neighbor {
    idx: usize,
    distance: f32,
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

    fn get_epigenetic_strand(&self, idx: usize) -> &[u8] {
        let start = idx * self.stride;
        &self.epigenetic_database[start..start + self.stride]
    }

    fn get_triplet_strand(&self, idx: usize) -> &[u8] {
        let start = idx * self.stride;
        &self.triplet_database[start..start + self.stride]
    }

    fn calculate_distance_lut(&self, lut: &[f32], target_idx: usize) -> f32 {
        let strand = self.get_strand(target_idx);
        let has_epi = !self.epigenetic_database.is_empty();
        let has_triplet = !self.triplet_database.is_empty();
        
        let mode = if has_triplet { 64 } else if has_epi { 16 } else { 4 };
        let lut_len = lut.len() / mode;

        #[cfg(target_arch = "aarch64")]
        {
            let epi_strand = if has_epi { Some(self.get_epigenetic_strand(target_idx)) } else { None };
            let tri_strand = if has_triplet { Some(self.get_triplet_strand(target_idx)) } else { None };
            return unsafe { calculate_distance_lut_neon(lut, strand, epi_strand, tri_strand, lut_len) };
        }

        let mut dist_sq = 0.0f32;
        let mut dims = 0;

        if !has_epi && !has_triplet {
            for &byte in strand {
                for j in 0..4 {
                    if dims >= lut_len { break; }
                    let shift = (3 - j) * 2;
                    let bits = (byte >> shift) & 0b11;
                    let b_idx = (bits ^ (bits >> 1)) as usize;
                    dist_sq += lut[dims * 4 + b_idx];
                    dims += 1;
                }
            }
        } else if has_epi && !has_triplet {
            let epi_strand = self.get_epigenetic_strand(target_idx);
            for i in 0..self.stride {
                let byte = strand[i];
                let epi_byte = epi_strand[i];
                for j in 0..4 {
                    if dims >= lut_len { break; }
                    let shift = (3 - j) * 2;
                    let bits = (byte >> shift) & 0b11;
                    let epi_bits = (epi_byte >> shift) & 0b11;
                    let b_idx = bits ^ (bits >> 1);
                    let e_idx = epi_bits ^ (epi_bits >> 1);
                    let lut_idx = ((b_idx << 2) | e_idx) as usize;
                    dist_sq += lut[dims * 16 + lut_idx];
                    dims += 1;
                }
            }
        } else {
            let epi_strand = self.get_epigenetic_strand(target_idx);
            let tri_strand = self.get_triplet_strand(target_idx);
            for i in 0..self.stride {
                let byte = strand[i];
                let epi_byte = epi_strand[i];
                let tri_byte = tri_strand[i];
                for j in 0..4 {
                    if dims >= lut_len { break; }
                    let shift = (3 - j) * 2;
                    let bits = (byte >> shift) & 0b11;
                    let epi_bits = (epi_byte >> shift) & 0b11;
                    let tri_bits = (tri_byte >> shift) & 0b11;
                    let b_idx = bits ^ (bits >> 1);
                    let e_idx = epi_bits ^ (epi_bits >> 1);
                    let t_idx = tri_bits ^ (tri_bits >> 1);
                    let lut_idx = ((b_idx << 4) | (e_idx << 2) | t_idx) as usize;
                    dist_sq += lut[dims * 64 + lut_idx];
                    dims += 1;
                }
            }
        }
        dist_sq.sqrt()
    }

    fn search_layer_lut(&self, lut: &[f32], ep: usize, ef: usize, level: usize) -> BinaryHeap<Neighbor> {
        let mut visited = std::collections::HashSet::new();
        let mut candidates = BinaryHeap::new();
        let d_ep = self.calculate_distance_lut(lut, ep);
        let ep_neigh = Neighbor { idx: ep, distance: d_ep };
        visited.insert(ep);
        candidates.push(ep_neigh);
        let mut found_neighbors = BinaryHeap::new();
        found_neighbors.push(std::cmp::Reverse(ep_neigh));
        while let Some(c) = candidates.pop() {
            let furthest_dist = found_neighbors.peek().map(|f| f.0.distance).unwrap_or(f32::MAX);
            if c.distance > furthest_dist && found_neighbors.len() >= ef { break; }
            for &neighbor_idx in &self.layers[level][c.idx] {
                if !visited.contains(&neighbor_idx) {
                    visited.insert(neighbor_idx);
                    let d = self.calculate_distance_lut(lut, neighbor_idx);
                    let n = Neighbor { idx: neighbor_idx, distance: d };
                    if d < furthest_dist || found_neighbors.len() < ef {
                        candidates.push(n);
                        found_neighbors.push(std::cmp::Reverse(n));
                        if found_neighbors.len() > ef { found_neighbors.pop(); }
                    }
                }
            }
        }
        found_neighbors.into_iter().map(|r| r.0).collect()
    }
}

#[pymethods]
impl GajeIndex {
    #[new]
    #[pyo3(signature = (database, centroids, epigenetic_database=Vec::new(), epigenetic_centroids=Vec::new(), triplet_database=Vec::new(), triplet_centroids=Vec::new(), m=32, ef_construction=200))]
    pub fn new(
        database: Vec<Vec<u8>>, 
        centroids: Vec<f32>, 
        epigenetic_database: Vec<Vec<u8>>,
        epigenetic_centroids: Vec<f32>,
        triplet_database: Vec<Vec<u8>>,
        triplet_centroids: Vec<f32>,
        m: usize, 
        ef_construction: usize
    ) -> Self {
        let stride = if database.is_empty() { 0 } else { database[0].len() };
        let mut flat_db = Vec::with_capacity(database.len() * stride);
        for s in database { flat_db.extend(s); }
        let mut flat_epi_db = Vec::with_capacity(epigenetic_database.len() * stride);
        for s in epigenetic_database { flat_epi_db.extend(s); }
        let mut flat_tri_db = Vec::with_capacity(triplet_database.len() * stride);
        for s in triplet_database { flat_tri_db.extend(s); }
        GajeIndex {
            database: flat_db, centroids,
            epigenetic_database: flat_epi_db, epigenetic_centroids,
            triplet_database: flat_tri_db, triplet_centroids,
            stride, layers: Vec::new(), max_level: -1, entry_point: None,
            ef_construction, m, level_mult: 1.0 / (m as f64).ln(),
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
        if self.stride == 0 && !strands.is_empty() { self.stride = strands[0].len(); }
        for s in strands { self.database.extend(s); }
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
        if k > 0 && k < results.len() { results.truncate(k); }
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
        if self.entry_point.is_none() { return self.flat_search(query_vector, k); }
        let ef_val = ef.unwrap_or(std::cmp::max(k, 50));
        let has_epi = !self.epigenetic_database.is_empty();
        let has_triplet = !self.triplet_database.is_empty();
        let lut = if has_triplet {
            let mut l = Vec::with_capacity(query_vector.len() * 64);
            let c_base = &self.centroids;
            let c_epi = &self.epigenetic_centroids;
            let c_tri = &self.triplet_centroids;
            let is_multi = c_base.len() == query_vector.len() * 4;
            let is_epi_multi = c_epi.len() == query_vector.len() * 4;
            let is_tri_multi = c_tri.len() == query_vector.len() * 4;
            for (d_idx, &val) in query_vector.iter().enumerate() {
                let cb = if is_multi { &c_base[d_idx*4..(d_idx+1)*4] } else { &c_base[0..4] };
                let ce = if is_epi_multi { &c_epi[d_idx*4..(d_idx+1)*4] } else { &c_epi[0..4] };
                let ct = if is_tri_multi { &c_tri[d_idx*4..(d_idx+1)*4] } else { &c_tri[0..4] };
                for &b_val in cb {
                    for &e_val in ce {
                        for &t_val in ct {
                            let diff = val - (b_val + e_val + t_val);
                            l.push(diff * diff);
                        }
                    }
                }
            }
            l
        } else if has_epi {
            let mut l = Vec::with_capacity(query_vector.len() * 16);
            let c_base = &self.centroids;
            let c_epi = &self.epigenetic_centroids;
            let is_multi = c_base.len() == query_vector.len() * 4;
            let is_epi_multi = c_epi.len() == query_vector.len() * 4;
            for (d_idx, &val) in query_vector.iter().enumerate() {
                let cb = if is_multi { &c_base[d_idx*4..(d_idx+1)*4] } else { &c_base[0..4] };
                let ce = if is_epi_multi { &c_epi[d_idx*4..(d_idx+1)*4] } else { &c_epi[0..4] };
                for &b_val in cb {
                    for &e_val in ce {
                        let diff = val - (b_val + e_val);
                        l.push(diff * diff);
                    }
                }
            }
            l
        } else {
            let mut l = Vec::with_capacity(query_vector.len() * 4);
            let c = &self.centroids;
            let is_multi = c.len() == query_vector.len() * 4;
            for (d_idx, &val) in query_vector.iter().enumerate() {
                let cd = if is_multi { &c[d_idx*4..(d_idx+1)*4] } else { &c[0..4] };
                for b in 0..4 {
                    let diff = val - cd[b];
                    l.push(diff * diff);
                }
            }
            l
        };
        let mut curr_ep = self.entry_point.unwrap();
        for l in (1..=self.max_level).rev() {
            let res = self.search_layer_lut(&lut, curr_ep, 1, l as usize);
            if let Some(best) = res.peek() { curr_ep = best.idx; }
        }
        let final_neighbors = self.search_layer_lut(&lut, curr_ep, ef_val, 0);
        let mut results: Vec<(usize, f32)> = final_neighbors.into_iter().map(|n| (n.idx, n.distance)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        if k > 0 && k < results.len() { results.truncate(k); }
        Ok(results)
    }

    #[pyo3(signature = (query_vector, target_idx, negative_idx=None, lr=0.01))]
    pub fn refine_search_centroids(&mut self, query_vector: Vec<f32>, target_idx: usize, negative_idx: Option<usize>, lr: f32) -> PyResult<()> {
        let has_epi = !self.epigenetic_database.is_empty();
        let q_len = query_vector.len();
        let stride = self.stride;
        let strand = self.get_strand(target_idx).to_vec();
        if !has_epi {
            let c_base = &mut self.centroids;
            let is_multi = c_base.len() == q_len * 4;
            let mut dims = 0;
            for &byte in &strand {
                for j in 0..4 {
                    if dims >= q_len { break; }
                    let shift = (3 - j) * 2;
                    let bits = (byte >> shift) & 0b11;
                    let c_idx = match bits { 0b00 => 0, 0b01 => 1, 0b11 => 2, 0b10 => 3, _ => 0 };
                    let g_c_idx = if is_multi { dims * 4 + c_idx } else { c_idx };
                    c_base[g_c_idx] += lr * (query_vector[dims] - c_base[g_c_idx]) * 0.1;
                    dims += 1;
                }
            }
        } else {
            let epi_strand = self.get_epigenetic_strand(target_idx).to_vec();
            let c_base = &mut self.centroids;
            let c_epi = &mut self.epigenetic_centroids;
            let is_multi = c_base.len() == q_len * 4;
            let is_epi_multi = c_epi.len() == q_len * 4;
            let mut dims = 0;
            for i in 0..stride {
                let byte = strand[i];
                let epi_byte = epi_strand[i];
                for j in 0..4 {
                    if dims >= q_len { break; }
                    let shift = (3 - j) * 2;
                    let bits = (byte >> shift) & 0b11;
                    let e_bits = (epi_byte >> shift) & 0b11;
                    let b_idx = match bits { 0b00 => 0, 0b01 => 1, 0b11 => 2, 0b10 => 3, _ => 0 };
                    let e_idx = match e_bits { 0b00 => 0, 0b01 => 1, 0b11 => 2, 0b10 => 3, _ => 0 };
                    let g_b_idx = if is_multi { dims * 4 + b_idx } else { b_idx };
                    let g_e_idx = if is_epi_multi { dims * 4 + e_idx } else { e_idx };
                    let target = query_vector[dims];
                    let current = c_base[g_b_idx] + c_epi[g_e_idx];
                    let err = target - current;
                    c_base[g_b_idx] += lr * err * 0.5;
                    c_epi[g_e_idx] += lr * err * 0.5;
                    dims += 1;
                }
            }
        }
        if let Some(neg_idx) = negative_idx {
            let n_strand = self.get_strand(neg_idx).to_vec();
            let c_base = &mut self.centroids;
            let is_multi = c_base.len() == q_len * 4;
            let mut dims = 0;
            for &byte in &n_strand {
                for j in 0..4 {
                    if dims >= q_len { break; }
                    let shift = (3 - j) * 2;
                    let bits = (byte >> shift) & 0b11;
                    let c_idx = match bits { 0b00 => 0, 0b01 => 1, 0b11 => 2, 0b10 => 3, _ => 0 };
                    let g_c_idx = if is_multi { dims * 4 + c_idx } else { c_idx };
                    c_base[g_c_idx] -= lr * (query_vector[dims] - c_base[g_c_idx]) * 0.05;
                    dims += 1;
                }
            }
        }
        Ok(())
    }
}

#[inline(always)]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let mut sum_v = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        sum_v = vfmaq_f32(sum_v, va, vb);
        i += 4;
    }
    let mut sum = vaddvq_f32(sum_v);
    while i < n { sum += a[i] * b[i]; i += 1; }
    sum
}

#[inline(always)]
unsafe fn rms_norm_neon(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    use std::arch::aarch64::*;
    let n = x.len();
    let mut sum_v = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= n {
        let vx = vld1q_f32(x.as_ptr().add(i));
        sum_v = vfmaq_f32(sum_v, vx, vx);
        i += 4;
    }
    let mut sum_sq = vaddvq_f32(sum_v);
    while i < n { sum_sq += x[i] * x[i]; i += 1; }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    let inv_rms_v = vdupq_n_f32(inv_rms);
    let mut out = vec![0.0f32; n];
    i = 0;
    while i + 4 <= n {
        let vx = vld1q_f32(x.as_ptr().add(i));
        let vw = vld1q_f32(weight.as_ptr().add(i));
        let res = vmulq_f32(vmulq_f32(vx, inv_rms_v), vw);
        vst1q_f32(out.as_mut_ptr().add(i), res);
        i += 4;
    }
    while i < n { out[i] = (x[i] * inv_rms) * weight[i]; i += 1; }
    out
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn calculate_distance_lut_neon(lut: &[f32], strand: &[u8], epi_strand: Option<&[u8]>, tri_strand: Option<&[u8]>, n_dims: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut sum_v = vdupq_n_f32(0.0);
    let mut dims = 0;
    if let (Some(epi), Some(tri)) = (epi_strand, tri_strand) {
        while dims + 4 <= n_dims {
            let b_byte = *strand.get_unchecked(dims / 4);
            let e_byte = *epi.get_unchecked(dims / 4);
            let t_byte = *tri.get_unchecked(dims / 4);
            let mut d_v = [0.0f32; 4];
            for j in 0..4 {
                let shift = (3 - j) * 2;
                let bb = (b_byte >> shift) & 0b11;
                let eb = (e_byte >> shift) & 0b11;
                let tb = (t_byte >> shift) & 0b11;
                let b_idx = bb ^ (bb >> 1);
                let e_idx = eb ^ (eb >> 1);
                let t_idx = tb ^ (tb >> 1);
                let idx = ((b_idx << 4) | (e_idx << 2) | t_idx) as usize;
                d_v[j] = *lut.get_unchecked(dims * 64 + idx);
                dims += 1;
            }
            sum_v = vaddq_f32(sum_v, vld1q_f32(d_v.as_ptr()));
        }
    } else if let Some(epi) = epi_strand {
        while dims + 4 <= n_dims {
            let b_byte = *strand.get_unchecked(dims / 4);
            let e_byte = *epi.get_unchecked(dims / 4);
            let mut d_v = [0.0f32; 4];
            for j in 0..4 {
                let shift = (3 - j) * 2;
                let bb = (b_byte >> shift) & 0b11;
                let eb = (e_byte >> shift) & 0b11;
                let b_idx = bb ^ (bb >> 1);
                let e_idx = eb ^ (eb >> 1);
                let idx = ((b_idx << 2) | e_idx) as usize;
                d_v[j] = *lut.get_unchecked(dims * 16 + idx);
                dims += 1;
            }
            sum_v = vaddq_f32(sum_v, vld1q_f32(d_v.as_ptr()));
        }
    } else {
        while dims + 4 <= n_dims {
            let byte = *strand.get_unchecked(dims / 4);
            let mut d_v = [0.0f32; 4];
            for j in 0..4 {
                let shift = (3 - j) * 2;
                let bits = (byte >> shift) & 0b11;
                let idx = (bits ^ (bits >> 1)) as usize;
                d_v[j] = *lut.get_unchecked(dims * 4 + idx);
                dims += 1;
            }
            sum_v = vaddq_f32(sum_v, vld1q_f32(d_v.as_ptr()));
        }
    }
    let mut total = vaddvq_f32(sum_v);
    while dims < n_dims {
        let shift = (3 - (dims % 4)) * 2;
        if let (Some(epi), Some(tri)) = (epi_strand, tri_strand) {
            let bb = (strand[dims/4] >> shift) & 0b11;
            let eb = (epi[dims/4] >> shift) & 0b11;
            let tb = (tri[dims/4] >> shift) & 0b11;
            let idx = (((bb ^ (bb >> 1)) << 4) | ((eb ^ (eb >> 1)) << 2) | (tb ^ (tb >> 1))) as usize;
            total += *lut.get_unchecked(dims * 64 + idx);
        } else if let Some(epi) = epi_strand {
            let bb = (strand[dims/4] >> shift) & 0b11;
            let eb = (epi[dims/4] >> shift) & 0b11;
            let idx = (((bb ^ (bb >> 1)) << 2) | (eb ^ (eb >> 1))) as usize;
            total += *lut.get_unchecked(dims * 16 + idx);
        } else {
            let bits = (strand[dims/4] >> shift) & 0b11;
            let idx = (bits ^ (bits >> 1)) as usize;
            total += *lut.get_unchecked(dims * 4 + idx);
        }
        dims += 1;
    }
    total.sqrt()
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

#[pyfunction]
#[pyo3(signature = (vector, thresholds=None))]
pub fn quantize_embedding(vector: Vec<f32>, thresholds: Option<Vec<f32>>, py: Python<'_>) -> PyResult<PyObject> {
    let t = thresholds.unwrap_or_else(|| vec![-0.43, 0.0, 0.43]);
    let n = vector.len();
    let mut packed = Vec::with_capacity((n + 3) / 4);
    for i in (0..n).step_by(4) {
        let mut byte = 0u8;
        for j in 0..4 {
            if i + j < n {
                let val = vector[i + j];
                let bits = if val < t[0] { 0b00 } else if val < t[1] { 0b01 } else if val < t[2] { 0b11 } else { 0b10 };
                byte = (byte << 2) | bits;
            }
        }
        packed.push(byte);
    }
    Ok(PyBytes::new(py, &packed).into())
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
    if k > 0 && k < results.len() { results.truncate(k); }
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
        if k > 0 && k < res.len() { res.truncate(k); }
        return Ok(res);
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Query error"))
}

#[pyfunction]
#[pyo3(signature = (logits, repetition_penalty=1.2, last_tokens=None))]
pub fn apply_repetition_penalty(logits: Vec<f32>, repetition_penalty: f32, last_tokens: Option<Vec<usize>>) -> PyResult<Vec<f32>> {
    let mut out = logits;
    if let Some(tokens) = last_tokens {
        for &tid in &tokens {
            if tid < out.len() {
                if out[tid] > 0.0 { out[tid] /= repetition_penalty; } else { out[tid] *= repetition_penalty; }
            }
        }
    }
    Ok(out)
}

#[pyfunction]
pub fn calculate_shannon_entropy(data: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    if data.is_empty() { return Ok(vec![]); }
    let n_vectors = data.len();
    let dim = data[0].len();
    let entropies: Vec<f32> = (0..dim).into_par_iter().map(|d_idx| {
        let mut values = Vec::with_capacity(n_vectors);
        for v in &data { values.push(v[d_idx]); }
        let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max - min;
        if range < 1e-6 { return 0.0f32; }
        let n_bins = 64;
        let mut bins = vec![0usize; n_bins];
        for &v in &values {
            let bin_idx = (((v - min) / range) * (n_bins - 1) as f32) as usize;
            bins[bin_idx.min(n_bins - 1)] += 1;
        }
        let mut entropy = 0.0f32;
        for &count in &bins {
            if count > 0 {
                let p = count as f32 / n_vectors as f32;
                entropy -= p * p.log2();
            }
        }
        entropy
    }).collect();
    Ok(entropies)
}

#[pyfunction]
pub fn genomize_f32_native(data_u8: Vec<u8>, block_size: usize, anchor_threshold: f32) -> PyResult<(Vec<u8>, Vec<f32>, Vec<f32>)> {
    let f32_data: &[f32] = unsafe { std::slice::from_raw_parts(data_u8.as_ptr() as *const f32, data_u8.len() / 4) };
    let n_elements = f32_data.len();
    let n_blocks = n_elements / block_size;
    let mut dna_database = Vec::with_capacity(n_elements / 4);
    let mut all_centroids = Vec::with_capacity(n_blocks * 4);
    let mut anchors = vec![0.0f32; n_elements];
    let base_c = [-1.510f32, -0.4528, 0.4528, 1.510];
    for i in 0..n_blocks {
        let start = i * block_size;
        let block_f32 = &f32_data[start..start + block_size];
        let mut sum = 0.0f32;
        for &val in block_f32 { sum += val; }
        let mean = sum / block_size as f32;
        let mut var_sum = 0.0f32;
        for &val in block_f32 { let diff = val - mean; var_sum += diff * diff; }
        let std = (var_sum / block_size as f32).sqrt() + 1e-6;
        let t = [mean - std, mean, mean + std];
        let c = [mean + base_c[0]*std, mean + base_c[1]*std, mean + base_c[2]*std, mean + base_c[3]*std];
        let mut packed_block = Vec::with_capacity(block_size / 4);
        for k in 0..(block_size / 4) {
            let mut byte = 0u8;
            for s in 0..4 {
                let val = block_f32[k * 4 + s];
                let bits = if val < t[0] { 0b00 } else if val < t[1] { 0b01 } else if val < t[2] { 0b11 } else { 0b10 };
                let c_val = match bits { 0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0 };
                let residual = val - c_val;
                if residual.abs() > anchor_threshold { anchors[start + k * 4 + s] = residual; }
                byte = (byte << 2) | bits;
            }
            packed_block.push(byte);
        }
        dna_database.extend(packed_block);
        for &cv in &c { all_centroids.push(cv); }
    }
    Ok((dna_database, all_centroids, anchors))
}

#[pyfunction]
pub fn genomize_f16_native(data_u8: Vec<u8>, block_size: usize, anchor_threshold: f32) -> PyResult<(Vec<u8>, Vec<f32>, Vec<f32>)> {
    let f16_data: &[f16] = unsafe { std::slice::from_raw_parts(data_u8.as_ptr() as *const f16, data_u8.len() / 2) };
    let n_elements = f16_data.len();
    let n_blocks = n_elements / block_size;
    let mut dna_database = Vec::with_capacity(n_elements / 4);
    let mut all_centroids = Vec::with_capacity(n_blocks * 4);
    let mut anchors = vec![0.0f32; n_elements];
    let base_c = [-1.510f32, -0.4528, 0.4528, 1.510];
    for i in 0..n_blocks {
        let start = i * block_size;
        let block_f16 = &f16_data[start..start + block_size];
        let mut block_f32 = vec![0.0f32; block_size];
        let mut sum = 0.0f32;
        for j in 0..block_size { let val = block_f16[j].to_f32(); block_f32[j] = val; sum += val; }
        let mean = sum / block_size as f32;
        let mut var_sum = 0.0f32;
        for &val in &block_f32 { let diff = val - mean; var_sum += diff * diff; }
        let std = (var_sum / block_size as f32).sqrt() + 1e-6;
        let t = [mean - std, mean, mean + std];
        let c = [mean + base_c[0]*std, mean + base_c[1]*std, mean + base_c[2]*std, mean + base_c[3]*std];
        let mut packed_block = Vec::with_capacity(block_size / 4);
        for k in 0..(block_size / 4) {
            let mut byte = 0u8;
            for s in 0..4 {
                let val = block_f32[k * 4 + s];
                let bits = if val < t[0] { 0b00 } else if val < t[1] { 0b01 } else if val < t[2] { 0b11 } else { 0b10 };
                let c_val = match bits { 0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0 };
                let residual = val - c_val;
                if residual.abs() > anchor_threshold { anchors[start + k * 4 + s] = residual; }
                byte = (byte << 2) | bits;
            }
            packed_block.push(byte);
        }
        dna_database.extend(packed_block);
        for &cv in &c { all_centroids.push(cv); }
    }
    Ok((dna_database, all_centroids, anchors))
}

#[pyfunction]
#[pyo3(signature = (data_u8, out_features, in_features))]
pub fn dequantize_q8_0_native(data_u8: Vec<u8>, out_features: usize, in_features: usize) -> PyResult<Vec<f32>> {
    let n_blocks = in_features / 32;
    let block_size = 34;
    let mut results = vec![0.0f32; out_features * in_features];
    results.par_chunks_mut(in_features).enumerate().for_each(|(i, row)| {
        let row_offset = i * n_blocks * block_size;
        for b in 0..n_blocks {
            let offset = row_offset + b * block_size;
            if offset + 2 > data_u8.len() { break; }
            let delta_bytes = [data_u8[offset], data_u8[offset + 1]];
            let delta = f16::from_le_bytes(delta_bytes).to_f32();
            for j in 0..32 {
                if offset + 2 + j >= data_u8.len() { break; }
                let q = data_u8[offset + 2 + j] as i8;
                row[b * 32 + j] = (q as f32) * delta;
            }
        }
    });
    Ok(results)
}

#[pyfunction]
#[pyo3(signature = (logits, temperature=1.0, top_p=0.9))]
pub fn sample_top_p(logits: Vec<f32>, temperature: f32, top_p: f32) -> PyResult<usize> {
    let mut probs: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &l)| (i, (l / temperature).exp())).collect();
    let sum_exp: f32 = probs.iter().map(|(_, p)| p).sum();
    for p in &mut probs { p.1 /= sum_exp; }
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let mut cumulative_prob = 0.0;
    let mut cutoff_idx = probs.len();
    for (i, &(_, p)) in probs.iter().enumerate() {
        cumulative_prob += p;
        if cumulative_prob > top_p { cutoff_idx = i + 1; break; }
    }
    probs.truncate(cutoff_idx);
    let final_sum: f32 = probs.iter().map(|(_, p)| p).sum();
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen::<f32>() * final_sum;
    let mut current_sum = 0.0;
    for &(id, p) in &probs {
        current_sum += p;
        if r <= current_sum { return Ok(id); }
    }
    Ok(probs[0].0)
}

#[pyclass]
pub struct GenomicLinear {
    #[pyo3(get)]
    pub database: Vec<u8>,
    #[pyo3(get)]
    pub anchors: Vec<f32>,
    #[pyo3(get, set)]
    pub centroids: Vec<f32>,
    #[pyo3(get)]
    pub out_features: usize,
    #[pyo3(get)]
    pub in_features: usize,
    #[pyo3(get)]
    pub block_size: usize,
    #[pyo3(get, set)]
    pub rmsnorm_weight: Vec<f32>,
    #[pyo3(get, set)]
    pub eps: f32,
    stride: usize,
}

#[pymethods]
impl GenomicLinear {
    #[new]
    #[pyo3(signature = (database, anchors, centroids, out_features, in_features, block_size, rmsnorm_weight=Vec::new(), eps=1e-6))]
    pub fn new(database: Vec<u8>, anchors: Vec<f32>, centroids: Vec<f32>, out_features: usize, in_features: usize, block_size: usize, rmsnorm_weight: Vec<f32>, eps: f32) -> Self {
        let stride = block_size / 4;
        GenomicLinear { database, anchors, centroids, out_features, in_features, block_size, rmsnorm_weight, eps, stride }
    }
    pub fn forward(&self, mut input_vector: Vec<f32>) -> PyResult<Vec<f32>> {
        if !self.rmsnorm_weight.is_empty() { input_vector = unsafe { rms_norm_neon(&input_vector, &self.rmsnorm_weight, self.eps) }; }
        let n_blocks_per_row = self.in_features / self.block_size;
        let has_anchors = !self.anchors.is_empty();
        let results: Vec<f32> = (0..self.out_features).into_par_iter().map(|i| {
            let mut row_sum = 0.0f32;
            let row_offset = i * n_blocks_per_row * self.stride;
            for j in 0..n_blocks_per_row {
                let block_start = row_offset + j * self.stride;
                let block_weights = &self.database[block_start..block_start + self.stride];
                let input_block = &input_vector[j * self.block_size .. (j + 1) * self.block_size];
                let c_offset = (i * n_blocks_per_row + j) * 4;
                let c = &self.centroids[c_offset..c_offset + 4];
                let mut dims = 0;
                for k in 0..self.stride {
                    let byte = block_weights[k];
                    for s in 0..4 {
                        let shift = (3 - s) * 2;
                        let bits = (byte >> shift) & 0b11;
                        let val = match bits { 0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0 };
                        row_sum += input_block[dims] * val;
                        dims += 1;
                    }
                }
            }
            if has_anchors {
                let anchor_row = &self.anchors[i * self.in_features .. (i + 1) * self.in_features];
                row_sum += unsafe { dot_product_neon(anchor_row, &input_vector) };
            }
            row_sum.clamp(-100.0, 100.0)
        }).collect();
        Ok(results)
    }
    pub fn refine_centroids(&mut self, input_vector: Vec<f32>, target_output: Vec<f32>, lr: f32) -> PyResult<()> {
        let n_blocks_per_row = self.in_features / self.block_size;
        let mut activations = input_vector.clone();
        if !self.rmsnorm_weight.is_empty() { activations = unsafe { rms_norm_neon(&activations, &self.rmsnorm_weight, self.eps) }; }
        let current_output = self.forward(input_vector)?;
        let block_scale = 1.0 / self.block_size as f32;
        for i in 0..self.out_features {
            let error = current_output[i] - target_output[i];
            let row_offset = i * n_blocks_per_row * self.stride;
            for j in 0..n_blocks_per_row {
                let block_start = row_offset + j * self.stride;
                let block_weights = &self.database[block_start..block_start + self.stride];
                let input_block = &activations[j * self.block_size .. (j + 1) * self.block_size];
                let c_offset = (i * n_blocks_per_row + j) * 4;
                let mut dims = 0;
                for k in 0..self.stride {
                    let byte = block_weights[k];
                    for s in 0..4 {
                        let shift = (3 - s) * 2;
                        let bits = (byte >> shift) & 0b11;
                        let c_idx = match bits { 0b00 => 0, 0b01 => 1, 0b11 => 2, 0b10 => 3, _ => 0 };
                        let grad = error * input_block[dims] * block_scale;
                        self.centroids[c_offset + c_idx] -= lr * grad;
                        dims += 1;
                    }
                }
            }
        }
        Ok(())
    }
}

#[pyclass]
pub struct GenomicAttention {
    pub q_database: Vec<u8>, pub k_database: Vec<u8>, pub v_database: Vec<u8>,
    pub centroids: Vec<f32>, pub stride: usize, pub n_head: usize, pub n_head_kv: usize,
    pub head_dim: usize, pub k_cache: Vec<Vec<u8>>, pub v_cache: Vec<Vec<u8>>,
}

#[pymethods]
impl GenomicAttention {
    #[new]
    pub fn new(q: Vec<u8>, k: Vec<u8>, v: Vec<u8>, centroids: Vec<f32>, stride: usize, n_head: usize, n_head_kv: usize) -> Self {
        GenomicAttention { q_database: q, k_database: k, v_database: v, centroids, stride, n_head, n_head_kv, head_dim: stride * 4, k_cache: Vec::new(), v_cache: Vec::new() }
    }
    pub fn forward(&mut self, input_vector: Vec<f32>, _pos: usize) -> PyResult<Vec<f32>> {
        let n_embd = input_vector.len();
        let q_len = self.n_head * self.head_dim;
        let mut query = vec![0.0f32; q_len];
        let c_base = &self.centroids;
        for h in 0..self.n_head {
            let q_dna = &self.q_database[h * self.stride .. (h + 1) * self.stride];
            let q_start = h * self.head_dim;
            let mut dims = 0;
            for &byte in q_dna {
                for s in 0..4 {
                    let shift = (3 - s) * 2;
                    let bits = (byte >> shift) & 0b11;
                    let c_idx = match bits { 0b00 => 0, 0b01 => 1, 0b11 => 2, 0b10 => 3, _ => 0 };
                    let val = c_base[q_start * 4 + dims * 4 + c_idx];
                    query[q_start + dims] = unsafe { dot_product_neon(&input_vector, &vec![val; n_embd]) };
                    dims += 1;
                }
            }
        }
        Ok(query)
    }
}

#[pyclass]
pub struct GenomicSwiGLU {
    pub w_gate: Vec<u8>, pub w_up: Vec<u8>, pub centroids: Vec<f32>,
    pub out_features: usize, pub in_features: usize, pub block_size: usize, pub stride: usize,
}

#[pymethods]
impl GenomicSwiGLU {
    #[new]
    pub fn new(w_gate: Vec<u8>, w_up: Vec<u8>, centroids: Vec<f32>, out_features: usize, in_features: usize, block_size: usize) -> Self {
        GenomicSwiGLU { w_gate, w_up, centroids, out_features, in_features, block_size, stride: block_size / 4 }
    }
    pub fn forward(&self, input_vector: Vec<f32>) -> PyResult<Vec<f32>> {
        let n_blocks_per_row = self.in_features / self.block_size;
        let silu = |x: f32| x / (1.0 + (-x).exp());
        let results: Vec<f32> = (0..self.out_features).into_par_iter().map(|i| {
            let mut gate_sum = 0.0f32;
            let mut up_sum = 0.0f32;
            let row_offset = i * n_blocks_per_row * self.stride;
            for j in 0..n_blocks_per_row {
                let block_start = row_offset + j * self.stride;
                let g_weights = &self.w_gate[block_start..block_start + self.stride];
                let u_weights = &self.w_up[block_start..block_start + self.stride];
                let input_block = &input_vector[j * self.block_size .. (j + 1) * self.block_size];
                let c_offset = (i * n_blocks_per_row + j) * 4;
                let c = &self.centroids[c_offset..c_offset + 4];
                let mut dims = 0;
                for k in 0..self.stride {
                    let g_byte = g_weights[k];
                    let u_byte = u_weights[k];
                    for s in 0..4 {
                        let shift = (3 - s) * 2;
                        let g_bits = (g_byte >> shift) & 0b11;
                        let u_bits = (u_byte >> shift) & 0b11;
                        let g_val = match g_bits { 0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0 };
                        let u_val = match u_bits { 0b00 => c[0], 0b01 => c[1], 0b11 => c[2], 0b10 => c[3], _ => 0.0 };
                        let inp = input_block[dims];
                        gate_sum += inp * g_val; up_sum += inp * u_val; dims += 1;
                    }
                }
            }
            silu(gate_sum) * up_sum
        }).collect();
        Ok(results)
    }
}

#[pymodule]
fn _impl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GajeIndex>()?;
    m.add_class::<GenomicAttention>()?;
    m.add_class::<GenomicLinear>()?;
    m.add_class::<GenomicSwiGLU>()?;
    m.add_function(wrap_pyfunction!(quantize_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_pq, m)?)?;
    m.add_function(wrap_pyfunction!(dna_similarity_search_adc, m)?)?;
    m.add_function(wrap_pyfunction!(dna_similarity_search, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(apply_repetition_penalty, m)?)?;
    m.add_function(wrap_pyfunction!(genomize_f32_native, m)?)?;
    m.add_function(wrap_pyfunction!(genomize_f16_native, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_q8_0_native, m)?)?;
    m.add_function(wrap_pyfunction!(sample_top_p, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_shannon_entropy, m)?)?;
    Ok(())
}
