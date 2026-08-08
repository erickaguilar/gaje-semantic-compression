use half::f16;
use rayon::prelude::*;
use std::sync::Arc;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[derive(Clone)]
pub enum WeightDatabase {
    Genomic2Bit(Arc<Vec<u8>>),
    Genomic4Bit(Arc<Vec<u8>>),
    GenomicF32(Arc<Vec<f32>>),
}

pub trait GenomicOperable {
    fn bit_depth(&self) -> u8;
    fn read(&self, byte_idx: usize, sub_idx: usize) -> u8;
    fn mutate(&mut self, byte_idx: usize, sub_idx: usize, new_bits: u8);
    fn len_bytes(&self) -> usize;
}

impl GenomicOperable for WeightDatabase {
    fn bit_depth(&self) -> u8 {
        match self {
            WeightDatabase::Genomic2Bit(_) => 2,
            WeightDatabase::Genomic4Bit(_) => 4,
            WeightDatabase::GenomicF32(_) => 32,
        }
    }
    fn read(&self, byte_idx: usize, sub_idx: usize) -> u8 {
        match self {
            WeightDatabase::Genomic2Bit(db) => (db[byte_idx] >> ((3 - sub_idx) * 2)) & 0b11,
            WeightDatabase::Genomic4Bit(db) => {
                if sub_idx == 0 {
                    db[byte_idx] >> 4
                } else {
                    db[byte_idx] & 0x0F
                }
            }
            _ => 0,
        }
    }
    fn mutate(&mut self, byte_idx: usize, sub_idx: usize, new_bits: u8) {
        match self {
            WeightDatabase::Genomic2Bit(ref mut db) => {
                let db_mut = Arc::make_mut(db);
                let shift = (3 - sub_idx) * 2;
                db_mut[byte_idx] &= !(0b11 << shift);
                db_mut[byte_idx] |= (new_bits & 0b11) << shift;
            }
            WeightDatabase::Genomic4Bit(ref mut db) => {
                let db_mut = Arc::make_mut(db);
                if sub_idx == 0 {
                    db_mut[byte_idx] = (db_mut[byte_idx] & 0x0F) | (new_bits << 4);
                } else {
                    db_mut[byte_idx] = (db_mut[byte_idx] & 0xF0) | (new_bits & 0x0F);
                }
            }
            _ => {}
        }
    }
    fn len_bytes(&self) -> usize {
        match self {
            WeightDatabase::Genomic2Bit(db) => db.len(),
            WeightDatabase::Genomic4Bit(db) => db.len(),
            WeightDatabase::GenomicF32(db) => db.len() * 4,
        }
    }
}

#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct GenomicLinear {
    pub weight_db: WeightDatabase,
    pub epi_strands: Arc<Vec<u8>>,
    pub tri_strands: Arc<Vec<u8>>,
    pub epi_cols: Arc<Vec<(usize, usize)>>,
    pub tri_cols: Arc<Vec<(usize, usize)>>,
    pub anchor_indices: Arc<Vec<u32>>,
    pub anchor_values: Arc<Vec<f16>>,
    pub anchor_row_ptrs: Arc<Vec<usize>>,
    pub centroids: Vec<f32>,
    pub epigenetic_centroids: Vec<f32>,
    pub triplet_centroids: Vec<f32>,
    pub out_features: usize,
    pub in_features: usize,
    pub block_size: usize,
    pub rmsnorm_weight: Vec<f32>,
    pub eps: f32,
    pub bias: Vec<f32>,
    pub stride: usize,
}

impl GenomicLinear {
    pub fn new(
        database: Vec<u8>,
        anchors_u8: Vec<u8>,
        centroids: Vec<f32>,
        out_features: usize,
        in_features: usize,
        block_size: usize,
        rmsnorm_weight: Vec<f32>,
        eps: f32,
        _precision_mask: Vec<u8>,
        _epigenetic_database: Vec<u8>,
        epigenetic_centroids: Vec<f32>,
        _triplet_database: Vec<u8>,
        triplet_centroids: Vec<f32>,
        bias: Vec<f32>,
        bit_depth: u8,
    ) -> Self {
        let weight_db = match bit_depth {
            4 => WeightDatabase::Genomic4Bit(Arc::new(database)),
            32 => {
                let f32_data: Vec<f32> = unsafe {
                    std::slice::from_raw_parts(database.as_ptr() as *const f32, database.len() / 4)
                        .to_vec()
                };
                WeightDatabase::GenomicF32(Arc::new(f32_data))
            }
            _ => WeightDatabase::Genomic2Bit(Arc::new(database)),
        };
        let stride = match bit_depth {
            4 => block_size / 2,
            32 => block_size,
            _ => block_size / 4,
        };
        let (anchor_indices, anchor_values, anchor_row_ptrs) =
            if anchors_u8.len() >= 4 && &anchors_u8[0..4] == b"GAJE" {
                let count = u32::from_le_bytes(anchors_u8[4..8].try_into().unwrap()) as usize;
                let mut indices = Vec::with_capacity(count);
                let mut values = Vec::with_capacity(count);
                let mut row_ptrs = vec![0; out_features + 1];
                let idx_s = 8;
                let val_s = idx_s + count * 4;
                let ptr_s = val_s + count * 2;
                for i in 0..count {
                    indices.push(u32::from_le_bytes(
                        anchors_u8[idx_s + i * 4..idx_s + (i + 1) * 4]
                            .try_into()
                            .unwrap(),
                    ));
                    values.push(f16::from_le_bytes(
                        anchors_u8[val_s + i * 2..val_s + (i + 1) * 2]
                            .try_into()
                            .unwrap(),
                    ));
                }
                if anchors_u8.len() >= ptr_s + (out_features + 1) * 8 {
                    for i in 0..=out_features {
                        row_ptrs[i] = u64::from_le_bytes(
                            anchors_u8[ptr_s + i * 8..ptr_s + (i + 1) * 8]
                                .try_into()
                                .unwrap(),
                        ) as usize;
                    }
                } else {
                    let mut current_row = 0;
                    for (anchor_idx, &flat_idx) in indices.iter().enumerate() {
                        let r = flat_idx as usize / in_features;
                        while current_row < r {
                            row_ptrs[current_row + 1] = anchor_idx;
                            current_row += 1;
                        }
                    }
                    while current_row < out_features {
                        row_ptrs[current_row + 1] = indices.len();
                        current_row += 1;
                    }
                }
                (indices, values, row_ptrs)
            } else {
                (Vec::new(), Vec::new(), vec![0; out_features + 1])
            };

        let n_blocks = in_features / block_size;
        let mut final_centroids = centroids;
        if bit_depth == 4 {
            if final_centroids.len() == 16 {
                let mut expanded = Vec::with_capacity(out_features * n_blocks * 16);
                for _ in 0..(out_features * n_blocks) {
                    expanded.extend_from_slice(&final_centroids[0..16]);
                }
                final_centroids = expanded;
            } else if final_centroids.len() == n_blocks * 16 {
                let mut expanded = Vec::with_capacity(out_features * n_blocks * 16);
                for _ in 0..out_features {
                    expanded.extend_from_slice(&final_centroids);
                }
                final_centroids = expanded;
            }
        } else if bit_depth != 32 {
            if final_centroids.len() == 4 {
                let mut expanded = Vec::with_capacity(out_features * n_blocks * 4);
                for _ in 0..(out_features * n_blocks) {
                    expanded.extend_from_slice(&final_centroids[0..4]);
                }
                final_centroids = expanded;
            } else if final_centroids.len() == n_blocks * 4 {
                let mut expanded = Vec::with_capacity(out_features * n_blocks * 4);
                for _ in 0..out_features {
                    expanded.extend_from_slice(&final_centroids);
                }
                final_centroids = expanded;
            }
        }

        GenomicLinear {
            weight_db,
            epi_strands: Arc::new(Vec::new()),
            tri_strands: Arc::new(Vec::new()),
            epi_cols: Arc::new(Vec::new()),
            tri_cols: Arc::new(Vec::new()),
            anchor_indices: Arc::new(anchor_indices),
            anchor_values: Arc::new(anchor_values),
            anchor_row_ptrs: Arc::new(anchor_row_ptrs),
            centroids: final_centroids,
            epigenetic_centroids,
            triplet_centroids,
            out_features,
            in_features,
            block_size,
            rmsnorm_weight,
            eps,
            bias,
            stride,
        }
    }

    pub fn database_mut(&mut self) -> &mut Vec<u8> {
        match &mut self.weight_db {
            WeightDatabase::Genomic2Bit(db) => Arc::make_mut(db),
            WeightDatabase::Genomic4Bit(db) => Arc::make_mut(db),
            _ => panic!("N/A"),
        }
    }
    pub fn database_ref(&self) -> &[u8] {
        match &self.weight_db {
            WeightDatabase::Genomic2Bit(db) => db.as_ref(),
            WeightDatabase::Genomic4Bit(db) => db.as_ref(),
            WeightDatabase::GenomicF32(db) => unsafe {
                std::slice::from_raw_parts(db.as_ptr() as *const u8, db.len() * 4)
            },
        }
    }
    pub fn bit_depth(&self) -> u8 {
        match &self.weight_db {
            WeightDatabase::Genomic2Bit(_) => 2,
            WeightDatabase::Genomic4Bit(_) => 4,
            WeightDatabase::GenomicF32(_) => 32,
        }
    }

    pub fn empty() -> Self {
        GenomicLinear {
            weight_db: WeightDatabase::GenomicF32(Arc::new(Vec::new())),
            epi_strands: Arc::new(Vec::new()),
            tri_strands: Arc::new(Vec::new()),
            epi_cols: Arc::new(Vec::new()),
            tri_cols: Arc::new(Vec::new()),
            anchor_indices: Arc::new(Vec::new()),
            anchor_values: Arc::new(Vec::new()),
            anchor_row_ptrs: Arc::new(vec![0]),
            centroids: Vec::new(),
            epigenetic_centroids: Vec::new(),
            triplet_centroids: Vec::new(),
            out_features: 0,
            in_features: 0,
            block_size: 32,
            stride: 8,
            rmsnorm_weight: Vec::new(),
            eps: 1e-6,
            bias: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn compute_single_row(
        &self,
        i: usize,
        input: &[f32],
        m_factors: &[f32; 4],
        n_blocks: usize,
    ) -> f32 {
        let mut sum = 0.0f32;
        match &self.weight_db {
            WeightDatabase::GenomicF32(db) => {
                let row_off = i * self.in_features;
                let row_weights = &db[row_off..row_off + self.in_features];
                if row_weights.len() == input.len() {
                    sum = unsafe { crate::compute::kernels::dot_product(input, row_weights) };
                } else {
                    sum = input
                        .iter()
                        .zip(row_weights.iter())
                        .map(|(x, w)| x * w)
                        .sum();
                }
            }
            WeightDatabase::Genomic2Bit(db) => {
                let row_off = i * n_blocks * self.stride;
                let db_slice = db
                    .get(row_off..row_off + n_blocks * self.stride)
                    .unwrap_or(&[]);
                let c_start = i * n_blocks * 4;
                let c_slice = if self.centroids.len() >= c_start + n_blocks * 4 {
                    &self.centroids[c_start..c_start + n_blocks * 4]
                } else if self.centroids.len() >= 4 {
                    &self.centroids[0..4]
                } else {
                    &[]
                };
                if !db_slice.is_empty() && !c_slice.is_empty() {
                    sum = unsafe {
                        crate::compute::kernels::genomic_dot_product(
                            db_slice,
                            input,
                            c_slice,
                            self.stride,
                            n_blocks,
                            m_factors,
                        )
                    };
                }
            }
            WeightDatabase::Genomic4Bit(db) => {
                let row_off = i * n_blocks * self.stride;
                let db_slice = db
                    .get(row_off..row_off + n_blocks * self.stride)
                    .unwrap_or(&[]);
                let c_start = i * n_blocks * 16;
                let c_slice = if self.centroids.len() >= c_start + n_blocks * 16 {
                    &self.centroids[c_start..c_start + n_blocks * 16]
                } else if self.centroids.len() >= 16 {
                    &self.centroids[0..16]
                } else {
                    &[]
                };
                if !db_slice.is_empty() && !c_slice.is_empty() {
                    sum = unsafe {
                        crate::compute::kernels::genomic_dot_product_4bit(
                            db_slice,
                            input,
                            c_slice,
                            self.stride,
                            n_blocks,
                        )
                    };
                }
            }
        }
        let a_s = self.anchor_row_ptrs.get(i).copied().unwrap_or(0);
        let a_e = self.anchor_row_ptrs.get(i + 1).copied().unwrap_or(a_s);
        if a_s < a_e && a_e <= self.anchor_indices.len() && a_e <= self.anchor_values.len() {
            for k in a_s..a_e {
                let col_idx = (self.anchor_indices[k] as usize) % self.in_features;
                if let Some(&in_val) = input.get(col_idx) {
                    sum += in_val * self.anchor_values[k].to_f32();
                }
            }
        }
        if let Some(&b_val) = self.bias.get(i) {
            sum += b_val;
        }
        sum
    }

    pub fn forward_core(
        &self,
        input: Vec<f32>,
        modulation_factors: Option<[f32; 4]>,
        _activate_rna: bool,
    ) -> Result<Vec<f32>, String> {
        let n_blocks = self.in_features / self.block_size;
        let m_factors = modulation_factors.unwrap_or([1.0f32; 4]);
        let results: Vec<f32> = (0..self.out_features)
            .into_par_iter()
            .map(|i| self.compute_single_row(i, &input, &m_factors, n_blocks))
            .collect();
        Ok(results)
    }

    pub fn forward_fused_3(
        l1: &GenomicLinear,
        l2: &GenomicLinear,
        l3: &GenomicLinear,
        input: &[f32],
        modulation_factors: Option<[f32; 4]>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let n_blocks = l1.in_features / l1.block_size;
        let m_factors = modulation_factors.unwrap_or([1.0f32; 4]);
        let o1 = l1.out_features;
        let o2 = l2.out_features;
        let o3 = l3.out_features;
        let total = o1 + o2 + o3;

        let fused_out: Vec<f32> = (0..total)
            .into_par_iter()
            .map(|idx| {
                if idx < o1 {
                    l1.compute_single_row(idx, input, &m_factors, n_blocks)
                } else if idx < o1 + o2 {
                    l2.compute_single_row(idx - o1, input, &m_factors, n_blocks)
                } else {
                    l3.compute_single_row(idx - o1 - o2, input, &m_factors, n_blocks)
                }
            })
            .collect();

        let res1 = fused_out[0..o1].to_vec();
        let res2 = fused_out[o1..o1 + o2].to_vec();
        let res3 = fused_out[o1 + o2..total].to_vec();
        Ok((res1, res2, res3))
    }

    pub fn forward_fused_2(
        l1: &GenomicLinear,
        l2: &GenomicLinear,
        input: &[f32],
        modulation_factors: Option<[f32; 4]>,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let n_blocks = l1.in_features / l1.block_size;
        let m_factors = modulation_factors.unwrap_or([1.0f32; 4]);
        let o1 = l1.out_features;
        let o2 = l2.out_features;
        let total = o1 + o2;

        let fused_out: Vec<f32> = (0..total)
            .into_par_iter()
            .map(|idx| {
                if idx < o1 {
                    l1.compute_single_row(idx, input, &m_factors, n_blocks)
                } else {
                    l2.compute_single_row(idx - o1, input, &m_factors, n_blocks)
                }
            })
            .collect();

        let res1 = fused_out[0..o1].to_vec();
        let res2 = fused_out[o1..total].to_vec();
        Ok((res1, res2))
    }

    pub fn get_row_core(&self, idx: usize) -> Result<Vec<f32>, String> {
        let mut res = vec![0.0f32; self.in_features];
        if self.out_features == 0 || self.in_features == 0 {
            return Ok(res);
        }
        let safe_idx = idx % self.out_features;
        let n_blocks = self.in_features / self.block_size;

        match &self.weight_db {
            WeightDatabase::GenomicF32(db) => {
                let row_off = safe_idx * self.in_features;
                if row_off + self.in_features <= db.len() {
                    res.copy_from_slice(&db[row_off..row_off + self.in_features]);
                }
            }
            WeightDatabase::Genomic2Bit(db) => {
                let row_off = safe_idx * n_blocks * self.stride;
                let req_bytes = n_blocks * self.stride;
                if row_off + req_bytes <= db.len() {
                    for b in 0..n_blocks {
                        let c_off = (safe_idx * n_blocks + b) * 4;
                        let c_slice = if self.centroids.len() == 4 {
                            &self.centroids[0..4]
                        } else if c_off + 4 <= self.centroids.len() {
                            &self.centroids[c_off..c_off + 4]
                        } else if !self.centroids.is_empty() {
                            &self.centroids[0..4.min(self.centroids.len())]
                        } else {
                            &[0.0, 0.0, 0.0, 0.0]
                        };
                        if let Ok(decoded) = crate::compute::math::dequantize_embedding_core(
                            &db[row_off + b * self.stride..row_off + (b + 1) * self.stride],
                            self.block_size,
                            Some(c_slice),
                        ) {
                            res[b * self.block_size..(b + 1) * self.block_size]
                                .copy_from_slice(&decoded);
                        }
                    }
                }
            }
            WeightDatabase::Genomic4Bit(db) => {
                let row_off = safe_idx * n_blocks * self.stride;
                let req_bytes = n_blocks * self.stride;
                if row_off + req_bytes <= db.len() {
                    for b in 0..n_blocks {
                        let c_off = (safe_idx * n_blocks + b) * 16;
                        let centroids = if self.centroids.len() == 16 {
                            &self.centroids[0..16]
                        } else if c_off + 16 <= self.centroids.len() {
                            &self.centroids[c_off..c_off + 16]
                        } else if !self.centroids.is_empty() {
                            &self.centroids[0..16.min(self.centroids.len())]
                        } else {
                            &[0.0; 16]
                        };
                        for k in 0..self.stride {
                            let byte = db[row_off + b * self.stride + k];
                            res[b * self.block_size + k * 2] = centroids[(byte >> 4) as usize];
                            res[b * self.block_size + k * 2 + 1] =
                                centroids[(byte & 0x0F) as usize];
                        }
                    }
                }
            }
        }
        Ok(res)
    }

    pub fn backward_core(&self, _d_output: Vec<f32>) -> Result<Vec<f32>, String> {
        Ok(vec![0.0; self.in_features])
    }
    pub fn refine_with_grads_core(
        &mut self,
        _input: Vec<f32>,
        _grads: Vec<f32>,
        _lr: f32,
    ) -> Result<(), String> {
        Ok(())
    }
    pub fn apply_mutation_core(&mut self, _delta: Vec<f32>, _undo: bool) -> Result<(), String> {
        Ok(())
    }
    pub fn apply_weighted_mutation_core(
        &mut self,
        _delta: Vec<f32>,
        _weight: f32,
    ) -> Result<(), String> {
        Ok(())
    }
    pub fn mutate_random_core(&mut self, _scale: f32) -> Result<Vec<f32>, String> {
        Ok(vec![])
    }
    pub fn recalibrate_centroids_core(&mut self, _shift: f32) -> Result<(), String> {
        Ok(())
    }
    pub fn apply_vector_equilibrium_alignment_core(
        &mut self,
        _strength: f32,
    ) -> Result<(), String> {
        Ok(())
    }
    pub fn anchors_sparse_buffer(&self) -> Vec<u8> {
        vec![]
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl GenomicLinear {
    #[new]
    #[pyo3(signature = (database, anchors_u8, centroids, out_features, in_features, block_size, rmsnorm_weight=Vec::new(), eps=1e-6, precision_mask=Vec::new(), epigenetic_database=Vec::new(), epigenetic_centroids=Vec::new(), triplet_database=Vec::new(), triplet_centroids=Vec::new(), bias=Vec::new(), bit_depth=2))]
    pub fn py_new(
        database: Vec<u8>,
        anchors_u8: Vec<u8>,
        centroids: Vec<f32>,
        out_features: usize,
        in_features: usize,
        block_size: usize,
        rmsnorm_weight: Vec<f32>,
        eps: f32,
        precision_mask: Vec<u8>,
        epigenetic_database: Vec<u8>,
        epigenetic_centroids: Vec<f32>,
        triplet_database: Vec<u8>,
        triplet_centroids: Vec<f32>,
        bias: Vec<f32>,
        bit_depth: u8,
    ) -> Self {
        GenomicLinear::new(
            database,
            anchors_u8,
            centroids,
            out_features,
            in_features,
            block_size,
            rmsnorm_weight,
            eps,
            precision_mask,
            epigenetic_database,
            epigenetic_centroids,
            triplet_database,
            triplet_centroids,
            bias,
            bit_depth,
        )
    }
    pub fn forward(&self, input: Vec<f32>, activate_rna: bool) -> PyResult<Vec<f32>> {
        self.forward_core(input, None, activate_rna)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    pub fn get_row(&self, idx: usize) -> PyResult<Vec<f32>> {
        self.get_row_core(idx)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    pub fn backward(&self, d_output: Vec<f32>) -> PyResult<Vec<f32>> {
        self.backward_core(d_output)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    pub fn refine_with_grads(&mut self, input: Vec<f32>, grads: Vec<f32>, lr: f32) -> PyResult<()> {
        self.refine_with_grads_core(input, grads, lr)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    pub fn recalibrate_centroids(&mut self, shift: f32) -> PyResult<()> {
        self.recalibrate_centroids_core(shift)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    pub fn apply_vector_equilibrium_alignment(&mut self, strength: f32) -> PyResult<()> {
        self.apply_vector_equilibrium_alignment_core(strength)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[getter]
    pub fn database(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            use pyo3::types::PyBytes;
            Ok(PyBytes::new(py, self.database_ref()).into())
        })
    }
    #[getter]
    pub fn centroids(&self) -> PyResult<Vec<f32>> {
        Ok(self.centroids.clone())
    }
}
