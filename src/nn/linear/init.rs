// =============================================================================
// init — Construcción y acceso crudo a los pesos de GenomicLinear
// =============================================================================
use half::f16;
use std::sync::Arc;

use crate::io::header::{Q4_0Block, Q8_0Block};
use crate::nn::linear::database::WeightDatabase;
use crate::nn::linear::GenomicLinear;

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
            4 => {
                if centroids.is_empty() {
                    let ptr = database.as_ptr() as *const Q4_0Block;
                    let count = database.len() / std::mem::size_of::<Q4_0Block>();
                    let blocks = unsafe { std::slice::from_raw_parts(ptr, count).to_vec() };
                    WeightDatabase::GenomicQ4_0(Arc::new(blocks))
                } else {
                    WeightDatabase::Genomic4Bit(Arc::new(database))
                }
            }
            8 => {
                let ptr = database.as_ptr() as *const Q8_0Block;
                let count = database.len() / std::mem::size_of::<Q8_0Block>();
                let blocks = unsafe { std::slice::from_raw_parts(ptr, count).to_vec() };
                WeightDatabase::GenomicQ8_0(Arc::new(blocks))
            }
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
            8 => block_size,
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
            WeightDatabase::GenomicQ4_0(_) => panic!("Q4_0 is read-only"),
            WeightDatabase::GenomicQ8_0(_) => panic!("Q8_0 is read-only"),
            _ => panic!("N/A"),
        }
    }
    pub fn database_ref(&self) -> &[u8] {
        match &self.weight_db {
            WeightDatabase::Genomic2Bit(db) => db.as_ref(),
            WeightDatabase::Genomic4Bit(db) => db.as_ref(),
            WeightDatabase::GenomicQ4_0(db) => unsafe {
                std::slice::from_raw_parts(
                    db.as_ptr() as *const u8,
                    db.len() * std::mem::size_of::<Q4_0Block>(),
                )
            },
            WeightDatabase::GenomicQ8_0(db) => unsafe {
                std::slice::from_raw_parts(
                    db.as_ptr() as *const u8,
                    db.len() * std::mem::size_of::<Q8_0Block>(),
                )
            },
            WeightDatabase::GenomicF32(db) => unsafe {
                std::slice::from_raw_parts(db.as_ptr() as *const u8, db.len() * 4)
            },
        }
    }
    pub fn bit_depth(&self) -> u8 {
        match &self.weight_db {
            WeightDatabase::Genomic2Bit(_) => 2,
            WeightDatabase::Genomic4Bit(_) => 4,
            WeightDatabase::GenomicQ4_0(_) => 4,
            WeightDatabase::GenomicQ8_0(_) => 8,
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
}