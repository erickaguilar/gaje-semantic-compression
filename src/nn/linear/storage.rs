// =============================================================================
// storage — WeightStorage y trait GenomicOperable
// =============================================================================
use std::sync::Arc;

use crate::io::header::{Q2_0Block, Q4_0Block, Q8_0Block};

/// Almacenamiento y buffer de pesos en memoria para capas lineales genómicas.
#[derive(Clone)]
pub enum WeightStorage {
    Genomic2Bit(Arc<Vec<u8>>),
    Genomic4Bit(Arc<Vec<u8>>),
    GenomicQ4_0(Arc<Vec<Q4_0Block>>),
    GenomicQ8_0(Arc<Vec<Q8_0Block>>),
    GenomicQ2_0(Arc<Vec<Q2_0Block>>),
    GenomicF32(Arc<Vec<f32>>),
}

/// Alias retrocompatible para WeightStorage.
pub type WeightDatabase = WeightStorage;

pub trait GenomicOperable {
    fn bit_depth(&self) -> u8;
    fn read(&self, byte_idx: usize, sub_idx: usize) -> u8;
    fn mutate(&mut self, byte_idx: usize, sub_idx: usize, new_bits: u8);
    fn len_bytes(&self) -> usize;
}

impl GenomicOperable for WeightStorage {
    fn bit_depth(&self) -> u8 {
        match self {
            WeightStorage::Genomic2Bit(_) => 2,
            WeightStorage::Genomic4Bit(_) => 4,
            WeightStorage::GenomicQ4_0(_) => 4,
            WeightStorage::GenomicQ8_0(_) => 8,
            WeightStorage::GenomicQ2_0(_) => 2,
            WeightStorage::GenomicF32(_) => 32,
        }
    }
    fn read(&self, byte_idx: usize, sub_idx: usize) -> u8 {
        match self {
            WeightStorage::Genomic2Bit(db) => (db[byte_idx] >> ((3 - sub_idx) * 2)) & 0b11,
            WeightStorage::Genomic4Bit(db) => {
                if sub_idx == 0 {
                    db[byte_idx] >> 4
                } else {
                    db[byte_idx] & 0x0F
                }
            }
            WeightStorage::GenomicQ4_0(db) => {
                let block_idx = byte_idx / 16;
                let qs_idx = byte_idx % 16;
                if let Some(block) = db.get(block_idx) {
                    let byte = block.qs[qs_idx];
                    if sub_idx == 0 {
                        byte & 0x0F
                    } else {
                        byte >> 4
                    }
                } else {
                    0
                }
            }
            WeightStorage::GenomicQ8_0(db) => {
                let block_idx = byte_idx / 32;
                let qs_idx = byte_idx % 32;
                if let Some(block) = db.get(block_idx) {
                    block.qs[qs_idx] as u8
                } else {
                    0
                }
            }
            WeightStorage::GenomicQ2_0(db) => {
                let block_idx = byte_idx / 8;
                let qs_idx = byte_idx % 8;
                if let Some(block) = db.get(block_idx) {
                    let byte = block.qs[qs_idx];
                    (byte >> ((3 - sub_idx) * 2)) & 0b11
                } else {
                    0
                }
            }
            _ => 0,
        }
    }
    fn mutate(&mut self, byte_idx: usize, sub_idx: usize, new_bits: u8) {
        match self {
            WeightStorage::Genomic2Bit(ref mut db) => {
                let db_mut = Arc::make_mut(db);
                let shift = (3 - sub_idx) * 2;
                db_mut[byte_idx] &= !(0b11 << shift);
                db_mut[byte_idx] |= (new_bits & 0b11) << shift;
            }
            WeightStorage::Genomic4Bit(ref mut db) => {
                let db_mut = Arc::make_mut(db);
                if sub_idx == 0 {
                    db_mut[byte_idx] = (db_mut[byte_idx] & 0x0F) | (new_bits << 4);
                } else {
                    db_mut[byte_idx] = (db_mut[byte_idx] & 0xF0) | (new_bits & 0x0F);
                }
            }
            WeightStorage::GenomicQ4_0(ref mut db) => {
                let db_mut = Arc::make_mut(db);
                let block_idx = byte_idx / 16;
                let qs_idx = byte_idx % 16;
                if let Some(block) = db_mut.get_mut(block_idx) {
                    if sub_idx == 0 {
                        block.qs[qs_idx] = (block.qs[qs_idx] & 0xF0) | (new_bits & 0x0F);
                    } else {
                        block.qs[qs_idx] = (block.qs[qs_idx] & 0x0F) | ((new_bits & 0x0F) << 4);
                    }
                }
            }
            _ => {}
        }
    }
    fn len_bytes(&self) -> usize {
        match self {
            WeightStorage::Genomic2Bit(db) => db.len(),
            WeightStorage::Genomic4Bit(db) => db.len(),
            WeightStorage::GenomicQ4_0(db) => db.len() * 16,
            WeightStorage::GenomicQ8_0(db) => db.len() * 32,
            WeightStorage::GenomicQ2_0(db) => db.len() * 12,
            WeightStorage::GenomicF32(db) => db.len() * 4,
        }
    }
}
