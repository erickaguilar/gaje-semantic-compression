// =============================================================================
// database — WeightDatabase y trait GenomicOperable
// =============================================================================
use std::sync::Arc;

use crate::io::header::{Q4_0Block, Q8_0Block};

#[derive(Clone)]
pub enum WeightDatabase {
    Genomic2Bit(Arc<Vec<u8>>),
    Genomic4Bit(Arc<Vec<u8>>),
    GenomicQ4_0(Arc<Vec<Q4_0Block>>),
    GenomicQ8_0(Arc<Vec<Q8_0Block>>),
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
            WeightDatabase::GenomicQ4_0(_) => 4,
            WeightDatabase::GenomicQ8_0(_) => 8,
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
            WeightDatabase::GenomicQ4_0(db) => {
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
            WeightDatabase::GenomicQ8_0(db) => {
                let block_idx = byte_idx / 32;
                let qs_idx = byte_idx % 32;
                if let Some(block) = db.get(block_idx) {
                    block.qs[qs_idx] as u8
                } else {
                    0
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
            WeightDatabase::GenomicQ4_0(db) => db.len() * 16,
            WeightDatabase::GenomicQ8_0(db) => db.len() * 32,
            WeightDatabase::GenomicF32(db) => db.len() * 4,
        }
    }
}