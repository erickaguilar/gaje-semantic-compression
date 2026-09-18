use std::sync::Arc;

use crate::io::flat_reader::FlatBufferSource;
use crate::io::header::{Q2_0Block, Q4_0Block, Q8_0Block};

/// Rebanada de pesos respaldada directamente por el buffer mmap o memoria viva sin copia al heap.
#[derive(Clone)]
pub struct WeightSlice<T> {
    pub owner: FlatBufferSource,
    pub ptr: *const T,
    pub len: usize,
}

unsafe impl<T: Send> Send for WeightSlice<T> {}
unsafe impl<T: Sync> Sync for WeightSlice<T> {}

impl<T> WeightSlice<T> {
    #[inline(always)]
    pub fn new(owner: FlatBufferSource, ptr: *const T, len: usize) -> Self {
        Self { owner, ptr, len }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

impl<T> std::ops::Deref for WeightSlice<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsRef<[T]> for WeightSlice<T> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

/// Buffer de pesos polimórfico: puede ser un vector en Heap (Arc<Vec<T>>) o una rebanada zero-copy (WeightSlice<T>).
#[derive(Clone)]
pub enum WeightBuffer<T> {
    Owned(Arc<Vec<T>>),
    Slice(WeightSlice<T>),
}

impl<T> std::ops::Deref for WeightBuffer<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        match self {
            WeightBuffer::Owned(vec) => &vec[..],
            WeightBuffer::Slice(slice) => slice.as_slice(),
        }
    }
}

impl<T> AsRef<[T]> for WeightBuffer<T> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        match self {
            WeightBuffer::Owned(vec) => &vec[..],
            WeightBuffer::Slice(slice) => slice.as_slice(),
        }
    }
}

impl<T> From<Arc<Vec<T>>> for WeightBuffer<T> {
    #[inline(always)]
    fn from(v: Arc<Vec<T>>) -> Self {
        WeightBuffer::Owned(v)
    }
}

impl<T> From<Vec<T>> for WeightBuffer<T> {
    #[inline(always)]
    fn from(v: Vec<T>) -> Self {
        WeightBuffer::Owned(Arc::new(v))
    }
}

impl<T: PartialEq> PartialEq for WeightBuffer<T> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl<'a, T> IntoIterator for &'a WeightBuffer<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.as_ref().iter()
    }
}

impl<T: 'static> WeightBuffer<T> {
    #[inline(always)]
    pub fn from_slice(owner: FlatBufferSource, ptr: *const T, len: usize) -> Self {
        WeightBuffer::Slice(WeightSlice::new(owner, ptr, len))
    }
}

impl<T: Clone> WeightBuffer<T> {
    pub fn make_mut(&mut self) -> &mut Vec<T> {
        match self {
            WeightBuffer::Owned(ref mut arc) => Arc::make_mut(arc),
            WeightBuffer::Slice(slice) => {
                let vec = slice.as_slice().to_vec();
                *self = WeightBuffer::Owned(Arc::new(vec));
                match self {
                    WeightBuffer::Owned(ref mut arc) => Arc::make_mut(arc),
                    _ => unreachable!(),
                }
            }
        }
    }
}

/// Almacenamiento y buffer de pesos en memoria para capas lineales genómicas.
#[derive(Clone)]
pub enum WeightStorage {
    Genomic2Bit(WeightBuffer<u8>),
    Genomic4Bit(WeightBuffer<u8>),
    GenomicQ4_0(WeightBuffer<Q4_0Block>),
    GenomicQ8_0(WeightBuffer<Q8_0Block>),
    GenomicQ2_0(WeightBuffer<Q2_0Block>),
    GenomicF32(WeightBuffer<f32>),
}

impl WeightStorage {
    #[inline(always)]
    pub fn from_q4_0_slice(owner: FlatBufferSource, ptr: *const Q4_0Block, len: usize) -> Self {
        WeightStorage::GenomicQ4_0(WeightBuffer::Slice(WeightSlice::new(owner, ptr, len)))
    }
    #[inline(always)]
    pub fn from_q8_0_slice(owner: FlatBufferSource, ptr: *const Q8_0Block, len: usize) -> Self {
        WeightStorage::GenomicQ8_0(WeightBuffer::Slice(WeightSlice::new(owner, ptr, len)))
    }
    #[inline(always)]
    pub fn from_q2_0_slice(owner: FlatBufferSource, ptr: *const Q2_0Block, len: usize) -> Self {
        WeightStorage::GenomicQ2_0(WeightBuffer::Slice(WeightSlice::new(owner, ptr, len)))
    }
    #[inline(always)]
    pub fn from_4bit_slice(owner: FlatBufferSource, ptr: *const u8, len: usize) -> Self {
        WeightStorage::Genomic4Bit(WeightBuffer::Slice(WeightSlice::new(owner, ptr, len)))
    }
    #[inline(always)]
    pub fn from_2bit_slice(owner: FlatBufferSource, ptr: *const u8, len: usize) -> Self {
        WeightStorage::Genomic2Bit(WeightBuffer::Slice(WeightSlice::new(owner, ptr, len)))
    }
    #[inline(always)]
    pub fn from_f32_slice(owner: FlatBufferSource, ptr: *const f32, len: usize) -> Self {
        WeightStorage::GenomicF32(WeightBuffer::Slice(WeightSlice::new(owner, ptr, len)))
    }
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
                let db_mut = db.make_mut();
                let shift = (3 - sub_idx) * 2;
                db_mut[byte_idx] &= !(0b11 << shift);
                db_mut[byte_idx] |= (new_bits & 0b11) << shift;
            }
            WeightStorage::Genomic4Bit(ref mut db) => {
                let db_mut = db.make_mut();
                if sub_idx == 0 {
                    db_mut[byte_idx] = (db_mut[byte_idx] & 0x0F) | (new_bits << 4);
                } else {
                    db_mut[byte_idx] = (db_mut[byte_idx] & 0xF0) | (new_bits & 0x0F);
                }
            }
            WeightStorage::GenomicQ4_0(ref mut db) => {
                let db_mut = db.make_mut();
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
