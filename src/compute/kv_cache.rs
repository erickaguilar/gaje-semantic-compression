/// Optimized Compressed KV Cache for the GAJE Island Model.
/// Uses a 2-bit interleaved layout: 48 values per 16-byte block (including scale).
/// Designed for SIMD-friendly access and high cache locality.

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct CacheBlock {
    pub data: [u8; 12], // 48 values * 2 bits = 96 bits
    pub scale: f32,     // Shared scale for these 48 values
}

impl Default for CacheBlock {
    fn default() -> Self {
        Self {
            data: [0; 12],
            scale: 1.0,
        }
    }
}

/// A compressed cache for a specific semantic niche (island).
#[cfg_attr(feature = "python", pyclass)]
pub struct CompressedKVCache {
    pub blocks: Vec<CacheBlock>,
    pub num_values: usize,
    pub island_id: u32,
}

impl CompressedKVCache {
    pub fn new(island_id: u32, capacity: usize) -> Self {
        let num_blocks = (capacity + 47) / 48;
        Self {
            blocks: vec![CacheBlock::default(); num_blocks],
            num_values: capacity,
            island_id,
        }
    }

    /// Packs a slice of f32 into 2-bit values with a calculated scale.
    /// Differential approach: each block of 48 gets its own scale.
    pub fn pack(&mut self, values: &[f32]) {
        for (i, chunk) in values.chunks(48).enumerate() {
            if i >= self.blocks.len() {
                break;
            }

            // 1. Find max absolute value for scaling
            let mut max_val = 0.0f32;
            for &v in chunk {
                max_val = max_val.max(v.abs());
            }

            let scale = if max_val > 0.0 { max_val / 3.0 } else { 1.0 };
            self.blocks[i].scale = scale;

            // 2. Pack 48 values into 12 bytes
            let mut block_data = [0u8; 12];
            for (j, &v) in chunk.iter().enumerate() {
                let quantized = (v / scale).round().clamp(0.0, 3.0) as u8;
                let byte_idx = j / 4;
                let bit_shift = (j % 4) * 2;
                block_data[byte_idx] |= quantized << bit_shift;
            }
            self.blocks[i].data = block_data;
        }
    }

    /// Unpacks a single value (Lazy Dequantization).
    pub fn get_value(&self, idx: usize) -> f32 {
        let block_idx = idx / 48;
        let sub_idx = idx % 48;
        if block_idx >= self.blocks.len() {
            return 0.0;
        }

        let block = &self.blocks[block_idx];
        let byte_idx = sub_idx / 4;
        let bit_shift = (sub_idx % 4) * 2;
        let quantized = (block.data[byte_idx] >> bit_shift) & 0b11;

        (quantized as f32) * block.scale
    }
}

#[cfg_attr(feature = "python", pymethods)]
impl CompressedKVCache {
    #[cfg(feature = "python")]
    #[new]
    pub fn new_py(island_id: u32, capacity: usize) -> Self {
        Self::new(island_id, capacity)
    }

    #[cfg(feature = "python")]
    #[pyo3(name = "pack")]
    pub fn pack_py(&mut self, values: Vec<f32>) {
        self.pack(&values);
    }

    #[cfg(feature = "python")]
    #[pyo3(name = "get_value")]
    pub fn get_value_py(&self, idx: usize) -> f32 {
        self.get_value(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_compression_accuracy() {
        let mut cache = CompressedKVCache::new(1, 48);
        let original = vec![0.0, 0.5, 1.2, 2.8, 3.0, 0.1, 1.5, 2.2];
        let mut full_chunk = vec![0.0; 48];
        for (i, &v) in original.iter().enumerate() {
            full_chunk[i] = v;
        }

        cache.pack(&full_chunk);

        for (i, &v) in original.iter().enumerate() {
            let recovered = cache.get_value(i);
            let diff = (v - recovered).abs();
            assert!(
                diff <= 0.51,
                "Error de cuantización demasiado alto en índice {}: original {}, recuperado {}",
                i,
                v,
                recovered
            );
        }
    }
}
