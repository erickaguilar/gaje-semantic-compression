//! # 🧬 Quantum Codebook Native (Rust SIMD-Ready Embedding Decompressor)
//!
//! Descomprime tablas de embeddings .qemb al vuelo en memoria mediante
//! superposición cuántica dispersa de m=4 meta-tokens.

pub const QEMB_MAGIC: &[u8; 4] = b"QEMB";
pub const QEMB_VERSION: u16 = 1;

#[derive(Clone, Debug)]
pub struct QuantumCodebookNative {
    pub num_meta_tokens: usize,
    pub dim: usize,
    pub centroids: Vec<f32>, // K * dim
}

impl QuantumCodebookNative {
    pub fn new(num_meta_tokens: usize, dim: usize, centroids: Vec<f32>) -> Self {
        Self {
            num_meta_tokens,
            dim,
            centroids,
        }
    }

    pub fn get_centroid(&self, idx: usize) -> &[f32] {
        let start = idx * self.dim;
        &self.centroids[start..start + self.dim]
    }
}

#[derive(Clone, Debug)]
pub struct QuantumEmbeddingTableNative {
    pub codebook: QuantumCodebookNative,
    pub num_tokens: usize,
    pub m: usize,
    pub indices: Vec<u16>,   // num_tokens * m
    pub amplitudes: Vec<u8>, // num_tokens * m (quantized [0..255])
}

impl QuantumEmbeddingTableNative {
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 64 {
            return Err("Archivo .qemb demasiado corto para cabecera de 64 bytes".into());
        }

        if &data[0..4] != QEMB_MAGIC {
            return Err(format!("Magic bytes inválidos: {:?}", &data[0..4]));
        }

        let version = u16::from_le_bytes(data[4..6].try_into().unwrap());
        if version != QEMB_VERSION {
            return Err(format!("Versión incompatible de .qemb: {}", version));
        }

        let m = u16::from_le_bytes(data[6..8].try_into().unwrap()) as usize;
        let num_meta_tokens = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let num_tokens = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;

        let centroids_bytes_len = num_meta_tokens * dim * 4;
        let indices_bytes_len = num_tokens * m * 2;
        let amplitudes_bytes_len = num_tokens * m;

        let total_expected = 64 + centroids_bytes_len + indices_bytes_len + amplitudes_bytes_len;
        if data.len() < total_expected {
            return Err(format!(
                "Archivo .qemb truncado: se esperaban {} bytes, recibidos {}",
                total_expected,
                data.len()
            ));
        }

        // 1. Centroids
        let mut offset = 64;
        let centroids_slice = &data[offset..offset + centroids_bytes_len];
        let mut centroids = Vec::with_capacity(num_meta_tokens * dim);
        for chunk in centroids_slice.chunks_exact(4) {
            centroids.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        offset += centroids_bytes_len;

        // 2. Indices
        let indices_slice = &data[offset..offset + indices_bytes_len];
        let mut indices = Vec::with_capacity(num_tokens * m);
        for chunk in indices_slice.chunks_exact(2) {
            indices.push(u16::from_le_bytes(chunk.try_into().unwrap()));
        }
        offset += indices_bytes_len;

        // 3. Amplitudes
        let amplitudes = data[offset..offset + amplitudes_bytes_len].to_vec();

        let codebook = QuantumCodebookNative::new(num_meta_tokens, dim, centroids);

        Ok(Self {
            codebook,
            num_tokens,
            m,
            indices,
            amplitudes,
        })
    }

    /// Descomprime y llena el vector `out` con el embedding reconstruido del token en O(m * dim)
    pub fn get_embedding(&self, token_id: usize, out: &mut [f32]) {
        let tid = if token_id < self.num_tokens {
            token_id
        } else {
            0
        };
        let dim = self.codebook.dim;
        assert_eq!(
            out.len(),
            dim,
            "El buffer de salida debe coincidir con la dimensión"
        );

        // Inicializar en 0
        for x in out.iter_mut() {
            *x = 0.0;
        }

        let base_idx = tid * self.m;
        for j in 0..self.m {
            let meta_idx = self.indices[base_idx + j] as usize;
            let amp = (self.amplitudes[base_idx + j] as f32) / 255.0;

            if amp > 1e-6 {
                let centroid = self.codebook.get_centroid(meta_idx);
                // Acumulación SIMD/FMA
                for d in 0..dim {
                    out[d] += amp * centroid[d];
                }
            }
        }

        // Normalización final
        let mut norm_sq = 0.0;
        for x in out.iter() {
            norm_sq += x * x;
        }
        if norm_sq > 1e-9 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for x in out.iter_mut() {
                *x *= inv_norm;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qemb_native_reconstruction() {
        let k = 4;
        let dim = 8;
        let num_tokens = 2;
        let m = 2;

        // Codebook de 4 centroides
        let mut centroids = vec![0.0f32; k * dim];
        centroids[0] = 1.0; // centroid 0: [1, 0, 0, ...]
        centroids[dim + 1] = 1.0; // centroid 1: [0, 1, 0, ...]

        let codebook = QuantumCodebookNative::new(k, dim, centroids);
        let indices = vec![0, 1, 0, 0]; // token 0 usa c0 y c1
        let amplitudes = vec![180, 180, 255, 0]; // aprox 0.707 cada uno

        let table = QuantumEmbeddingTableNative {
            codebook,
            num_tokens,
            m,
            indices,
            amplitudes,
        };

        let mut out = vec![0.0f32; dim];
        table.get_embedding(0, &mut out);

        // out debe tener valores en componentes 0 y 1
        assert!(out[0] > 0.5);
        assert!(out[1] > 0.5);
        let norm = (out[0] * out[0] + out[1] * out[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }
}
