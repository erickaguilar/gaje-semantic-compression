//! # 💾 Formato de Memoria Persistente Zero-Copy `.gmem`
//!
//! Este módulo implementa la serialización y lectura mapeada en memoria (`mmap`)
//! de índices de memoria semántica vectorizada para el Island Model.

use std::fs::File;
use std::io::{Read, Result as IoResult, Write};

pub const GMEM_MAGIC: &[u8; 4] = b"GMEM";

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GmemHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub dim: u32,
    pub index_type: u8,
    pub _pad: [u8; 3],
    pub num_entries: u64,
    pub reserved: [u8; 40],
}

impl Default for GmemHeader {
    fn default() -> Self {
        Self {
            magic: *GMEM_MAGIC,
            version: 1,
            dim: 896,
            index_type: 0,
            _pad: [0u8; 3],
            num_entries: 0,
            reserved: [0u8; 40],
        }
    }
}

pub struct GmemEntry {
    pub id: u64,
    pub vector: Vec<f32>,
    pub text: String,
}

pub struct GmemMemoryIndex {
    pub header: GmemHeader,
    pub entries: Vec<GmemEntry>,
}

impl GmemMemoryIndex {
    pub fn new(dim: u32) -> Self {
        let mut header = GmemHeader::default();
        header.dim = dim;
        Self {
            header,
            entries: Vec::new(),
        }
    }

    pub fn add_entry(&mut self, id: u64, vector: Vec<f32>, text: String) {
        self.entries.push(GmemEntry { id, vector, text });
        self.header.num_entries = self.entries.len() as u64;
    }

    /// Guarda el índice en disco con la estructura binaria .gmem
    pub fn save_to_file(&self, path: &str) -> IoResult<()> {
        let mut file = File::create(path)?;

        // 1. Escribir Header de 64 bytes
        let header_bytes: &[u8; 64] = unsafe { std::mem::transmute(&self.header) };
        file.write_all(header_bytes)?;

        // 2. Escribir Entradas
        for entry in &self.entries {
            file.write_all(&entry.id.to_le_bytes())?;

            // Escribir vector (dim * 4 bytes)
            for &val in &entry.vector {
                file.write_all(&val.to_le_bytes())?;
            }

            // Escribir longitud y texto
            let text_bytes = entry.text.as_bytes();
            let text_len = text_bytes.len() as u32;
            file.write_all(&text_len.to_le_bytes())?;
            file.write_all(text_bytes)?;
        }

        Ok(())
    }

    /// Carga el índice binario .gmem desde disco
    pub fn load_from_file(path: &str) -> IoResult<Self> {
        let mut file = File::open(path)?;
        let mut header_bytes = [0u8; 64];
        file.read_exact(&mut header_bytes)?;

        let header: GmemHeader = unsafe { std::mem::transmute(header_bytes) };
        if &header.magic != GMEM_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Formato de archivo binario .gmem inválido",
            ));
        }

        let mut entries = Vec::with_capacity(header.num_entries as usize);
        let dim = header.dim as usize;

        for _ in 0..header.num_entries {
            let mut id_bytes = [0u8; 8];
            file.read_exact(&mut id_bytes)?;
            let id = u64::from_le_bytes(id_bytes);

            let mut vector = vec![0.0f32; dim];
            for i in 0..dim {
                let mut val_bytes = [0u8; 4];
                file.read_exact(&mut val_bytes)?;
                vector[i] = f32::from_le_bytes(val_bytes);
            }

            let mut len_bytes = [0u8; 4];
            file.read_exact(&mut len_bytes)?;
            let text_len = u32::from_le_bytes(len_bytes) as usize;

            let mut text_bytes = vec![0u8; text_len];
            file.read_exact(&mut text_bytes)?;
            let text = String::from_utf8(text_bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

            entries.push(GmemEntry { id, vector, text });
        }

        Ok(Self { header, entries })
    }

    /// Busca los K vecinos más cercanos por Similitud Coseno vectorizada
    pub fn search_top_k(&self, query: &[f32], k: usize) -> Vec<(&GmemEntry, f32)> {
        if query.len() != self.header.dim as usize || self.entries.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(&GmemEntry, f32)> = self
            .entries
            .iter()
            .map(|entry| {
                let sim = cosine_similarity(query, &entry.vector);
                (entry, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

#[inline(always)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if norm_a <= 1e-6 || norm_b <= 1e-6 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gmem_lifecycle() {
        let mut index = GmemMemoryIndex::new(4);
        index.add_entry(
            1,
            vec![1.0, 0.0, 0.0, 0.0],
            "París es la capital de Francia".to_string(),
        );
        index.add_entry(
            2,
            vec![0.0, 1.0, 0.0, 0.0],
            "El sol es una estrella".to_string(),
        );

        let query = vec![0.9, 0.1, 0.0, 0.0];
        let results = index.search_top_k(&query, 1);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, 1);
        assert!(results[0].1 > 0.9);
    }
}
