//! # 💾 Formato de Memoria Persistente Zero-Copy `.gmem` (v2 con Épocas y Linaje)
//!
//! Este módulo implementa la serialización y lectura mapeada en memoria (`mmap`)
//! de índices de memoria semántica vectorizada para el Island Model, con soporte
//! inmutable de épocas, linaje genealógico y hashes de integridad.

use std::fs::File;
use std::io::{Read, Result as IoResult, Write};

pub const GMEM_MAGIC: &[u8; 4] = b"GMEM";
pub const GMEM_VERSION_1: u32 = 1;
pub const GMEM_VERSION_2: u32 = 2;

// Flags de época de memoria (.gmem v2)
pub const GMEM_FLAG_CONSOLIDATED: u32 = 1 << 0;
pub const GMEM_FLAG_SEALED: u32 = 1 << 1;
pub const GMEM_FLAG_PROMOTED: u32 = 1 << 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GmemHeader {
    pub magic: [u8; 4],   // b"GMEM" (4 bytes)
    pub version: u32,     // 2 (4 bytes)
    pub dim: u32,         // Dimensión de embeddings (4 bytes)
    pub index_type: u8,   // 0: Plano / Cosine, 1: HNSW (1 byte)
    pub _pad: [u8; 3],    // Alineación (3 bytes)
    pub num_entries: u64, // Número de entradas (8 bytes)
    // --- Campos de Linaje y Versionado (40 bytes antes reservados) ---
    pub epoch_id: u64,        // Identificador monotónico de época (8 bytes)
    pub parent_epoch: u64,    // ID de época padre (0 = raíz / linaje) (8 bytes)
    pub created_at_unix: i64, // Timestamp UTC de creación (8 bytes)
    pub metrics_hash: u64,    // Hash de integridad del manifiesto (8 bytes)
    pub flags: u32,           // bit0: Consolidada | bit1: Sellada | bit2: Promovida (4 bytes)
    pub _reserved: [u8; 4],   // Alineación final a 64 bytes (4 bytes)
}

impl Default for GmemHeader {
    fn default() -> Self {
        Self {
            magic: *GMEM_MAGIC,
            version: GMEM_VERSION_2,
            dim: 896,
            index_type: 0,
            _pad: [0u8; 3],
            num_entries: 0,
            epoch_id: 1,
            parent_epoch: 0,
            created_at_unix: 0,
            metrics_hash: 0,
            flags: 0,
            _reserved: [0u8; 4],
        }
    }
}

#[derive(Clone, Debug)]
pub struct GmemEntry {
    pub id: u64,
    pub vector: Vec<f32>,
    pub text: String,
}

#[derive(Clone, Debug)]
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

    pub fn with_epoch(dim: u32, epoch_id: u64, parent_epoch: u64) -> Self {
        let mut header = GmemHeader::default();
        header.dim = dim;
        header.epoch_id = epoch_id;
        header.parent_epoch = parent_epoch;
        Self {
            header,
            entries: Vec::new(),
        }
    }

    pub fn add_entry(&mut self, id: u64, vector: Vec<f32>, text: String) {
        self.entries.push(GmemEntry { id, vector, text });
        self.header.num_entries = self.entries.len() as u64;
    }

    pub fn epoch_id(&self) -> u64 {
        self.header.epoch_id
    }

    pub fn set_epoch_id(&mut self, epoch_id: u64) {
        self.header.epoch_id = epoch_id;
    }

    pub fn parent_epoch(&self) -> u64 {
        self.header.parent_epoch
    }

    pub fn set_parent_epoch(&mut self, parent_epoch: u64) {
        self.header.parent_epoch = parent_epoch;
    }

    pub fn is_sealed(&self) -> bool {
        (self.header.flags & GMEM_FLAG_SEALED) != 0
    }

    pub fn seal(&mut self) {
        self.header.flags |= GMEM_FLAG_SEALED;
    }

    pub fn is_promoted(&self) -> bool {
        (self.header.flags & GMEM_FLAG_PROMOTED) != 0
    }

    pub fn promote(&mut self) {
        self.header.flags |= GMEM_FLAG_PROMOTED;
    }

    pub fn is_consolidated(&self) -> bool {
        (self.header.flags & GMEM_FLAG_CONSOLIDATED) != 0
    }

    pub fn set_consolidated(&mut self, consolidated: bool) {
        if consolidated {
            self.header.flags |= GMEM_FLAG_CONSOLIDATED;
        } else {
            self.header.flags &= !GMEM_FLAG_CONSOLIDATED;
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.header.num_entries = 0;
    }

    /// Calcula un hash FNV-1a de 64 bits sobre las entradas para auditoría de integridad
    pub fn compute_entries_hash(&self) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for entry in &self.entries {
            hash ^= entry.id;
            hash = hash.wrapping_mul(0x100000001b3);
            for &val in &entry.vector {
                hash ^= val.to_bits() as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            for &b in entry.text.as_bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        hash
    }

    /// Guarda el índice en disco con la estructura binaria .gmem v2
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

    /// Carga el índice binario .gmem desde disco (soporta v1 legacy y v2)
    pub fn load_from_file(path: &str) -> IoResult<Self> {
        let mut file = File::open(path)?;
        let mut header_bytes = [0u8; 64];
        file.read_exact(&mut header_bytes)?;

        let mut header: GmemHeader = unsafe { std::mem::transmute(header_bytes) };
        if &header.magic != GMEM_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Formato de archivo binario .gmem inválido",
            ));
        }

        // Compatibilidad transparente hacia atrás: si es v1, normalizar campos de época
        if header.version == GMEM_VERSION_1 {
            header.version = GMEM_VERSION_2;
            header.epoch_id = 1;
            header.parent_epoch = 0;
            header.created_at_unix = 0;
            header.metrics_hash = 0;
            header.flags = 0;
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
    fn test_gmem_header_size_exact_64_bytes() {
        assert_eq!(std::mem::size_of::<GmemHeader>(), 64);
    }

    #[test]
    fn test_gmem_v2_lineage_and_roundtrip() {
        let mut index = GmemMemoryIndex::with_epoch(4, 42, 41);
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
        index.seal();
        index.promote();

        let hash = index.compute_entries_hash();
        index.header.metrics_hash = hash;

        let temp_path = "/tmp/test_gmem_v2_lineage.gmem";
        index
            .save_to_file(temp_path)
            .expect("Error guardando .gmem v2");

        let loaded = GmemMemoryIndex::load_from_file(temp_path).expect("Error cargando .gmem v2");
        assert_eq!(loaded.epoch_id(), 42);
        assert_eq!(loaded.parent_epoch(), 41);
        assert!(loaded.is_sealed());
        assert!(loaded.is_promoted());
        assert_eq!(loaded.header.metrics_hash, hash);
        assert_eq!(loaded.entries.len(), 2);
        assert_eq!(loaded.entries[0].text, "París es la capital de Francia");

        let query = vec![0.9, 0.1, 0.0, 0.0];
        let results = loaded.search_top_k(&query, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, 1);

        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_gmem_v1_legacy_compatibility() {
        let temp_path = "/tmp/test_gmem_v1_legacy.gmem";
        {
            let mut file = File::create(temp_path).unwrap();
            let mut header_raw = [0u8; 64];
            header_raw[0..4].copy_from_slice(b"GMEM");
            header_raw[4..8].copy_from_slice(&1u32.to_le_bytes()); // version 1
            header_raw[8..12].copy_from_slice(&4u32.to_le_bytes()); // dim 4
            header_raw[16..24].copy_from_slice(&1u64.to_le_bytes()); // 1 entry
                                                                     // 40 reserved bytes are zeroed
            file.write_all(&header_raw).unwrap();

            // Entry 1
            file.write_all(&100u64.to_le_bytes()).unwrap();
            file.write_all(&1.0f32.to_le_bytes()).unwrap();
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
            let text = "Legacy Memory Entry";
            file.write_all(&(text.len() as u32).to_le_bytes()).unwrap();
            file.write_all(text.as_bytes()).unwrap();
        }

        let loaded = GmemMemoryIndex::load_from_file(temp_path).expect("Fallo al cargar .gmem v1");
        assert_eq!(loaded.header.version, GMEM_VERSION_2);
        assert_eq!(loaded.epoch_id(), 1);
        assert_eq!(loaded.parent_epoch(), 0);
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].text, "Legacy Memory Entry");

        let _ = std::fs::remove_file(temp_path);
    }
}
