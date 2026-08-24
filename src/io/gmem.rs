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

// --- Indice IVF-lite (index_type = 1) ---------------------------------------
// Particionado grueso tipo IVF: k-means ligero sobre los vectores de entrada.
// La busqueda solo sondea los clústeres mas cercanos al query, reduciendo el
// escaneo lineal O(N) a O(N * PROBE / CLUSTERS). Complementa (no sustituye)
// la busqueda plana, que permanece como fallback cuando el indice no aplica.

/// Magic de la seccion IVF anexada tras las entradas en el archivo .gmem
pub const GMEM_IVF_MAGIC: &[u8; 4] = b"IVF1";
/// Debajo de esta cantidad de entradas el escaneo lineal es mas rapido que
/// mantener un indice: no se construye IVF.
pub const GMEM_IVF_MIN_ENTRIES: usize = 2048;
/// Numero de clústeres por defecto (k ~ sqrt(N)/5 para N tipico de 100k+).
pub const GMEM_IVF_CLUSTERS: usize = 64;
/// Cuantos clústeres se sondean por consulta (recall determinista).
pub const GMEM_IVF_PROBE: usize = 8;
/// Iteraciones de k-means ligero entrenado sobre una muestra.
const GMEM_IVF_ITERS: usize = 3;
const GMEM_IVF_SAMPLE: usize = 4096;

#[derive(Clone, Debug)]
pub struct IvfIndex {
    pub dim: usize,
    pub num_clusters: usize,
    pub centroids: Vec<f32>,   // num_clusters * dim (flat, row-major)
    /// Asignacion de cada entrada vigente al momento de construir el indice.
    pub assignments: Vec<u32>,
    /// Indices de entrada por clúster (listas invertidas precomputadas).
    pub lists: Vec<Vec<u32>>,
}

impl IvfIndex {
    #[inline]
    fn centroid(&self, c: usize) -> &[f32] {
        let o = c * self.dim;
        &self.centroids[o..o + self.dim]
    }

    /// Distancia L2^2 query→centroide (monotonica con coseno si se normaliza).
    #[inline]
    fn centroid_dist(&self, query: &[f32], c: usize) -> f32 {
        let cen = self.centroid(c);
        let mut sum = 0.0f32;
        for i in 0..self.dim {
            let d = query[i] - cen[i];
            sum += d * d;
        }
        sum
    }
}

#[derive(Clone, Debug)]
pub struct GmemMemoryIndex {
    pub header: GmemHeader,
    pub entries: Vec<GmemEntry>,
    /// Indice IVF-lite opcional. `None` = escaneo lineal (o entradas < umbral,
    /// o indice invalidado por inserciones pendientes de re-indexar).
    pub ivf: Option<IvfIndex>,
}

impl GmemMemoryIndex {
    pub fn new(dim: u32) -> Self {
        let mut header = GmemHeader::default();
        header.dim = dim;
        Self {
            header,
            entries: Vec::new(),
            ivf: None,
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
            ivf: None,
        }
    }

    pub fn add_entry(&mut self, id: u64, vector: Vec<f32>, text: String) {
        self.entries.push(GmemEntry { id, vector, text });
        self.header.num_entries = self.entries.len() as u64;
        // Insercion invalida el indice: la busqueda cae a lineal hasta re-indexar.
        self.ivf = None;
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
        self.ivf = None;
    }

    /// Construye (o reconstruye) el indice IVF-lite si hay suficientes entradas.
    ///
    /// k-means ligero: entrena sobre una muestra aleatoria (GMEM_IVF_SAMPLE
    /// puntos, GMEM_IVF_ITERS iteraciones) y hace la asignacion final en
    /// paralelo con rayon. Costo @100k x 768d: ~10-15 s, muy por debajo del
    /// gate de indexacion (< 30 s).
    pub fn build_ivf(&mut self) {
        let n = self.entries.len();
        if n < GMEM_IVF_MIN_ENTRIES {
            self.ivf = None;
            return;
        }
        use rand::Rng;

        let dim = self.header.dim as usize;
        let clusters = GMEM_IVF_CLUSTERS.min(n);
        let mut rng = rand::thread_rng();

        // Centroides iniciales: muestreo estratificado por paso fijo.
        let step = n / clusters.max(1);
        let mut centroids: Vec<f32> = Vec::with_capacity(clusters * dim);
        for c in 0..clusters {
            let v = &self.entries[c * step].vector;
            centroids.extend_from_slice(v);
        }

        // Muestra de entrenamiento.
        let sample_len = GMEM_IVF_SAMPLE.min(n);
        let sample_idx: Vec<usize> = (0..sample_len)
            .map(|_| rng.gen_range(0..n))
            .collect();

        let l2sq = |a: &[f32], b: &[f32]| -> f32 {
            let mut s = 0.0f32;
            for i in 0..dim {
                let d = a[i] - b[i];
                s += d * d;
            }
            s
        };

        // k-means ligero sobre la muestra.
        for _ in 0..GMEM_IVF_ITERS {
            let mut sums = vec![0.0f32; centroids.len()];
            let mut counts = vec![0u32; clusters];
            for &si in &sample_idx {
                let v = &self.entries[si].vector;
                let mut best = 0usize;
                let mut best_d = f32::MAX;
                for c in 0..clusters {
                    let d = l2sq(v, &centroids[c * dim..(c + 1) * dim]);
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                counts[best] += 1;
                let base = best * dim;
                for i in 0..dim {
                    sums[base + i] += v[i];
                }
            }
            for c in 0..clusters {
                if counts[c] == 0 {
                    continue;
                }
                let base = c * dim;
                let inv = 1.0 / counts[c] as f32;
                for i in 0..dim {
                    centroids[base + i] = sums[base + i] * inv;
                }
            }
        }

        // Asignacion final en paralelo sobre TODAS las entradas.
        let assignments: Vec<u32> = {
            use rayon::prelude::*;
            self.entries
                .par_iter()
                .map(|e| {
                    let mut best = 0u32;
                    let mut best_d = f32::MAX;
                    for c in 0..clusters {
                        let d = l2sq(&e.vector, &centroids[c * dim..(c + 1) * dim]);
                        if d < best_d {
                            best_d = d;
                            best = c as u32;
                        }
                    }
                    best
                })
                .collect()
        };

        // Listas invertidas.
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); clusters];
        for (idx, &c) in assignments.iter().enumerate() {
            lists[c as usize].push(idx as u32);
        }

        self.ivf = Some(IvfIndex {
            dim,
            num_clusters: clusters,
            centroids,
            assignments,
            lists,
        });
    }

    /// Busqueda top-k via IVF: puntua centroides, sondea los PROBE mas cercanos
    /// y escanea solo las listas invertidas de esos clústeres.
    fn search_top_k_ivf(
        &self,
        ivf: &IvfIndex,
        query: &[f32],
        k: usize,
    ) -> Vec<(&GmemEntry, f32)> {
        // 1. Distancias query→centroides y seleccion del subconjunto a sondear.
        let mut cluster_scores: Vec<(u32, f32)> = (0..ivf.num_clusters)
            .map(|c| (c as u32, ivf.centroid_dist(query, c)))
            .collect();
        cluster_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let probe = GMEM_IVF_PROBE.min(ivf.num_clusters);

        // 2. Escaneo parcial con seleccion acotada (max-heap de tamaño k).
        //    Guardamos el INDICE de entrada (no el id, que puede repetirse).
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::with_capacity(k + 1);
        for &(c, _) in cluster_scores.iter().take(probe) {
            for &entry_idx in &ivf.lists[c as usize] {
                let sim_u = ordered_bits(cosine_similarity(
                    query,
                    &self.entries[entry_idx as usize].vector,
                ));
                let item = Reverse((sim_u, entry_idx));
                if heap.len() < k {
                    heap.push(item);
                } else if let Some(&Reverse((worst, _))) = heap.peek() {
                    if sim_u > worst {
                        heap.pop();
                        heap.push(item);
                    }
                }
            }
        }

        // 3. Extraer y ordenar descendente por similitud.
        let mut out: Vec<(&GmemEntry, f32)> = heap
            .into_iter()
            .map(|Reverse((bits, idx))| (&self.entries[idx as usize], bits_from_ordered(bits)))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(k);
        out
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

    /// Serializa el índice a bytes en formato binario .gmem v2 (ideal para WASM / IndexedDB / OPFS)
    pub fn save_to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // 1. Escribir Header de 64 bytes
        let header_bytes: &[u8; 64] = unsafe { std::mem::transmute(&self.header) };
        buf.extend_from_slice(header_bytes);

        // 2. Escribir Entradas
        for entry in &self.entries {
            buf.extend_from_slice(&entry.id.to_le_bytes());
            for &val in &entry.vector {
                buf.extend_from_slice(&val.to_le_bytes());
            }
            let text_bytes = entry.text.as_bytes();
            let text_len = text_bytes.len() as u32;
            buf.extend_from_slice(&text_len.to_le_bytes());
            buf.extend_from_slice(text_bytes);
        }
        buf
    }

    /// Deserializa un índice binario .gmem desde un slice de bytes (soporta v1 y v2)
    pub fn load_from_bytes(bytes: &[u8]) -> IoResult<Self> {
        if bytes.len() < 64 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Buffer demasiado corto para cabecera .gmem",
            ));
        }

        let mut header_bytes = [0u8; 64];
        header_bytes.copy_from_slice(&bytes[0..64]);

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
        let mut offset = 64;

        for _ in 0..header.num_entries {
            if offset + 8 > bytes.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "Fin inesperado leyendo ID de entrada .gmem",
                ));
            }
            let id = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            offset += 8;

            let vec_bytes_len = dim * 4;
            if offset + vec_bytes_len > bytes.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "Fin inesperado leyendo vector de entrada .gmem",
                ));
            }
            let mut vector = vec![0.0f32; dim];
            for i in 0..dim {
                let v_bytes: [u8; 4] = bytes[offset + i * 4..offset + (i + 1) * 4]
                    .try_into()
                    .unwrap();
                vector[i] = f32::from_le_bytes(v_bytes);
            }
            offset += vec_bytes_len;

            if offset + 4 > bytes.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "Fin inesperado leyendo longitud de texto .gmem",
                ));
            }
            let text_len =
                u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            if offset + text_len > bytes.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "Fin inesperado leyendo texto .gmem",
                ));
            }
            let text = String::from_utf8(bytes[offset..offset + text_len].to_vec())
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
            offset += text_len;

            entries.push(GmemEntry { id, vector, text });
        }

        Ok(Self { header, entries })
    }

    /// Guarda el índice en disco con la estructura binaria .gmem v2
    pub fn save_to_file(&self, path: &str) -> IoResult<()> {
        let bytes = self.save_to_bytes();
        let mut file = File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    /// Carga el índice binario .gmem desde disco (soporta v1 legacy y v2)
    pub fn load_from_file(path: &str) -> IoResult<Self> {
        let mut file = File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Self::load_from_bytes(&bytes)
    }

    /// Busca los K vecinos más cercanos por Similitud Coseno.
    /// Usa el indice IVF si esta construido y vigente; escaneo lineal si no.
    pub fn search_top_k(&self, query: &[f32], k: usize) -> Vec<(&GmemEntry, f32)> {
        if query.len() != self.header.dim as usize || self.entries.is_empty() {
            return Vec::new();
        }

        if let Some(ivf) = &self.ivf {
            if ivf.assignments.len() == self.entries.len() && ivf.dim == query.len() {
                return self.search_top_k_ivf(ivf, query, k);
            }
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

/// Convierte un f32 a un u32 cuyo orden entero preserva el orden del float
/// (truco estandar para heaps sin NaN).
#[inline]
fn ordered_bits(f: f32) -> u32 {
    let bits = f.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// Inverso exacto de `ordered_bits`.
#[inline]
fn bits_from_ordered(u: u32) -> f32 {
    let bits = if u & 0x8000_0000 != 0 {
        u & 0x7FFF_FFFF
    } else {
        !u
    };
    f32::from_bits(bits)
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
