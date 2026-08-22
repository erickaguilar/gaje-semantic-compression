//! GAJE Helix — Native Binary Tokenizer Parser (GTOK v1.0).
//!
//! High-performance, zero-external-dependency BPE tokenizer implementation in pure Rust std.
//! Directly parses contiguous binary `.gtok` files with sub-millisecond cold-start.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

#[cfg(feature = "python")]
use pyo3::prelude::*;

pub const GTOK_MAGIC: &[u8; 4] = b"GTOK";
pub const GTOK_VERSION: u16 = 1;

#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Debug)]
pub struct GtokNativeTokenizer {
    pub vocab: Vec<String>,
    pub token_to_id: HashMap<String, u32>,
    pub merges: Vec<(u32, u32, u32)>, // Tripletas (left, right, target)
    pub merges_map: HashMap<(u32, u32), u32>,
    pub bos_id: u32,
    pub eos_id: u32,
    pub unk_id: u32,
    pub pad_id: u32,
    pub extra_stop_ids: Vec<u32>,
    pub version: u16,
    pub flags: u16,
}

#[cfg_attr(feature = "python", pymethods)]
impl GtokNativeTokenizer {
    #[cfg(feature = "python")]
    #[staticmethod]
    pub fn py_from_file(path: &str) -> PyResult<Self> {
        Self::from_file(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))
    }

    #[cfg(feature = "python")]
    #[staticmethod]
    pub fn py_from_bytes(bytes: &[u8]) -> PyResult<Self> {
        Self::from_bytes(bytes).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    #[cfg(feature = "python")]
    #[pyo3(name = "encode")]
    pub fn py_encode(&self, text: &str) -> Vec<u32> {
        self.encode(text)
    }

    #[cfg(feature = "python")]
    #[pyo3(name = "decode")]
    pub fn py_decode(&self, token_ids: Vec<u32>) -> String {
        self.decode(&token_ids)
    }

    #[cfg(feature = "python")]
    #[pyo3(name = "vocab_size")]
    pub fn py_vocab_size(&self) -> usize {
        self.vocab_size()
    }

    #[cfg(feature = "python")]
    #[pyo3(name = "get_stop_tokens")]
    pub fn py_get_stop_tokens(&self) -> Vec<u32> {
        self.get_stop_tokens()
    }
}

impl GtokNativeTokenizer {
    /// Carga el tokenizador desde un archivo binario `.gtok`.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let mut file = File::open(path.as_ref())
            .map_err(|e| format!("Error abriendo archivo .gtok: {}", e))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| format!("Error leyendo buffer .gtok: {}", e))?;
        Self::from_bytes(&buffer)
    }

    /// Deserializa el tokenizador desde un slice binario en memoria.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 36 {
            return Err("Buffer demasiado pequeño para cabecera GTOK".to_string());
        }

        // 1. Cabecera (16 bytes)
        if &data[0..4] != GTOK_MAGIC {
            return Err(format!("Firma mágica inválida: esperada {:?}", GTOK_MAGIC));
        }

        let version = u16::from_le_bytes([data[4], data[5]]);
        let flags = u16::from_le_bytes([data[6], data[7]]);
        let vocab_size = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let merges_count = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;

        let mut offset = 16;

        // 2. Tokens especiales (18 bytes iniciales)
        if data.len() < offset + 18 {
            return Err("Buffer truncado en sección de tokens especiales".to_string());
        }

        let bos_id = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        let eos_id = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        let unk_id = u32::from_le_bytes([
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
        ]);
        let pad_id = u32::from_le_bytes([
            data[offset + 12],
            data[offset + 13],
            data[offset + 14],
            data[offset + 15],
        ]);
        let extra_stops_count = u16::from_le_bytes([data[offset + 16], data[offset + 17]]) as usize;
        offset += 18;

        let mut extra_stop_ids = Vec::with_capacity(extra_stops_count);
        for _ in 0..extra_stops_count {
            if data.len() < offset + 4 {
                return Err("Buffer truncado en lista de stop tokens".to_string());
            }
            let sid = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            extra_stop_ids.push(sid);
            offset += 4;
        }

        // 3. String Table (Offsets + UTF-8 Pool)
        let offsets_count = vocab_size + 1;
        if data.len() < offset + offsets_count * 4 {
            return Err("Buffer truncado en tabla de desplazamientos de cadenas".to_string());
        }

        let mut string_offsets = Vec::with_capacity(offsets_count);
        for _ in 0..offsets_count {
            let off = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            string_offsets.push(off);
            offset += 4;
        }

        let pool_size = *string_offsets.last().unwrap_or(&0);
        if data.len() < offset + pool_size {
            return Err("Buffer truncado en el pool de cadenas UTF-8".to_string());
        }

        let string_pool = &data[offset..offset + pool_size];
        offset += pool_size;

        let mut vocab = Vec::with_capacity(vocab_size);
        let mut token_to_id = HashMap::with_capacity(vocab_size);

        for i in 0..vocab_size {
            let start = string_offsets[i];
            let end = string_offsets[i + 1];
            let token_bytes = &string_pool[start..end];
            let token_str = String::from_utf8_lossy(token_bytes).into_owned();
            token_to_id.insert(token_str.clone(), i as u32);
            vocab.push(token_str);
        }

        // 4. Binary Merges Table
        if data.len() < offset + merges_count * 12 {
            return Err("Buffer truncado en tabla de fusiones BPE".to_string());
        }

        let mut merges = Vec::with_capacity(merges_count);
        let mut merges_map = HashMap::with_capacity(merges_count);

        for _ in 0..merges_count {
            let left = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            let right = u32::from_le_bytes([
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            let target = u32::from_le_bytes([
                data[offset + 8],
                data[offset + 9],
                data[offset + 10],
                data[offset + 11],
            ]);
            merges.push((left, right, target));
            merges_map.insert((left, right), target);
            offset += 12;
        }

        Ok(Self {
            vocab,
            token_to_id,
            merges,
            merges_map,
            bos_id,
            eos_id,
            unk_id,
            pad_id,
            extra_stop_ids,
            version,
            flags,
        })
    }

    /// Codifica texto a IDs de tokens usando BPE.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut tokens: Vec<u32> = Vec::with_capacity(text.len());
        for c in text.chars() {
            let s = c.to_string();
            let g_s = format!("Ġ{}", c);
            let sp_s = format!(" {}", c);

            if let Some(&id) = self.token_to_id.get(&s) {
                tokens.push(id);
            } else if let Some(&id) = self.token_to_id.get(&g_s) {
                tokens.push(id);
            } else if let Some(&id) = self.token_to_id.get(&sp_s) {
                tokens.push(id);
            } else {
                tokens.push(self.unk_id);
            }
        }

        // BPE iterative merge loop
        if tokens.len() > 1 && !self.merges_map.is_empty() {
            loop {
                let mut best_pair = None;
                let mut best_idx = 0;

                for i in 0..(tokens.len() - 1) {
                    let pair = (tokens[i], tokens[i + 1]);
                    if let Some(&target) = self.merges_map.get(&pair) {
                        best_pair = Some(target);
                        best_idx = i;
                        break;
                    }
                }

                if let Some(target_id) = best_pair {
                    tokens[best_idx] = target_id;
                    tokens.remove(best_idx + 1);
                } else {
                    break;
                }
            }
        }

        tokens
    }

    /// Decodifica IDs a texto plano UTF-8.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut result = String::new();
        for &id in token_ids {
            if let Some(token_str) = self.vocab.get(id as usize) {
                result.push_str(token_str);
            }
        }
        result.replace("Ġ", " ").replace(" ", " ")
    }

    /// Tamaño del vocabulario.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Retorna todos los IDs de detención de generación.
    pub fn get_stop_tokens(&self) -> Vec<u32> {
        let mut stops = self.extra_stop_ids.clone();
        if !stops.contains(&self.eos_id) {
            stops.push(self.eos_id);
        }
        stops
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gtok_native_roundtrip() {
        let data = vec![
            // Header: "GTOK", v=1, flags=1, vocab=4, merges=1
            b'G', b'T', b'O', b'K', 1, 0, 1, 0, 4, 0, 0, 0, 1, 0, 0, 0,
            // Specials: bos=1, eos=2, unk=0, pad=0, extra=0
            1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            // String offsets: [0, 5, 8, 12, 14]
            0, 0, 0, 0, 5, 0, 0, 0, 8, 0, 0, 0, 12, 0, 0, 0, 14, 0, 0, 0,
            // String pool: "<unk><s></s>AB"
            b'<', b'u', b'n', b'k', b'>', b'<', b's', b'>', b'<', b'/', b's', b'>', b'A', b'B',
            // Merges: left=1, right=2 -> target=3
            1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0,
        ];

        let tokenizer = GtokNativeTokenizer::from_bytes(&data).expect("Error parsing GTOK bytes");
        assert_eq!(tokenizer.vocab_size(), 4);
        assert_eq!(tokenizer.vocab[0], "<unk>");
        assert_eq!(tokenizer.vocab[3], "AB");
        assert_eq!(tokenizer.eos_id, 2);
    }
}
