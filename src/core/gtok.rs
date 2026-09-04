//! GAJE Helix — Native Binary Tokenizer Parser (GTOK v1.0).
//!
//! High-performance, zero-external-dependency BPE tokenizer implementation in pure Rust std.
//! Directly parses contiguous binary `.gtok` files with sub-millisecond cold-start.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[cfg(feature = "python")]
use pyo3::prelude::*;

pub const GTOK_MAGIC: &[u8; 4] = b"GTOK";
pub const GTOK_VERSION: u16 = 1;
pub const GTOK_VERSION_V2: u16 = 2;

/// Formato de Tokenizador GTOK
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GtokFormat {
    V1BpeClassic,
    V2GenomicMorphological,
}

/// Bases nucleótidas de 2-bits asociadas a desinencias gramaticales (GTOK v2)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NucleotideBase {
    Adenine = 0,  // 00 (Fase 0°)   - Forma base / Lema neutro
    Cytosine = 1, // 01 (Fase 90°)  - Género / Plural / Concordancia
    Guanine = 2,  // 10 (Fase 180°) - Acción / Conjugación temporal
    Thymine = 3,  // 11 (Fase 270°) - Modificador / Adjetivo / Derivación
}

impl NucleotideBase {
    pub fn from_u8(val: u8) -> Self {
        match val & 0b11 {
            0 => Self::Adenine,
            1 => Self::Cytosine,
            2 => Self::Guanine,
            _ => Self::Thymine,
        }
    }

    pub fn to_char(&self) -> char {
        match self {
            Self::Adenine => 'A',
            Self::Cytosine => 'C',
            Self::Guanine => 'G',
            Self::Thymine => 'T',
        }
    }
}

lazy_static::lazy_static! {
    static ref BYTE_TO_UNICODE: Vec<String> = {
        let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
            .chain(161..=172)
            .chain(174..=255)
            .collect();
        let mut cs = bs.clone();
        let mut n = 0;
        for b in 0..=255 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        let mut v = vec![String::new(); 256];
        for (b, c) in bs.into_iter().zip(cs.into_iter()) {
            if let Some(ch) = char::from_u32(c) {
                v[b as usize] = ch.to_string();
            }
        }
        v
    };
    static ref UNICODE_TO_BYTE: HashMap<char, u8> = {
        let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
            .chain(161..=172)
            .chain(174..=255)
            .collect();
        let mut cs = bs.clone();
        let mut n = 0;
        for b in 0..=255 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        let mut map = HashMap::new();
        for (b, c) in bs.into_iter().zip(cs.into_iter()) {
            if let Some(ch) = char::from_u32(c) {
                map.insert(ch, b as u8);
            }
        }
        map
    };
}

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

    /// Codifica texto a IDs de tokens usando Byte-Level BPE determinista.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // 1. Identificar y preservar tokens especiales
        let mut special_tokens: Vec<&str> = self
            .token_to_id
            .keys()
            .filter(|k| {
                k.starts_with("<|")
                    || *k == "<s>"
                    || *k == "</s>"
                    || *k == "<think>"
                    || *k == "</think>"
            })
            .map(|s| s.as_str())
            .collect();
        special_tokens.sort_by(|a, b| b.len().cmp(&a.len()));

        let mut chunks: Vec<(&str, bool)> = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            let mut matched_special = None;
            for &st in &special_tokens {
                if remaining.starts_with(st) {
                    matched_special = Some(st);
                    break;
                }
            }

            if let Some(st) = matched_special {
                chunks.push((st, true));
                remaining = &remaining[st.len()..];
            } else {
                let next_special_idx = special_tokens
                    .iter()
                    .filter_map(|&st| remaining.find(st))
                    .min();

                let regular_chunk_len = next_special_idx.unwrap_or(remaining.len());
                if regular_chunk_len > 0 {
                    chunks.push((&remaining[..regular_chunk_len], false));
                    remaining = &remaining[regular_chunk_len..];
                }
            }
        }

        let mut final_tokens: Vec<u32> = Vec::new();

        for (chunk, is_special) in chunks {
            if is_special {
                if let Some(&id) = self.token_to_id.get(chunk) {
                    final_tokens.push(id);
                }
                continue;
            }

            if let Some(&id) = self.token_to_id.get(chunk) {
                final_tokens.push(id);
                continue;
            }

            // Mapear bytes UTF-8 a caracteres estándar BPE
            let mut tokens: Vec<u32> = Vec::with_capacity(chunk.len());
            for &b in chunk.as_bytes() {
                let unicode_char_str = &BYTE_TO_UNICODE[b as usize];
                if let Some(&id) = self.token_to_id.get(unicode_char_str) {
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

            final_tokens.extend(tokens);
        }

        final_tokens
    }

    /// Decodifica IDs a texto plano UTF-8 reconstruyendo bytes de BPE.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in token_ids {
            if let Some(token_str) = self.vocab.get(id as usize) {
                for c in token_str.chars() {
                    if let Some(&b) = UNICODE_TO_BYTE.get(&c) {
                        bytes.push(b);
                    } else {
                        let mut buf = [0u8; 4];
                        let s = c.encode_utf8(&mut buf);
                        bytes.extend_from_slice(s.as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
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

    /// Retorna el formato activo del tokenizador (v1 Clásico vs v2 Genómico).
    pub fn format(&self) -> GtokFormat {
        if self.version >= GTOK_VERSION_V2 {
            GtokFormat::V2GenomicMorphological
        } else {
            GtokFormat::V1BpeClassic
        }
    }

    /// Codifica una palabra en su raíz léxica (lema) y codón nucleótido de 2-bits (GTOK v2).
    ///
    /// Mapeo de bases:
    /// - 'A' (00): Forma base / Lema exacto
    /// - 'C' (01): Plural / Desinencia nominal (ej. 's', 'es')
    /// - 'G' (10): Conjugación verbal / Acción (ej. 'ndo', 'do', 'ó', 'aron')
    /// - 'T' (11): Modificador / Adjetivo / Diminutivo (ej. 'mente', 'ito', 'ivo')
    pub fn encode_morphological_codon(&self, word: &str) -> (Option<u32>, NucleotideBase) {
        // 1. Probar coincidencia exacta (Adenina = forma base)
        if let Some(&id) = self.token_to_id.get(word) {
            return (Some(id), NucleotideBase::Adenine);
        }

        let lower = word.to_lowercase();
        if let Some(&id) = self.token_to_id.get(&lower) {
            return (Some(id), NucleotideBase::Adenine);
        }

        // Si es un ideograma CJK unificado (chino, japonés kanji: U+4E00..=U+9FFF), cada carácter es su propio lema base
        if word.chars().any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c)) {
            if let Some(&id) = self.token_to_id.get(word) {
                return (Some(id), NucleotideBase::Adenine);
            }
        }

        // 2. Desinencias verbales / Acción (Guanina = 10)
        // Español: -ndo, -iendo, -ando, -aron, -ieron, -amos, -emos, -ado, -ido, -aba, -ía
        // Inglés:  -ing, -ed, -es, -s, -tion, -ated
        for suffix in &[
            "ndo", "iendo", "ando", "aron", "ieron", "amos", "emos", "ado", "ido", "aba", "ía",
            "ing", "ed", "tion", "ated"
        ] {
            if lower.ends_with(suffix) && lower.len() > suffix.len() + 2 {
                let stem = &lower[..lower.len() - suffix.len()];
                if let Some(&id) = self.token_to_id.get(stem) {
                    return (Some(id), NucleotideBase::Guanine);
                }
            }
        }

        // 3. Desinencias de número/género / Flexión nominal (Citosina = 01)
        // Español: -es, -as, -os, -s, -a, -o
        // Inglés:  -s, -es, -ies
        for suffix in &["es", "as", "os", "s", "a", "o", "ies"] {
            if lower.ends_with(suffix) && lower.len() > suffix.len() + 2 {
                let stem = &lower[..lower.len() - suffix.len()];
                if let Some(&id) = self.token_to_id.get(stem) {
                    return (Some(id), NucleotideBase::Cytosine);
                }
            }
        }

        // 4. Modificadores / Adjetivos / Derivación (Timina = 11)
        // Español: -mente, -ivo, -iva, -ito, -ita, -able, -ible
        // Inglés:  -ly, -ful, -less, -able, -ible, -ness, -ment, -ive
        for suffix in &[
            "mente", "ivo", "iva", "ito", "ita", "able", "ible",
            "ly", "ful", "less", "ness", "ment", "ive"
        ] {
            if lower.ends_with(suffix) && lower.len() > suffix.len() + 2 {
                let stem = &lower[..lower.len() - suffix.len()];
                if let Some(&id) = self.token_to_id.get(stem) {
                    return (Some(id), NucleotideBase::Thymine);
                }
            }
        }

        // Si no hay raíz conocida, delegar al BPE estándar con base neutra (Adenina)
        (self.token_to_id.get(word).copied(), NucleotideBase::Adenine)
    }

    /// Serializa el tokenizador a formato binario GTOK (v1 o v2).
    pub fn to_bytes(&self, target_version: u16) -> Vec<u8> {
        let mut buf = Vec::new();

        // 1. Cabecera (16 bytes)
        buf.extend_from_slice(GTOK_MAGIC);
        buf.extend_from_slice(&target_version.to_le_bytes());
        buf.extend_from_slice(&self.flags.to_le_bytes());
        buf.extend_from_slice(&(self.vocab.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(self.merges.len() as u32).to_le_bytes());

        // 2. Tokens especiales (18 bytes)
        buf.extend_from_slice(&self.bos_id.to_le_bytes());
        buf.extend_from_slice(&self.eos_id.to_le_bytes());
        buf.extend_from_slice(&self.unk_id.to_le_bytes());
        buf.extend_from_slice(&self.pad_id.to_le_bytes());
        buf.extend_from_slice(&(self.extra_stop_ids.len() as u16).to_le_bytes());

        for &sid in &self.extra_stop_ids {
            buf.extend_from_slice(&sid.to_le_bytes());
        }

        // 3. String Table (Offsets + UTF-8 Pool)
        let mut string_pool = Vec::new();
        let mut offsets: Vec<u32> = Vec::with_capacity(self.vocab.len() + 1);
        offsets.push(0);

        for token in &self.vocab {
            string_pool.extend_from_slice(token.as_bytes());
            offsets.push(string_pool.len() as u32);
        }

        for off in offsets {
            buf.extend_from_slice(&off.to_le_bytes());
        }
        buf.extend_from_slice(&string_pool);

        // 4. Binary Merges Table
        for &(left, right, target) in &self.merges {
            buf.extend_from_slice(&left.to_le_bytes());
            buf.extend_from_slice(&right.to_le_bytes());
            buf.extend_from_slice(&target.to_le_bytes());
        }

        buf
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
        assert_eq!(tokenizer.format(), GtokFormat::V1BpeClassic);
    }

    #[test]
    fn test_gtok_v2_morphology_and_roundtrip() {
        let v1_data = vec![
            b'G', b'T', b'O', b'K', 1, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 5, 0, 0, 0, 11, 0, 0, 0, 15, 0, 0, 0,
            b'<', b'u', b'n', b'k', b'>', b'c', b'a', b'm', b'i', b'n', b'a', b'c', b'a', b's', b'a',
        ];

        let t1 = GtokNativeTokenizer::from_bytes(&v1_data).unwrap();
        assert_eq!(t1.format(), GtokFormat::V1BpeClassic);

        // Serializar a v2
        let v2_bytes = t1.to_bytes(GTOK_VERSION_V2);
        let t2 = GtokNativeTokenizer::from_bytes(&v2_bytes).unwrap();
        assert_eq!(t2.version, GTOK_VERSION_V2);
        assert_eq!(t2.format(), GtokFormat::V2GenomicMorphological);

        // Prueba de codón morfológico (desinencias nucleótidas)
        // 1. "camina" es forma base (Adenina = 00)
        let (id1, base1) = t2.encode_morphological_codon("camina");
        assert_eq!(id1, Some(1));
        assert_eq!(base1, NucleotideBase::Adenine);

        // 2. "caminando" deriva de "camina" con desinencia verbal (Guanina = 10)
        let (id2, base2) = t2.encode_morphological_codon("caminando");
        assert_eq!(id2, Some(1));
        assert_eq!(base2, NucleotideBase::Guanine);

        // 3. "casas" deriva de "casa" con desinencia plural (Citosina = 01)
        let (id3, base3) = t2.encode_morphological_codon("casas");
        assert_eq!(id3, Some(2));
        assert_eq!(base3, NucleotideBase::Cytosine);

        // 4. Prueba en inglés y CJK:
        // Token 0: "walk" (4 bytes: 0..4)
        // Token 1: "中" (3 bytes UTF-8: 4..7)
        let english_data = vec![
            b'G', b'T', b'O', b'K', 2, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 4, 0, 0, 0, 7, 0, 0, 0,
            b'w', b'a', b'l', b'k', 0xe4, 0xb8, 0xad,
        ];
        let t_en = GtokNativeTokenizer::from_bytes(&english_data).unwrap();
        let (id_walk, base_walk) = t_en.encode_morphological_codon("walking");
        assert_eq!(id_walk, Some(0));
        assert_eq!(base_walk, NucleotideBase::Guanine);

        // 5. Prueba CJK (chino): Los ideogramas son lemas puros en Adenina (00)
        let (id_cjk, base_cjk) = t_en.encode_morphological_codon("中");
        assert_eq!(id_cjk, Some(1));
        assert_eq!(base_cjk, NucleotideBase::Adenine);
    }
}
