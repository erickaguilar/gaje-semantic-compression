use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use crate::nn::GenomicLinear;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Codebook {
    pub centroids: HashMap<String, Vec<f32>>,
}

pub struct GAJEArchive {
    pub codebook: HashMap<String, Vec<f32>>,
    pub epigenetic_codebook: Option<HashMap<String, Vec<f32>>>,
    pub entries: HashMap<String, GenomicEntry>,
}

pub struct GenomicEntry {
    pub dna: Vec<u8>,
    pub epi_dna: Option<Vec<u8>>,
}

impl GAJEArchive {
    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut f = File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"GAJE" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid magic"));
        }

        let mut ver_buf = [0u8; 2];
        f.read_exact(&mut ver_buf)?;
        let ver = u16::from_le_bytes(ver_buf);

        // Base Codebook
        let mut len_buf = [0u8; 4];
        f.read_exact(&mut len_buf)?;
        let cb_len = u32::from_le_bytes(len_buf) as usize;
        let mut cb_json = vec![0u8; cb_len];
        f.read_exact(&mut cb_json)?;
        let codebook: HashMap<String, Vec<f32>> = serde_json::from_slice(&cb_json)?;

        let mut epi_codebook = None;
        if ver >= 3 {
            f.read_exact(&mut len_buf)?;
            let epi_cb_len = u32::from_le_bytes(len_buf) as usize;
            let mut epi_cb_json = vec![0u8; epi_cb_len];
            f.read_exact(&mut epi_cb_json)?;
            if epi_cb_len > 2 { // Not just "{}"
                epi_codebook = Some(serde_json::from_slice(&epi_cb_json)?);
            }
        }

        f.read_exact(&mut len_buf)?;
        let count = u32::from_le_bytes(len_buf) as usize;
        let mut entries = HashMap::new();

        for _ in 0..count {
            f.read_exact(&mut len_buf)?;
            let l_len = u32::from_le_bytes(len_buf) as usize;
            let mut label_buf = vec![0u8; l_len];
            f.read_exact(&mut label_buf)?;
            let label = String::from_utf8_lossy(&label_buf).into_owned();

            f.read_exact(&mut len_buf)?;
            let d_len = u32::from_le_bytes(len_buf) as usize;
            let mut dna = vec![0u8; d_len];
            f.read_exact(&mut dna)?;

            let mut epi_dna = None;
            if ver >= 3 {
                f.read_exact(&mut len_buf)?;
                let e_len = u32::from_le_bytes(len_buf) as usize;
                if e_len > 0 {
                    let mut e_dna = vec![0u8; e_len];
                    f.read_exact(&mut e_dna)?;
                    epi_dna = Some(e_dna);
                }
            }

            entries.insert(label, GenomicEntry { dna, epi_dna });
        }

        Ok(GAJEArchive {
            codebook,
            epigenetic_codebook,
            entries,
        })
    }
}
