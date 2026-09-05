//! # 🧬 Motor de Adaptación Genómica y Mutación In-Place para .gaje
//!
//! Permite aplicar mutaciones directas a centroides (Codebook Tuning),
//! calibraciones SPSA y registrar el linaje genealógico directamente en
//! la cabecera y cuerpo del modelo unificado sin reescribir gigabytes de pesos.

use crate::io::flat_reader::FlatTensorEntry;
use crate::io::header::FlatHeaderV2;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct MutationReport {
    pub tensor_name: String,
    pub num_centroids: usize,
    pub intensity: f32,
    pub old_centroids_sample: Vec<f32>,
    pub new_centroids_sample: Vec<f32>,
    pub mutation_index: u32,
    pub parent_hash: u64,
    pub new_lineage_hash: u64,
}

#[derive(Debug, Clone)]
pub struct LineageReport {
    pub path: String,
    pub num_tensors: u32,
    pub num_mutations: u32,
    pub parent_hash_hex: String,
    pub current_hash_hex: String,
    pub has_adaptive_section: bool,
    pub quant_format: String,
    pub arch_summary: String,
}

fn compute_mutation_hash(parent_hash: u64, tensor: &str, intensity: f32, step: u32) -> u64 {
    let mut h = parent_hash.wrapping_mul(31);
    for b in tensor.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    h = h.wrapping_add(intensity.to_bits() as u64);
    h = h.wrapping_add((step as u64).wrapping_mul(101));
    if h == 0 {
        1
    } else {
        h
    }
}

/// Aplica una mutación de centroides (Codebook Tuning) in-place en un modelo .gaje
pub fn mutate_model_centroids(
    path: &Path,
    target_tensor_name: Option<&str>,
    intensity: f32,
) -> Result<MutationReport, Box<dyn std::error::Error + Send + Sync>> {
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;

    // 1. Leer y validar cabecera FlatHeaderV2 (4096 bytes)
    let mut header_bytes = [0u8; FlatHeaderV2::SIZE];
    file.read_exact(&mut header_bytes)?;
    let mut header = FlatHeaderV2::from_bytes(&header_bytes)?;

    // 2. Leer directorio de tensores
    let dir_offset = FlatHeaderV2::SIZE as u64 + header.meta_len;
    file.seek(SeekFrom::Start(dir_offset))?;
    let mut dir_bytes = vec![0u8; header.dir_len as usize];
    file.read_exact(&mut dir_bytes)?;

    let entries: Vec<FlatTensorEntry> = serde_json::from_slice(&dir_bytes)?;
    let mut tensor_map: HashMap<String, FlatTensorEntry> = HashMap::new();
    for e in entries {
        tensor_map.insert(e.name.clone(), e);
    }

    // 3. Identificar tensor a mutar
    let chosen_entry = if let Some(name) = target_tensor_name {
        tensor_map
            .get(name)
            .ok_or_else(|| format!("Tensor '{}' no encontrado en el modelo.", name))?
    } else {
        // Por defecto: buscar el tensor de salida del último bloque o lm_head
        let mut last_attn = None;
        for i in (0..header.arch_n_blocks).rev() {
            let candidate = format!("blk.{}.attn_output", i);
            if tensor_map.contains_key(&candidate) {
                last_attn = tensor_map.get(&candidate);
                break;
            }
        }
        last_attn
            .or_else(|| tensor_map.get("lm_head"))
            .ok_or_else(|| {
                "No se encontró un tensor adecuado para mutar automáticamente.".to_string()
            })?
    };

    let mut old_sample = Vec::new();
    let mut new_sample = Vec::new();
    let num_mutated_params: usize;

    use rand::Rng;
    let mut rng = rand::thread_rng();

    if chosen_entry.c_len > 0 {
        // Caso A: Matriz con tabla explícita de centroides
        let c_absolute_offset = header.weights_offset + chosen_entry.c_off as u64;
        file.seek(SeekFrom::Start(c_absolute_offset))?;
        let mut c_bytes = vec![0u8; chosen_entry.c_len];
        file.read_exact(&mut c_bytes)?;

        let num_centroids = chosen_entry.c_len / 4;
        let mut centroids: Vec<f32> = Vec::with_capacity(num_centroids);
        for chunk in c_bytes.chunks_exact(4) {
            let val = f32::from_le_bytes(chunk.try_into().unwrap());
            centroids.push(val);
        }

        old_sample = centroids.iter().take(4).copied().collect();
        for c in centroids.iter_mut() {
            let noise = rng.gen_range(-1.0f32..1.0f32) * intensity;
            *c += noise;
        }
        new_sample = centroids.iter().take(4).copied().collect();

        let mut new_c_bytes = Vec::with_capacity(chosen_entry.c_len);
        for c in &centroids {
            new_c_bytes.extend_from_slice(&c.to_le_bytes());
        }
        file.seek(SeekFrom::Start(c_absolute_offset))?;
        file.write_all(&new_c_bytes)?;
        num_mutated_params = num_centroids;
    } else if chosen_entry.bit_depth == 4 && chosen_entry.dna_len >= 20 {
        // Caso B: Bloques Q4_0 estándar (mutación de escalas y offset de cuantización scale/min)
        let dna_absolute_offset = header.weights_offset + chosen_entry.dna_off as u64;
        file.seek(SeekFrom::Start(dna_absolute_offset))?;
        let mut dna_bytes = vec![0u8; chosen_entry.dna_len];
        file.read_exact(&mut dna_bytes)?;

        let n_blocks = chosen_entry.dna_len / 20;
        for i in 0..n_blocks {
            let b_off = i * 20;
            let scale_f16 = half::f16::from_le_bytes([dna_bytes[b_off], dna_bytes[b_off + 1]]);
            let min_f16 = half::f16::from_le_bytes([dna_bytes[b_off + 2], dna_bytes[b_off + 3]]);

            if i < 4 {
                old_sample.push(scale_f16.to_f32());
            }

            let noise_s = rng.gen_range(-1.0f32..1.0f32) * intensity * 0.1;
            let noise_m = rng.gen_range(-1.0f32..1.0f32) * intensity * 0.1;

            let new_scale = half::f16::from_f32(scale_f16.to_f32() + noise_s);
            let new_min = half::f16::from_f32(min_f16.to_f32() + noise_m);

            if i < 4 {
                new_sample.push(new_scale.to_f32());
            }

            let s_bytes = new_scale.to_le_bytes();
            let m_bytes = new_min.to_le_bytes();
            dna_bytes[b_off] = s_bytes[0];
            dna_bytes[b_off + 1] = s_bytes[1];
            dna_bytes[b_off + 2] = m_bytes[0];
            dna_bytes[b_off + 3] = m_bytes[1];
        }

        file.seek(SeekFrom::Start(dna_absolute_offset))?;
        file.write_all(&dna_bytes)?;
        num_mutated_params = n_blocks * 2;
    } else if chosen_entry.dna_len >= 4 {
        // Caso C: Pesos FP32 directos
        let dna_absolute_offset = header.weights_offset + chosen_entry.dna_off as u64;
        file.seek(SeekFrom::Start(dna_absolute_offset))?;
        let mut dna_bytes = vec![0u8; chosen_entry.dna_len];
        file.read_exact(&mut dna_bytes)?;

        let count = chosen_entry.dna_len / 4;
        let mut weights: Vec<f32> = Vec::with_capacity(count);
        for chunk in dna_bytes.chunks_exact(4) {
            weights.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }

        old_sample = weights.iter().take(4).copied().collect();
        for w in weights.iter_mut() {
            let noise = rng.gen_range(-1.0f32..1.0f32) * intensity;
            *w += noise;
        }
        new_sample = weights.iter().take(4).copied().collect();

        let mut new_bytes = Vec::with_capacity(chosen_entry.dna_len);
        for w in &weights {
            new_bytes.extend_from_slice(&w.to_le_bytes());
        }
        file.seek(SeekFrom::Start(dna_absolute_offset))?;
        file.write_all(&new_bytes)?;
        num_mutated_params = count;
    } else {
        return Err(format!(
            "El tensor '{}' no tiene datos de pesos mutables.",
            chosen_entry.name
        )
        .into());
    }

    // 7. Actualizar cabecera con el nuevo linaje y mutación
    let parent_hash = if header.lineage_current_hash == 0 {
        0xCAFE_BABE_DEAD_BEEF
    } else {
        header.lineage_current_hash
    };
    header.lineage_parent_hash = parent_hash;
    header.num_mutations += 1;
    let new_hash = compute_mutation_hash(
        parent_hash,
        &chosen_entry.name,
        intensity,
        header.num_mutations,
    );
    header.lineage_current_hash = new_hash;
    header.adapt_flags |= 1; // Bit 0: Adaptación activa

    let mut new_header_bytes = [0u8; FlatHeaderV2::SIZE];
    unsafe {
        std::ptr::write(new_header_bytes.as_mut_ptr() as *mut FlatHeaderV2, header);
    }
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&new_header_bytes)?;
    file.sync_all()?;

    Ok(MutationReport {
        tensor_name: chosen_entry.name.clone(),
        num_centroids: num_mutated_params,
        intensity,
        old_centroids_sample: old_sample,
        new_centroids_sample: new_sample,
        mutation_index: header.num_mutations,
        parent_hash,
        new_lineage_hash: new_hash,
    })
}

/// Inspecciona el linaje y estado adaptativo de un modelo .gaje
pub fn inspect_lineage(
    path: &Path,
) -> Result<LineageReport, Box<dyn std::error::Error + Send + Sync>> {
    let mut file = OpenOptions::new().read(true).open(path)?;
    let mut header_bytes = [0u8; FlatHeaderV2::SIZE];
    file.read_exact(&mut header_bytes)?;
    let header = FlatHeaderV2::from_bytes(&header_bytes)?;

    let quant_str = format!("{:?}", header.quantization_type());
    let arch_summary = format!(
        "Fam: {}, Dim: {}, Capas: {}, Heads: {} (KV: {})",
        header.arch_family,
        header.arch_n_embd,
        header.arch_n_blocks,
        header.arch_n_head,
        header.arch_n_head_kv
    );

    Ok(LineageReport {
        path: path.display().to_string(),
        num_tensors: header.num_tensors,
        num_mutations: header.num_mutations,
        parent_hash_hex: format!("{:#018x}", header.lineage_parent_hash),
        current_hash_hex: format!("{:#018x}", header.lineage_current_hash),
        has_adaptive_section: header.has_adaptive_section() || (header.adapt_flags & 1 != 0),
        quant_format: quant_str,
        arch_summary,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::config::ModelConfig;
    use crate::io::flat_writer::init_born_genomic_model;

    #[test]
    fn test_adaptive_mutation_and_lineage_roundtrip() {
        let temp_dir = std::env::temp_dir();
        let model_path = temp_dir.join("test_adaptive_born.gaje");
        let path_str = model_path.to_str().unwrap();

        let config = ModelConfig {
            config: crate::io::config::ArchConfig::default(),
            n_embd: 64,
            n_head: 2,
            n_head_kv: 2,
            n_blocks: 2,
            vocab_size: Some(128),
            eps: 1e-6,
        };

        let _ = init_born_genomic_model(path_str, config, 128).unwrap();

        // 1. Verificar linaje inicial
        let report_init = inspect_lineage(Path::new(path_str)).unwrap();
        assert_eq!(report_init.num_mutations, 0);

        // 2. Aplicar mutación in-place
        let mut_report = mutate_model_centroids(Path::new(path_str), None, 0.05).unwrap();
        assert_eq!(mut_report.mutation_index, 1);
        assert!(mut_report.new_lineage_hash != 0);

        // 3. Verificar linaje post-mutación
        let report_mut = inspect_lineage(Path::new(path_str)).unwrap();
        assert_eq!(report_mut.num_mutations, 1);
        assert!(report_mut.has_adaptive_section);
        assert_eq!(
            report_mut.current_hash_hex,
            format!("{:#018x}", mut_report.new_lineage_hash)
        );

        // Limpieza
        let _ = std::fs::remove_file(model_path);
    }
}
