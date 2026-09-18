// =============================================================================
// genomize — Genomización de f32/f16, 4-bit y cuantización toroidal
// =============================================================================
use half::f16;
use std::cmp::Ordering;

pub fn genomize_f32_core(
    f32_data: &[f32],
    block_size: usize,
    anchor_threshold: f32,
    custom_base_c: Option<[f32; 4]>,
) -> (Vec<u8>, Vec<f32>, Vec<u8>) {
    let n_elements = f32_data.len();
    let n_blocks = n_elements / block_size;

    let mut dna_database = Vec::with_capacity(n_elements / 4);
    let mut all_centroids = Vec::with_capacity(n_blocks * 4);

    // Si anchor_threshold > 0 y < 1.0, lo interpretamos como densidad (ej: 0.01 = 1%)
    let mut actual_threshold = anchor_threshold;
    if anchor_threshold > 0.0 && anchor_threshold < 1.0 {
        let mut abs_vals: Vec<f32> = f32_data.iter().map(|v| v.abs()).collect();
        abs_vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let top_idx = (n_elements as f32 * anchor_threshold) as usize;
        actual_threshold = if top_idx < n_elements {
            abs_vals[top_idx]
        } else {
            0.0
        };
    }

    let mut anchor_indices = Vec::new();
    let mut anchor_values = Vec::new();
    let _anchor_row_ptrs = [0u64; 1]; // Temporal, se ajustará después si es multidimensional

    let base_c = custom_base_c.unwrap_or([-1.510f32, -0.4528, 0.4528, 1.510]);

    for i in 0..n_blocks {
        let start = i * block_size;
        let block_f32 = &f32_data[start..start + block_size];

        let mut sum = 0.0f32;
        for &val in block_f32 {
            sum += val;
        }
        let mean = sum / block_size as f32;

        let mut var_sum = 0.0f32;
        for &val in block_f32 {
            let diff = val - mean;
            var_sum += diff * diff;
        }
        let std = (var_sum / block_size as f32).sqrt() + 1e-6;

        let t = [mean - std, mean, mean + std];
        let c = [
            mean + base_c[0] * std,
            mean + base_c[1] * std,
            mean + base_c[2] * std,
            mean + base_c[3] * std,
        ];

        for k in 0..(block_size / 4) {
            let mut byte = 0u8;
            for s in 0..4 {
                let idx = k * 4 + s;
                let val = block_f32[idx];
                let bits = if val < t[0] {
                    0b00
                } else if val < t[1] {
                    0b01
                } else if val < t[2] {
                    0b11
                } else {
                    0b10
                };

                let c_val = match bits {
                    0b00 => c[0],
                    0b01 => c[1],
                    0b11 => c[2],
                    0b10 => c[3],
                    _ => 0.0,
                };

                let residual = val - c_val;
                if anchor_threshold > 0.0 && val.abs() >= actual_threshold {
                    anchor_indices.push((start + idx) as u32);
                    anchor_values.push(half::f16::from_f32(residual));
                }
                byte = (byte << 2) | bits;
            }
            dna_database.push(byte);
        }
        for &cv in &c {
            all_centroids.push(cv);
        }
    }

    // Empaquetar en formato "GAJE"
    let mut anchors_u8 = Vec::new();
    if !anchor_indices.is_empty() {
        anchors_u8.extend_from_slice(b"GAJE");
        let count = anchor_indices.len() as u32;
        anchors_u8.extend_from_slice(&count.to_le_bytes());
        for &idx in &anchor_indices {
            anchors_u8.extend_from_slice(&idx.to_le_bytes());
        }
        for &val in &anchor_values {
            anchors_u8.extend_from_slice(&val.to_le_bytes());
        }
        // Para genomize simple, asumimos una sola fila virtual o dejamos row_ptrs para el llamador
        // NOTA: GenomicLinear::new espera row_ptrs de tamaño out_features + 1.
        // Aquí genomize_f32_core no conoce out_features directamente (solo n_elements).
        // Por ahora, pondremos [0, count] asumiendo una fila, pero lo ideal es que el llamador lo gestione.
        anchors_u8.extend_from_slice(&0u64.to_le_bytes());
        anchors_u8.extend_from_slice(&(count as u64).to_le_bytes());
    }

    (dna_database, all_centroids, anchors_u8)
}

/// # 🧬 Cuantización Toroidal (Fase Compleja)
///
/// Proyecta tensores f32 en el cuerpo ciclotómico Q(zeta_16), tratando los
/// valores como ángulos en un toroide semántico.
pub fn quantize_toroidal_core(data: &[f32], block_size: usize) -> (Vec<u8>, Vec<f32>) {
    let n_elements = data.len();
    let n_blocks = n_elements / block_size;
    let mut dna = Vec::with_capacity(n_elements / 4);
    let mut phase_centroids = Vec::with_capacity(n_blocks * 4);

    for i in 0..n_blocks {
        let block = &data[i * block_size..(i + 1) * block_size];

        // En la topología toroidal, los centroides representan los "polos" de resonancia
        // de la fase compleja. Usamos 4 polos cardinales por bloque.
        let mut sum = 0.0f32;
        for &v in block {
            sum += v;
        }
        let mean = sum / block_size as f32;

        // Determinamos la amplitud del toroide local (dispersión)
        let mut var = 0.0f32;
        for &v in block {
            var += (v - mean).powi(2);
        }
        let std = (var / block_size as f32).sqrt() + 1e-6;

        // Polos cardinales: N, S, E, W en el plano complejo semántico
        let c = [mean - std, mean + std, mean - std * 0.5, mean + std * 0.5];
        for &val in &c {
            phase_centroids.push(val);
        }

        for k in 0..(block_size / 4) {
            let mut byte = 0u8;
            for s in 0..4 {
                let val = block[k * 4 + s];
                // Mapeo a fase (2 bits)
                let bits = if val < c[0] {
                    0b00
                } else if val < c[2] {
                    0b01
                } else if val < c[3] {
                    0b11
                } else {
                    0b10
                };
                byte = (byte << 2) | bits;
            }
            dna.push(byte);
        }
    }
    (dna, phase_centroids)
}

pub fn genomize_f16_core(
    f16_data: &[f16],
    block_size: usize,
    anchor_threshold: f32,
    custom_base_c: Option<[f32; 4]>,
) -> (Vec<u8>, Vec<f32>, Vec<u8>) {
    let n_elements = f16_data.len();
    let n_blocks = n_elements / block_size;
    let mut dna_database = Vec::with_capacity(n_elements / 4);
    let mut all_centroids = Vec::with_capacity(n_blocks * 4);

    let mut actual_threshold = anchor_threshold;
    if anchor_threshold > 0.0 && anchor_threshold < 1.0 {
        let mut abs_vals: Vec<f32> = f16_data.iter().map(|v| v.to_f32().abs()).collect();
        abs_vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let top_idx = (n_elements as f32 * anchor_threshold) as usize;
        actual_threshold = if top_idx < n_elements {
            abs_vals[top_idx]
        } else {
            0.0
        };
    }

    let mut anchor_indices = Vec::new();
    let mut anchor_values = Vec::new();

    let base_c = custom_base_c.unwrap_or([-1.510f32, -0.4528, 0.4528, 1.510]);
    for i in 0..n_blocks {
        let start = i * block_size;
        let block_f16 = &f16_data[start..start + block_size];
        let mut block_f32 = vec![0.0f32; block_size];
        let mut sum = 0.0f32;
        for j in 0..block_size {
            let val = block_f16[j].to_f32();
            block_f32[j] = val;
            sum += val;
        }
        let mean = sum / block_size as f32;
        let mut var_sum = 0.0f32;
        for &val in &block_f32 {
            let diff = val - mean;
            var_sum += diff * diff;
        }
        let std = (var_sum / block_size as f32).sqrt() + 1e-6;
        let t = [mean - std, mean, mean + std];
        let c = [
            mean + base_c[0] * std,
            mean + base_c[1] * std,
            mean + base_c[2] * std,
            mean + base_c[3] * std,
        ];

        for k in 0..(block_size / 4) {
            let mut byte = 0u8;
            for s in 0..4 {
                let idx = k * 4 + s;
                let val = block_f32[idx];
                let bits = if val < t[0] {
                    0b00
                } else if val < t[1] {
                    0b01
                } else if val < t[2] {
                    0b11
                } else {
                    0b10
                };

                let c_val = match bits {
                    0b00 => c[0],
                    0b01 => c[1],
                    0b11 => c[2],
                    0b10 => c[3],
                    _ => 0.0,
                };

                let residual = val - c_val;
                if anchor_threshold > 0.0 && val.abs() >= actual_threshold {
                    anchor_indices.push((start + idx) as u32);
                    anchor_values.push(half::f16::from_f32(residual));
                }
                byte = (byte << 2) | bits;
            }
            dna_database.push(byte);
        }
        for &cv in &c {
            all_centroids.push(cv);
        }
    }

    let mut anchors_u8 = Vec::new();
    if !anchor_indices.is_empty() {
        anchors_u8.extend_from_slice(b"GAJE");
        let count = anchor_indices.len() as u32;
        anchors_u8.extend_from_slice(&count.to_le_bytes());
        for &idx in &anchor_indices {
            anchors_u8.extend_from_slice(&idx.to_le_bytes());
        }
        for &val in &anchor_values {
            anchors_u8.extend_from_slice(&val.to_le_bytes());
        }
        anchors_u8.extend_from_slice(&0u64.to_le_bytes());
        anchors_u8.extend_from_slice(&(count as u64).to_le_bytes());
    }

    (dna_database, all_centroids, anchors_u8)
}

pub fn genomize_4bit_core(
    f32_data: &[f32],
    block_size: usize,
    anchor_threshold: f32,
) -> (Vec<u8>, Vec<f32>, Vec<u8>) {
    let n_elements = f32_data.len();
    let n_blocks = n_elements / block_size;
    let mut dna_database = Vec::with_capacity(n_elements / 2);
    let mut all_centroids = Vec::with_capacity(n_blocks * 16);

    let mut actual_threshold = anchor_threshold;
    if anchor_threshold > 0.0 && anchor_threshold < 1.0 {
        let mut abs_vals: Vec<f32> = f32_data.iter().map(|v| v.abs()).collect();
        abs_vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let top_idx = (n_elements as f32 * anchor_threshold) as usize;
        actual_threshold = if top_idx < n_elements {
            abs_vals[top_idx]
        } else {
            0.0
        };
    }

    let mut anchor_indices = Vec::new();
    let mut anchor_values = Vec::new();

    for i in 0..n_blocks {
        let start = i * block_size;
        let block_f32 = &f32_data[start..start + block_size];

        // 16 centroides lineales para 4 bits en el rango del bloque
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in block_f32 {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }

        let mut c = [0.0f32; 16];
        let step = (max_val - min_val) / 15.0;
        for j in 0..16 {
            c[j] = min_val + j as f32 * step;
            all_centroids.push(c[j]);
        }

        for k in 0..(block_size / 2) {
            let mut byte = 0u8;
            for s in 0..2 {
                let idx = k * 2 + s;
                let val = block_f32[idx];

                // Cuantización 4-bit (Centroide más cercano)
                let mut best_idx = 0;
                let mut min_dist = f32::MAX;
                for j in 0..16 {
                    let dist = (val - c[j]).abs();
                    if dist < min_dist {
                        min_dist = dist;
                        best_idx = j;
                    }
                }

                if s == 0 {
                    byte |= (best_idx as u8) << 4;
                } else {
                    byte |= best_idx as u8;
                }

                if anchor_threshold > 0.0 && val.abs() >= actual_threshold {
                    anchor_indices.push((start + idx) as u32);
                    anchor_values.push(f16::from_f32(val));
                }
            }
            dna_database.push(byte);
        }
    }

    // Empaquetar anclas en formato GAJE
    let mut anchors_buf = Vec::new();
    anchors_buf.extend_from_slice(b"GAJE");
    anchors_buf.extend_from_slice(&(anchor_indices.len() as u32).to_le_bytes());
    for &idx in &anchor_indices {
        anchors_buf.extend_from_slice(&idx.to_le_bytes());
    }
    for &val in &anchor_values {
        anchors_buf.extend_from_slice(&val.to_le_bytes());
    }
    let row_ptrs = [0u64, anchor_indices.len() as u64]; // Simplificación para exportación de 1 layer
    for &ptr in &row_ptrs {
        anchors_buf.extend_from_slice(&ptr.to_le_bytes());
    }

    (dna_database, all_centroids, anchors_buf)
}

/// Cuantiza datos en f32 a bloques `Q8_0Block` en paralelo con Rayon.
pub fn quantize_to_q8_0(f32_data: &[f32]) -> Vec<crate::io::header::Q8_0Block> {
    use rayon::prelude::*;
    let n_blocks = f32_data.len() / 32;
    (0..n_blocks)
        .into_par_iter()
        .map(|i| {
            let start = i * 32;
            let block_f32 = &f32_data[start..start + 32];

            let mut max_abs = 0.0f32;
            for &v in block_f32 {
                let abs_v = v.abs();
                if abs_v > max_abs {
                    max_abs = abs_v;
                }
            }

            let scale = max_abs / 127.0;
            let f16_scale = half::f16::from_f32(scale);
            let eff_scale = f16_scale.to_f32();
            let inv_scale = if eff_scale > 1e-7 { 1.0 / eff_scale } else { 0.0 };

            let mut qs = [0i8; 32];
            for k in 0..32 {
                let q = if eff_scale > 1e-7 {
                    (block_f32[k] * inv_scale).round().clamp(-128.0, 127.0) as i8
                } else {
                    0
                };
                qs[k] = q;
            }

            crate::io::header::Q8_0Block {
                scale: f16_scale,
                qs,
            }
        })
        .collect()
}

/// Cuantiza bytes BF16 directamente a bloques `Q8_0Block` en paralelo con Rayon sin asignar f32 en heap.
pub fn quantize_bf16_to_q8_0(bf16_bytes: &[u8]) -> Vec<crate::io::header::Q8_0Block> {
    use rayon::prelude::*;
    let n_elements = bf16_bytes.len() / 2;
    let n_blocks = n_elements / 32;
    (0..n_blocks)
        .into_par_iter()
        .map(|i| {
            let start_byte = i * 32 * 2;
            let mut block_f32 = [0.0f32; 32];
            let mut max_abs = 0.0f32;

            for k in 0..32 {
                let b_idx = start_byte + k * 2;
                let u_val = u16::from_le_bytes([bf16_bytes[b_idx], bf16_bytes[b_idx + 1]]);
                let f_val = f32::from_bits((u_val as u32) << 16);
                block_f32[k] = f_val;
                let abs_v = f_val.abs();
                if abs_v > max_abs {
                    max_abs = abs_v;
                }
            }

            let scale = max_abs / 127.0;
            let f16_scale = half::f16::from_f32(scale);
            let eff_scale = f16_scale.to_f32();
            let inv_scale = if eff_scale > 1e-7 { 1.0 / eff_scale } else { 0.0 };

            let mut qs = [0i8; 32];
            for k in 0..32 {
                let q = if eff_scale > 1e-7 {
                    (block_f32[k] * inv_scale).round().clamp(-128.0, 127.0) as i8
                } else {
                    0
                };
                qs[k] = q;
            }

            crate::io::header::Q8_0Block {
                scale: f16_scale,
                qs,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_to_q8_0_fidelity() {
        let mut original = vec![0.0f32; 64];
        for (i, v) in original.iter_mut().enumerate() {
            *v = (i as f32 - 32.0) * 0.125;
        }

        let blocks = quantize_to_q8_0(&original);
        assert_eq!(blocks.len(), 2);

        for (i, &orig_val) in original.iter().enumerate() {
            let block_idx = i / 32;
            let within_block = i % 32;
            let dequant = blocks[block_idx].dequantize_weight(within_block);
            let diff = (orig_val - dequant).abs();
            assert!(
                diff < 0.05,
                "Error en idx {}: orig={}, dequant={}, diff={}",
                i,
                orig_val,
                dequant,
                diff
            );
        }
    }

    #[test]
    fn test_quantize_bf16_to_q8_0_equivalence() {
        let mut f32_vals = vec![0.0f32; 32];
        let mut bf16_bytes = vec![0u8; 64];

        for i in 0..32 {
            let val = (i as f32 - 15.5) * 0.25;
            let f32_bits = val.to_bits();
            let bf16_bits = (f32_bits >> 16) as u16;
            let reconstructed_f32 = f32::from_bits((bf16_bits as u32) << 16);
            f32_vals[i] = reconstructed_f32;

            let b = bf16_bits.to_le_bytes();
            bf16_bytes[i * 2] = b[0];
            bf16_bytes[i * 2 + 1] = b[1];
        }

        let blocks_from_f32 = quantize_to_q8_0(&f32_vals);
        let blocks_from_bf16 = quantize_bf16_to_q8_0(&bf16_bytes);

        assert_eq!(blocks_from_f32.len(), 1);
        assert_eq!(blocks_from_bf16.len(), 1);

        assert_eq!(
            blocks_from_f32[0].scale.to_bits(),
            blocks_from_bf16[0].scale.to_bits()
        );
        assert_eq!(blocks_from_f32[0].qs, blocks_from_bf16[0].qs);
    }
}


