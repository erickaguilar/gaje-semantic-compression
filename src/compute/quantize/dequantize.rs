// =============================================================================
// dequantize — Reconstrucción de embeddings y bloques cuantizados
// =============================================================================
use rayon::prelude::*;

pub fn dequantize_embedding_core(
    dna_packed: &[u8],
    dims: usize,
    centroids: Option<&[f32]>,
) -> Result<Vec<f32>, String> {
    let c = centroids.unwrap_or(&[-0.68, -0.17, 0.17, 0.68]);
    let mut rec = Vec::with_capacity(dims);
    let mut dp = 0;
    let is_multi = c.len() == dims * 4;
    for &byte in dna_packed {
        for j in 0..4 {
            if dp >= dims {
                break;
            }
            let s = (3 - j) * 2;
            let bits = (byte >> s) & 0b11;
            let cent = if is_multi {
                let b = dp * 4;
                match bits {
                    0b00 => c[b],
                    0b01 => c[b + 1],
                    0b11 => c[b + 2],
                    0b10 => c[b + 3],
                    _ => 0.0,
                }
            } else {
                match bits {
                    0b00 => c[0],
                    0b01 => c[1],
                    0b11 => c[2],
                    0b10 => c[3],
                    _ => 0.0,
                }
            };
            rec.push(cent);
            dp += 1;
        }
    }
    Ok(rec)
}

pub fn dequantize_q8_0_core(data_u8: &[u8], out_features: usize, in_features: usize) -> Vec<f32> {
    let n_blocks = in_features / 32;
    let block_size = 34;
    let mut results = vec![0.0f32; out_features * in_features];
    results
        .par_chunks_mut(in_features)
        .enumerate()
        .for_each(|(i, row)| {
            let row_offset = i * n_blocks * block_size;
            for b in 0..n_blocks {
                let offset = row_offset + b * block_size;
                if offset + 2 > data_u8.len() {
                    break;
                }
                let delta =
                    half::f16::from_le_bytes([data_u8[offset], data_u8[offset + 1]]).to_f32();
                for j in 0..32 {
                    if offset + 2 + j >= data_u8.len() {
                        break;
                    }
                    row[b * 32 + j] = (data_u8[offset + 2 + j] as i8 as f32) * delta;
                }
            }
        });
    results
}

pub fn dequantize_q4_0_core(data_u8: &[u8], out_features: usize, in_features: usize) -> Vec<f32> {
    let n_blocks = in_features / 32;
    let block_size = 18; // 2 bytes delta f16 + 16 bytes nibbles
    let mut results = vec![0.0f32; out_features * in_features];
    results
        .par_chunks_mut(in_features)
        .enumerate()
        .for_each(|(i, row)| {
            let row_offset = i * n_blocks * block_size;
            for b in 0..n_blocks {
                let offset = row_offset + b * block_size;
                if offset + 18 > data_u8.len() {
                    break;
                }
                let delta =
                    half::f16::from_le_bytes([data_u8[offset], data_u8[offset + 1]]).to_f32();
                let qs = &data_u8[offset + 2..offset + 18];
                for j in 0..16 {
                    let byte = qs[j];
                    let v0 = (byte & 0x0F) as i8 - 8;
                    let v1 = ((byte >> 4) & 0x0F) as i8 - 8;
                    row[b * 32 + j] = (v0 as f32) * delta;
                    row[b * 32 + j + 16] = (v1 as f32) * delta;
                }
            }
        });
    results
}

pub fn dequantize_q6_k_core(data_u8: &[u8], out_features: usize, in_features: usize) -> Vec<f32> {
    let n_blocks = in_features / 256;
    let block_size = 210; // 128 (ql) + 64 (qh) + 16 (scales) + 2 (d)
    let mut results = vec![0.0f32; out_features * in_features];
    results
        .par_chunks_mut(in_features)
        .enumerate()
        .for_each(|(i, row)| {
            let row_offset = i * n_blocks * block_size;
            for b in 0..n_blocks {
                let offset = row_offset + b * block_size;
                if offset + block_size > data_u8.len() {
                    break;
                }
                let block = &data_u8[offset..offset + block_size];
                let ql = &block[0..128];
                let qh = &block[128..192];
                let scales = &block[192..208];
                let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();

                for half in 0..2 {
                    let ql_h = &ql[half * 64..(half + 1) * 64];
                    let qh_h = &qh[half * 32..(half + 1) * 32];
                    let sc_h = &scales[half * 8..(half + 1) * 8];
                    for l in 0..32 {
                        let is_ = l / 16;
                        let sc0 = (sc_h[is_ + 0] as i8) as f32;
                        let sc2 = (sc_h[is_ + 2] as i8) as f32;
                        let sc4 = (sc_h[is_ + 4] as i8) as f32;
                        let sc6 = (sc_h[is_ + 6] as i8) as f32;

                        let q1 = ((ql_h[l] & 0x0F) | (((qh_h[l] >> 0) & 0x03) << 4)) as i8 - 32;
                        let q2 = ((ql_h[l + 32] & 0x0F) | (((qh_h[l] >> 2) & 0x03) << 4)) as i8 - 32;
                        let q3 = (((ql_h[l] >> 4) & 0x0F) | (((qh_h[l] >> 4) & 0x03) << 4)) as i8 - 32;
                        let q4 = (((ql_h[l + 32] >> 4) & 0x0F) | (((qh_h[l] >> 6) & 0x03) << 4)) as i8 - 32;

                        let base_idx = b * 256 + half * 128 + l;
                        row[base_idx + 0] = d * sc0 * (q1 as f32);
                        row[base_idx + 32] = d * sc2 * (q2 as f32);
                        row[base_idx + 64] = d * sc4 * (q3 as f32);
                        row[base_idx + 96] = d * sc6 * (q4 as f32);
                    }
                }
            }
        });
    results
}

pub fn generate_default_centroids(n_blocks: usize) -> Vec<f32> {
    let mut centroids = Vec::with_capacity(n_blocks * 4);
    for _ in 0..n_blocks {
        centroids.push(-1.51);
        centroids.push(-0.45);
        centroids.push(0.45);
        centroids.push(1.51);
    }
    centroids
}
