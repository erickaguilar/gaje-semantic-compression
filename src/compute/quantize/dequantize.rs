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
