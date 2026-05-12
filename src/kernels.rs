use std::arch::aarch64::*;

#[inline(always)]
pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut sum_v = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        sum_v = vfmaq_f32(sum_v, va, vb);
        i += 4;
    }
    let mut sum = vaddvq_f32(sum_v);
    while i < n { sum += a[i] * b[i]; i += 1; }
    sum
}

#[inline(always)]
pub unsafe fn rms_norm_neon(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut sum_v = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= n {
        let vx = vld1q_f32(x.as_ptr().add(i));
        sum_v = vfmaq_f32(sum_v, vx, vx);
        i += 4;
    }
    let mut sum_sq = vaddvq_f32(sum_v);
    while i < n { sum_sq += x[i] * x[i]; i += 1; }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    let inv_rms_v = vdupq_n_f32(inv_rms);
    let mut out = vec![0.0f32; n];
    i = 0;
    while i + 4 <= n {
        let vx = vld1q_f32(x.as_ptr().add(i));
        let vw = vld1q_f32(weight.as_ptr().add(i));
        let res = vmulq_f32(vmulq_f32(vx, inv_rms_v), vw);
        vst1q_f32(out.as_mut_ptr().add(i), res);
        i += 4;
    }
    while i < n { out[i] = (x[i] * inv_rms) * weight[i]; i += 1; }
    out
}

// Precomputed shuffle masks for decoding 2-bit values to float indices
// Each byte index (0-255) maps to 16 bytes (4 floats)
static mut SHUFFLE_MASK_TABLE: [[u8; 16]; 256] = [[0; 16]; 256];
static mut SHUFFLE_TABLE_INITIALIZED: bool = false;

pub unsafe fn init_shuffle_table() {
    if SHUFFLE_TABLE_INITIALIZED { return; }
    for b in 0..256usize {
        for i in 0..4 {
            let shift = (3 - i) * 2;
            let bits = (b >> shift) & 0b11;
            // bits 00->0, 01->1, 11->2, 10->3
            let idx = (bits ^ (bits >> 1)) as u8;
            for j in 0..4 {
                SHUFFLE_MASK_TABLE[b][(i * 4 + j) as usize] = idx * 4 + j as u8;
            }
        }
    }
    SHUFFLE_TABLE_INITIALIZED = true;
}

#[inline(always)]
pub unsafe fn genomic_dot_product_neon(
    weights: &[u8],
    input: &[f32],
    centroids: &[f32],
    stride: usize,
    n_blocks: usize,
) -> f32 {
    let mut sum_v = vdupq_n_f32(0.0);
    #[allow(static_mut_refs)]
    let table_ptr = SHUFFLE_MASK_TABLE.as_ptr();

    for j in 0..n_blocks {
        let c_v = vld1q_u8(centroids.as_ptr().add(j * 4) as *const u8);
        let input_block_ptr = input.as_ptr().add(j * stride * 4);
        let weights_block_ptr = weights.as_ptr().add(j * stride);

        for k in 0..stride {
            let byte = *weights_block_ptr.add(k);
            let mask = vld1q_u8(table_ptr.add(byte as usize) as *const u8);
            let v_vals = vqtbl1q_u8(c_v, mask);
            let v_vals_f = vreinterpretq_f32_u8(v_vals);
            let v_in = vld1q_f32(input_block_ptr.add(k * 4));
            sum_v = vfmaq_f32(sum_v, v_vals_f, v_in);
        }
    }
    vaddvq_f32(sum_v)
}
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn calculate_distance_lut_neon(lut_base: &[f32], lut_epi: &[f32], lut_tri: &[f32], strand: &[u8], epi_strand: &[u8], tri_strand: &[u8], mask: &[u8], n_dims: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut sum_v = vdupq_n_f32(0.0);
    let mut dims = 0;
    let n_blocks = n_dims / 4;
    for i in 0..n_blocks {
        let mode = *mask.get(i).unwrap_or(&0);
        let b_byte = *strand.get(i).unwrap_or(&0);
        let mut d_v = [0.0f32; 4];
        for j in 0..4 {
            let shift = (3 - j) * 2;
            let bb = (b_byte >> shift) & 0b11;
            let b_idx = (bb ^ (bb >> 1)) as usize;
            if mode == 0 { d_v[j] = *lut_base.get(dims * 4 + b_idx).unwrap_or(&0.0); }
            else if mode == 1 { let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11; let e_idx = (eb ^ (eb >> 1)) as usize; d_v[j] = *lut_epi.get(dims * 16 + (b_idx << 2 | e_idx)).unwrap_or(&0.0); }
            else { let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11; let tb = (*tri_strand.get(i).unwrap_or(&0) >> shift) & 0b11; let e_idx = (eb ^ (eb >> 1)) as usize; let t_idx = (tb ^ (tb >> 1)) as usize; d_v[j] = *lut_tri.get(dims * 64 + (b_idx << 4 | e_idx << 2 | t_idx)).unwrap_or(&0.0); }
            dims += 1;
        }
        sum_v = vaddq_f32(sum_v, vld1q_f32(d_v.as_ptr()));
    }
    let mut total = vaddvq_f32(sum_v);
    while dims < n_dims {
        let i = dims / 4; let mode = *mask.get(i).unwrap_or(&0); let shift = (3 - (dims % 4)) * 2; let bb = (*strand.get(i).unwrap_or(&0) >> shift) & 0b11; let b_idx = (bb ^ (bb >> 1)) as usize;
        if mode == 0 { total += *lut_base.get(dims * 4 + b_idx).unwrap_or(&0.0); }
        else if mode == 1 { let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11; let e_idx = (eb ^ (eb >> 1)) as usize; total += *lut_epi.get(dims * 16 + (b_idx << 2 | e_idx)).unwrap_or(&0.0); }
        else { let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11; let tb = (*tri_strand.get(i).unwrap_or(&0) >> shift) & 0b11; let e_idx = (eb ^ (eb >> 1)) as usize; let t_idx = (tb ^ (tb >> 1)) as usize; total += *lut_tri.get(dims * 64 + (b_idx << 4 | e_idx << 2 | t_idx)).unwrap_or(&0.0); }
        dims += 1;
    }
    total.sqrt()
}
