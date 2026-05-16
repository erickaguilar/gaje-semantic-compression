// =============================================================================
// kernels.rs — Motor SIMD multiplataforma para GAJE
//
// aarch64  → ARM NEON (Android/Termux)
// x86_64   → AVX2 + FMA (Windows/Linux PC)
// fallback → escalar puro (cualquier otro target)
// =============================================================================

// ─── Importaciones condicionales ─────────────────────────────────────────────
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// =============================================================================
// dot_product — Producto punto vectorizado
// =============================================================================

#[cfg(target_arch = "aarch64")]
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
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();

    // Verificación dinámica de AVX2+FMA en tiempo de ejecución
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            acc = _mm256_fmadd_ps(va, vb, acc);
            i += 8;
        }
        // Reducción horizontal: 8 floats → 1
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        let mut sum = _mm_cvtss_f32(result);
        // Cola escalar
        while i < n { sum += a[i] * b[i]; i += 1; }
        sum
    } else {
        // Fallback SSE2 (siempre disponible en x86_64)
        let mut acc = _mm_setzero_ps();
        let mut i = 0;
        while i + 4 <= n {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
            i += 4;
        }
        let shuf = _mm_movehdup_ps(acc);
        let sums = _mm_add_ps(acc, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        let mut sum = _mm_cvtss_f32(result);
        while i < n { sum += a[i] * b[i]; i += 1; }
        sum
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// =============================================================================
// rms_norm — Normalización RMS vectorizada
// =============================================================================

#[cfg(target_arch = "aarch64")]
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
    while i < n {
        sum_sq += x[i] * x[i];
        i += 1;
    }
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
    while i < n {
        out[i] = (x[i] * inv_rms) * weight[i];
        i += 1;
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn rms_norm_neon(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut out = vec![0.0f32; n];

    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // Paso 1: Calcular sum(x²) con AVX2+FMA
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));
            acc = _mm256_fmadd_ps(vx, vx, acc);
            i += 8;
        }
        // Reducción horizontal
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        let mut sum_sq = _mm_cvtss_f32(result);
        while i < n { sum_sq += x[i] * x[i]; i += 1; }

        let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
        let inv_rms_v = _mm256_set1_ps(inv_rms);

        // Paso 2: out[i] = x[i] * inv_rms * weight[i]
        i = 0;
        while i + 8 <= n {
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));
            let vw = _mm256_loadu_ps(weight.as_ptr().add(i));
            let normalized = _mm256_mul_ps(vx, inv_rms_v);
            let res = _mm256_mul_ps(normalized, vw);
            _mm256_storeu_ps(out.as_mut_ptr().add(i), res);
            i += 8;
        }
        while i < n { out[i] = x[i] * inv_rms * weight[i]; i += 1; }
    } else {
        // Fallback SSE2
        let mut acc = _mm_setzero_ps();
        let mut i = 0;
        while i + 4 <= n {
            let vx = _mm_loadu_ps(x.as_ptr().add(i));
            acc = _mm_add_ps(acc, _mm_mul_ps(vx, vx));
            i += 4;
        }
        let shuf = _mm_movehdup_ps(acc);
        let sums = _mm_add_ps(acc, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        let mut sum_sq = _mm_cvtss_f32(result);
        while i < n { sum_sq += x[i] * x[i]; i += 1; }

        let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
        let inv_rms_v = _mm_set1_ps(inv_rms);

        i = 0;
        while i + 4 <= n {
            let vx = _mm_loadu_ps(x.as_ptr().add(i));
            let vw = _mm_loadu_ps(weight.as_ptr().add(i));
            let normalized = _mm_mul_ps(vx, inv_rms_v);
            let res = _mm_mul_ps(normalized, vw);
            _mm_storeu_ps(out.as_mut_ptr().add(i), res);
            i += 4;
        }
        while i < n { out[i] = x[i] * inv_rms * weight[i]; i += 1; }
    }
    out
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
pub unsafe fn rms_norm_neon(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / x.len() as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    x.iter().zip(weight.iter()).map(|(&v, &w)| v * inv_rms * w).collect()
}

// =============================================================================
// Tabla de shuffle para decodificación de bases genómicas (2-bit → índice)
// =============================================================================

// Precomputed shuffle masks for decoding 2-bit values to float indices
// Each byte index (0-255) maps to 16 bytes (4 floats)
static mut SHUFFLE_MASK_TABLE: [[u8; 16]; 256] = [[0; 16]; 256];
static mut SHUFFLE_TABLE_INITIALIZED: bool = false;

pub unsafe fn init_shuffle_table() {
    if SHUFFLE_TABLE_INITIALIZED {
        return;
    }
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

// =============================================================================
// genomic_dot_product — Producto punto genómico con decodificación 2-bit
// =============================================================================

#[cfg(target_arch = "aarch64")]
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

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn genomic_dot_product_neon(
    weights: &[u8],
    input: &[f32],
    centroids: &[f32],
    stride: usize,
    n_blocks: usize,
) -> f32 {
    // Usamos SSE (128-bit) porque los centroides operan en bloques de 4 floats
    // _mm_shuffle_epi8 es el equivalente de vqtbl1q_u8 en NEON
    if is_x86_feature_detected!("ssse3") {
        #[allow(static_mut_refs)]
        let table_ptr = SHUFFLE_MASK_TABLE.as_ptr();
        let mut acc = _mm_setzero_ps();

        for j in 0..n_blocks {
            // Cargar los 4 centroides como bytes (16 bytes = 4 x f32)
            let c_v = _mm_loadu_si128(centroids.as_ptr().add(j * 4) as *const __m128i);
            let input_block_ptr = input.as_ptr().add(j * stride * 4);
            let weights_block_ptr = weights.as_ptr().add(j * stride);

            for k in 0..stride {
                let byte = *weights_block_ptr.add(k);
                let mask = _mm_loadu_si128(table_ptr.add(byte as usize) as *const __m128i);
                // pshufb: reordena bytes del centroide usando la máscara de lookup
                let v_vals = _mm_shuffle_epi8(c_v, mask);
                let v_vals_f = _mm_castsi128_ps(v_vals);
                let v_in = _mm_loadu_ps(input_block_ptr.add(k * 4));

                if is_x86_feature_detected!("fma") {
                    acc = _mm_fmadd_ps(v_vals_f, v_in, acc);
                } else {
                    acc = _mm_add_ps(acc, _mm_mul_ps(v_vals_f, v_in));
                }
            }
        }
        // Reducción horizontal de 4 floats
        let shuf = _mm_movehdup_ps(acc);
        let sums = _mm_add_ps(acc, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(result)
    } else {
        // Fallback escalar para CPUs muy antiguas sin SSSE3
        genomic_dot_product_scalar(weights, input, centroids, stride, n_blocks)
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
pub unsafe fn genomic_dot_product_neon(
    weights: &[u8],
    input: &[f32],
    centroids: &[f32],
    stride: usize,
    n_blocks: usize,
) -> f32 {
    genomic_dot_product_scalar(weights, input, centroids, stride, n_blocks)
}

/// Implementación escalar del producto punto genómico (fallback universal)
#[inline(always)]
pub unsafe fn genomic_dot_product_scalar(
    weights: &[u8],
    input: &[f32],
    centroids: &[f32],
    stride: usize,
    n_blocks: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for j in 0..n_blocks {
        let input_block_ptr = input.as_ptr().add(j * stride * 4);
        let weights_block_ptr = weights.as_ptr().add(j * stride);
        for k in 0..stride {
            let byte = *weights_block_ptr.add(k);
            for b in 0..4usize {
                let shift = (3 - b) * 2;
                let bits = (byte >> shift) & 0b11;
                let c_idx = (bits ^ (bits >> 1)) as usize;
                let val = *centroids.as_ptr().add(j * 4 + c_idx);
                sum += val * *input_block_ptr.add(k * 4 + b);
            }
        }
    }
    sum
}

// =============================================================================
// calculate_distance_lut — Distancia LUT para búsqueda HNSW (solo aarch64)
// =============================================================================

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn calculate_distance_lut_neon(
    lut_base: &[f32],
    lut_epi: &[f32],
    lut_tri: &[f32],
    strand: &[u8],
    epi_strand: &[u8],
    tri_strand: &[u8],
    mask: &[u8],
    n_dims: usize,
) -> f32 {
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
            if mode == 0 {
                d_v[j] = *lut_base.get(dims * 4 + b_idx).unwrap_or(&0.0);
            } else if mode == 1 {
                let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                let e_idx = (eb ^ (eb >> 1)) as usize;
                d_v[j] = *lut_epi
                    .get(dims * 16 + (b_idx << 2 | e_idx))
                    .unwrap_or(&0.0);
            } else {
                let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                let tb = (*tri_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                let e_idx = (eb ^ (eb >> 1)) as usize;
                let t_idx = (tb ^ (tb >> 1)) as usize;
                d_v[j] = *lut_tri
                    .get(dims * 64 + (b_idx << 4 | e_idx << 2 | t_idx))
                    .unwrap_or(&0.0);
            }
            dims += 1;
        }
        sum_v = vaddq_f32(sum_v, vld1q_f32(d_v.as_ptr()));
    }
    let mut total = vaddvq_f32(sum_v);
    while dims < n_dims {
        let i = dims / 4;
        let mode = *mask.get(i).unwrap_or(&0);
        let shift = (3 - (dims % 4)) * 2;
        let bb = (*strand.get(i).unwrap_or(&0) >> shift) & 0b11;
        let b_idx = (bb ^ (bb >> 1)) as usize;
        if mode == 0 {
            total += *lut_base.get(dims * 4 + b_idx).unwrap_or(&0.0);
        } else if mode == 1 {
            let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
            let e_idx = (eb ^ (eb >> 1)) as usize;
            total += *lut_epi
                .get(dims * 16 + (b_idx << 2 | e_idx))
                .unwrap_or(&0.0);
        } else {
            let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
            let tb = (*tri_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
            let e_idx = (eb ^ (eb >> 1)) as usize;
            let t_idx = (tb ^ (tb >> 1)) as usize;
            total += *lut_tri
                .get(dims * 64 + (b_idx << 4 | e_idx << 2 | t_idx))
                .unwrap_or(&0.0);
        }
        dims += 1;
    }
    total.sqrt()
}
