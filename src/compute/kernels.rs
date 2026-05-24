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

use rayon::prelude::*;

// =============================================================================
// dot_product — Producto punto vectorizado universal
// =============================================================================

/// # Safety
/// Esta función es unsafe porque utiliza instrucciones intrínsecas SIMD y realiza
/// acceso directo a memoria mediante punteros. El llamador debe asegurar que:
/// 1. Los slices `a` y `b` tengan la misma longitud.
/// 2. Los punteros derivados de los slices sean válidos para lectura.
#[inline(always)]
pub unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
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

    #[cfg(target_arch = "x86_64")]
    {
        let n = a.len();
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;
            while i + 8 <= n {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                acc = _mm256_fmadd_ps(va, vb, acc);
                i += 8;
            }
            let hi = _mm256_extractf128_ps(acc, 1);
            let lo = _mm256_castps256_ps128(acc);
            let sum128 = _mm_add_ps(lo, hi);
            let shuf = _mm_movehdup_ps(sum128);
            let sums = _mm_add_ps(sum128, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            let result = _mm_add_ss(sums, shuf2);
            let mut sum = _mm_cvtss_f32(result);
            while i < n { sum += a[i] * b[i]; i += 1; }
            sum
        } else {
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
    {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

// Alias para compatibilidad con rama windows
/// # Safety
/// Ver `dot_product`.
#[inline(always)]
pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    dot_product(a, b)
}

// =============================================================================
// rms_norm — Normalización RMS vectorizada universal
// =============================================================================

/// # Safety
/// Esta función es unsafe porque utiliza instrucciones intrínsecas SIMD y realiza
/// acceso directo a memoria mediante punteros. El llamador debe asegurar que:
/// 1. Los slices `x` y `weight` tengan la misma longitud.
/// 2. El slice `out` (interno) tenga el tamaño suficiente.
#[inline(always)]
pub unsafe fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut out = vec![0.0f32; n];

    #[cfg(target_arch = "aarch64")]
    {
        let mut sum_v = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 4 <= n {
            let vx = vld1q_f32(x.as_ptr().add(i));
            sum_v = vfmaq_f32(sum_v, vx, vx);
            i += 4;
        }
        let mut sum_sq = vaddvq_f32(sum_v);
        while i < n { sum_sq += x[i] * x[i]; i += 1; }
        let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
        let inv_rms_v = vdupq_n_f32(inv_rms);
        i = 0;
        while i + 4 <= n {
            let vx = vld1q_f32(x.as_ptr().add(i));
            let vw = vld1q_f32(weight.as_ptr().add(i));
            let res = vmulq_f32(vmulq_f32(vx, inv_rms_v), vw);
            vst1q_f32(out.as_mut_ptr().add(i), res);
            i += 4;
        }
        while i < n { out[i] = (x[i] * inv_rms) * weight[i]; i += 1; }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;
            while i + 8 <= n {
                let vx = _mm256_loadu_ps(x.as_ptr().add(i));
                acc = _mm256_fmadd_ps(vx, vx, acc);
                i += 8;
            }
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
            i = 0;
            while i + 8 <= n {
                let vx = _mm256_loadu_ps(x.as_ptr().add(i));
                let vw = _mm256_loadu_ps(weight.as_ptr().add(i));
                let res = _mm256_mul_ps(_mm256_mul_ps(vx, inv_rms_v), vw);
                _mm256_storeu_ps(out.as_mut_ptr().add(i), res);
                i += 8;
            }
            while i < n { out[i] = x[i] * inv_rms * weight[i]; i += 1; }
        } else {
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
                let res = _mm_mul_ps(_mm_mul_ps(vx, inv_rms_v), vw);
                _mm_storeu_ps(out.as_mut_ptr().add(i), res);
                i += 4;
            }
            while i < n { out[i] = x[i] * inv_rms * weight[i]; i += 1; }
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / x.len() as f32 + eps).sqrt();
        for i in 0..n { out[i] = x[i] * inv_rms * weight[i]; }
    }
    out
}

// Alias para compatibilidad con rama windows
/// # Safety
/// Ver `rms_norm`.
#[inline(always)]
pub unsafe fn rms_norm_neon(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    rms_norm(x, weight, eps)
}

// =============================================================================
// swiglu — Activación SwiGLU vectorizada universal con Estabilización Dinámica
// =============================================================================

#[inline(always)]
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    let _n = gate.len();
    out.par_iter_mut()
        .zip(gate.par_iter())
        .zip(up.par_iter())
        .for_each(|((o, &g), &u)| {
            // Estabilización de SwiGLU (Silu gating)
            // Limitamos el rango dinámico para evitar que el ruido de cuantización
            // de 2 bits se magnifique en las colas de la exponencial.
            let g_safe = g.clamp(-64.0, 64.0);
            let sigmoid = if g_safe >= 0.0 {
                1.0 / (1.0 + (-g_safe).exp())
            } else {
                let ex = g_safe.exp();
                ex / (1.0 + ex)
            };
            let silu = g * sigmoid;
            
            // Clamping adaptativo: reduce la probabilidad de explosión de gradiente/activación
            // en modelos profundos (>24 bloques).
            *o = (silu * u).clamp(-96.0, 96.0);
        });
}

/// Versión balanceada de SwiGLU que compensa el sesgo (bias) introducido 
/// por la cuantización asimétrica de 2 bits.
#[inline(always)]
pub fn swiglu_balanced(gate: &[f32], up: &[f32], out: &mut [f32], h_scale: f32) {
    out.par_iter_mut()
        .zip(gate.par_iter())
        .zip(up.par_iter())
        .for_each(|((o, &g), &u)| {
            let g_safe = g.clamp(-64.0, 64.0);
            let sigmoid = if g_safe >= 0.0 {
                1.0 / (1.0 + (-g_safe).exp())
            } else {
                let ex = g_safe.exp();
                ex / (1.0 + ex)
            };
            
            // Aplicamos h_scale como un factor de temperancia para suavizar 
            // la respuesta ante inputs ruidosos.
            let silu = g * sigmoid;
            *o = (silu * u * h_scale).clamp(-96.0, 96.0);
        });
}

#[inline(always)]
pub fn geglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    out.par_iter_mut()
        .zip(gate.par_iter())
        .zip(up.par_iter())
        .for_each(|((o, &g), &u)| {
            // Estabilización de GeGLU
            let g_safe = g.clamp(-20.0, 20.0);
            let tanh_inner = 0.7978846f32 * (g_safe + 0.044715f32 * g_safe * g_safe * g_safe);
            let gelu = 0.5f32 * g_safe * (1.0f32 + tanh_inner.tanh());
            *o = (gelu * u).clamp(-128.0, 128.0);
        });
}

#[inline(always)]
pub fn relu_glu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    out.par_iter_mut()
        .zip(gate.par_iter())
        .zip(up.par_iter())
        .for_each(|((o, &g), &u)| {
            *o = (g.max(0.0) * u).clamp(-128.0, 128.0);
        });
}

// =============================================================================
// Tabla de shuffle para decodificación de bases genómicas (2-bit → índice)
// =============================================================================

static mut SHUFFLE_MASK_TABLE: [[u8; 16]; 256] = [[0; 16]; 256];
static mut SHUFFLE_TABLE_INITIALIZED: bool = false;

/// # Safety
/// Esta función accede y modifica variables estáticas globales mutables sin sincronización.
/// Debe ser llamada una sola vez durante la inicialización del programa o garantizando
/// que no haya condiciones de carrera.
pub unsafe fn init_shuffle_table() {
    if SHUFFLE_TABLE_INITIALIZED { return; }
    for b in 0..256usize {
        for i in 0..4 {
            let shift = (3 - i) * 2;
            let bits = (b >> shift) & 0b11;
            let idx = (bits ^ (bits >> 1)) as u8;
            for j in 0..4 {
                SHUFFLE_MASK_TABLE[b][(i * 4 + j) as usize] = idx * 4 + j as u8;
            }
        }
    }
    SHUFFLE_TABLE_INITIALIZED = true;
}

// =============================================================================
// genomic_dot_product — Producto punto genómico universal
// =============================================================================

/// # Safety
/// Esta función es unsafe porque utiliza instrucciones intrínsecas SIMD, accede a
/// tablas de shuffle estáticas y realiza aritmética de punteros sin comprobación de límites.
#[inline(always)]
pub unsafe fn genomic_dot_product(
    weights: &[u8],
    input: &[f32],
    centroids: &[f32],
    stride: usize,
    n_blocks: usize,
    anchors: &[half::f16],
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        let mut sum_v = vdupq_n_f32(0.0);
        let has_anchors = !anchors.is_empty();
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
                let mut v_weights = vreinterpretq_f32_u8(v_vals);
                
                if has_anchors {
                    let a_ptr = anchors.as_ptr().add(j * stride * 4 + k * 4);
                    // Optimized f16 -> f32 conversion using NEON
                    let a_v16 = vld1_u16(a_ptr as *const u16);
                    let v_anchors = vcvt_f32_f16(vreinterpret_f16_u16(a_v16));
                    v_weights = vaddq_f32(v_weights, v_anchors);
                }

                sum_v = vfmaq_f32(sum_v, v_weights, vld1q_f32(input_block_ptr.add(k * 4)));
            }
        }
        vaddvq_f32(sum_v)
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("ssse3") {
            #[allow(static_mut_refs)]
            let table_ptr = SHUFFLE_MASK_TABLE.as_ptr();
            let mut acc = _mm_setzero_ps();
            let has_anchors = !anchors.is_empty();
            let has_f16c = is_x86_feature_detected!("f16c");

            for j in 0..n_blocks {
                let c_v = _mm_loadu_si128(centroids.as_ptr().add(j * 4) as *const __m128i);
                let input_block_ptr = input.as_ptr().add(j * stride * 4);
                let weights_block_ptr = weights.as_ptr().add(j * stride);
                for k in 0..stride {
                    let byte = *weights_block_ptr.add(k);
                    let mask = _mm_loadu_si128(table_ptr.add(byte as usize) as *const __m128i);
                    let mut v_vals_f = _mm_castsi128_ps(_mm_shuffle_epi8(c_v, mask));
                    
                    if has_anchors {
                        let a_ptr = anchors.as_ptr().add(j * stride * 4 + k * 4);
                        if has_f16c {
                            // F16C optimization
                            let v_anchors = _mm_cvtph_ps(_mm_loadl_epi64(a_ptr as *const __m128i));
                            v_vals_f = _mm_add_ps(v_vals_f, v_anchors);
                        } else {
                            let mut a_f32 = [0.0f32; 4];
                            for s in 0..4 { a_f32[s] = (*a_ptr.add(s)).to_f32(); }
                            v_vals_f = _mm_add_ps(v_vals_f, _mm_loadu_ps(a_f32.as_ptr()));
                        }
                    }

                    let v_in = _mm_loadu_ps(input_block_ptr.add(k * 4));
                    if is_x86_feature_detected!("fma") { acc = _mm_fmadd_ps(v_vals_f, v_in, acc); }
                    else { acc = _mm_add_ps(acc, _mm_mul_ps(v_vals_f, v_in)); }
                }
            }
            let shuf = _mm_movehdup_ps(acc);
            let sums = _mm_add_ps(acc, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
        } else {
            genomic_dot_product_scalar(weights, input, centroids, stride, n_blocks, anchors)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        genomic_dot_product_scalar(weights, input, centroids, stride, n_blocks, anchors)
    }
}

// Alias para compatibilidad con rama windows
/// # Safety
/// Ver `genomic_dot_product`.
#[inline(always)]
pub unsafe fn genomic_dot_product_neon(
    weights: &[u8],
    input: &[f32],
    centroids: &[f32],
    stride: usize,
    n_blocks: usize,
) -> f32 {
    genomic_dot_product(weights, input, centroids, stride, n_blocks, &[])
}

/// # Safety
/// Esta función utiliza `get_unchecked` y aritmética de punteros para maximizar
/// el rendimiento. El llamador debe garantizar que los tamaños de los slices
/// sean coherentes con `n_blocks` y `stride`.
#[inline(always)]
pub unsafe fn genomic_dot_product_scalar(
    weights: &[u8],
    input: &[f32],
    centroids: &[f32],
    stride: usize,
    n_blocks: usize,
    anchors: &[half::f16],
) -> f32 {
    let mut sum = 0.0f32;
    let has_anchors = !anchors.is_empty();

    for j in 0..n_blocks {
        let input_block_ptr = input.as_ptr().add(j * stride * 4);
        let weights_block_ptr = weights.as_ptr().add(j * stride);
        let centroids_ptr = centroids.as_ptr().add(j * 4);
        
        for k in 0..stride {
            let byte = *weights_block_ptr.add(k);
            let anchor_offset = j * stride * 4 + k * 4;
            
            for b in 0..4usize {
                let shift = (3 - b) * 2;
                let bits = (byte >> shift) & 0b11;
                let c_idx = (bits ^ (bits >> 1)) as usize;
                
                let mut weight_val = *centroids_ptr.add(c_idx);
                
                if has_anchors {
                    weight_val += (*anchors.get_unchecked(anchor_offset + b)).to_f32();
                }
                
                sum += weight_val * *input_block_ptr.add(k * 4 + b);
            }
        }
    }
    sum
}

// =============================================================================
// calculate_distance_lut — Distancia LUT universal
// =============================================================================

/// # Safety
/// Esta función realiza accesos directos a memoria mediante `get_unchecked` (implícito en la lógica nativa)
/// y asume que todos los strands y máscaras tienen longitudes coherentes con `n_dims`.
#[inline(always)]
pub unsafe fn calculate_distance_lut(
    lut_base: &[f32],
    lut_epi: &[f32],
    lut_tri: &[f32],
    strand: &[u8],
    epi_strand: &[u8],
    tri_strand: &[u8],
    mask: &[u8],
    n_dims: usize,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
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
                else if mode == 1 {
                    let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                    d_v[j] = *lut_epi.get(dims * 16 + (b_idx << 2 | (eb ^ (eb >> 1)) as usize)).unwrap_or(&0.0);
                } else {
                    let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                    let tb = (*tri_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                    d_v[j] = *lut_tri.get(dims * 64 + (b_idx << 4 | ((eb ^ (eb >> 1)) as usize) << 2 | (tb ^ (tb >> 1)) as usize)).unwrap_or(&0.0);
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
            if mode == 0 { total += *lut_base.get(dims * 4 + b_idx).unwrap_or(&0.0); }
            else if mode == 1 {
                let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                total += *lut_epi.get(dims * 16 + (b_idx << 2 | (eb ^ (eb >> 1)) as usize)).unwrap_or(&0.0);
            } else {
                let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                let tb = (*tri_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                total += *lut_tri.get(dims * 64 + (b_idx << 4 | ((eb ^ (eb >> 1)) as usize) << 2 | (tb ^ (tb >> 1)) as usize)).unwrap_or(&0.0);
            }
            dims += 1;
        }
        total.sqrt()
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut total = 0.0f32;
        for dims in 0..n_dims {
            let i = dims / 4;
            let mode = *mask.get(i).unwrap_or(&0);
            let shift = (3 - (dims % 4)) * 2;
            let bb = (*strand.get(i).unwrap_or(&0) >> shift) & 0b11;
            let b_idx = (bb ^ (bb >> 1)) as usize;
            if mode == 0 { total += *lut_base.get(dims * 4 + b_idx).unwrap_or(&0.0); }
            else if mode == 1 {
                let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                total += *lut_epi.get(dims * 16 + (b_idx << 2 | (eb ^ (eb >> 1)) as usize)).unwrap_or(&0.0);
            } else {
                let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                let tb = (*tri_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                total += *lut_tri.get(dims * 64 + (b_idx << 4 | ((eb ^ (eb >> 1)) as usize) << 2 | (tb ^ (tb >> 1)) as usize)).unwrap_or(&0.0);
            }
        }
        total.sqrt()
    }
}

// Alias para compatibilidad con rama windows
/// # Safety
/// Ver `calculate_distance_lut`.
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
    calculate_distance_lut(lut_base, lut_epi, lut_tri, strand, epi_strand, tri_strand, mask, n_dims)
}
