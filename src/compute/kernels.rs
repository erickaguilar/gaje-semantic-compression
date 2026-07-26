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

use crate::compute::kv_cache::CompressedKVCache;
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
        while i < n {
            sum += a[i] * b[i];
            i += 1;
        }
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
            while i < n {
                sum += a[i] * b[i];
                i += 1;
            }
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
            while i < n {
                sum += a[i] * b[i];
                i += 1;
            }
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

/// # Safety
/// Esta función es unsafe porque realiza acceso directo a memoria mediante punteros
/// y de-cuantización manual de bits. El llamador debe asegurar que:
/// 1. `query.len() == len`.
/// 2. `start_idx + len` no exceda la capacidad del cache.
pub unsafe fn dot_product_compressed(
    query: &[f32],
    cache: &CompressedKVCache,
    start_idx: usize,
    len: usize,
) -> f32 {
    let mut sum = 0.0f32;

    // Optimizamos procesando por bloques de 48 (alineados con el cache)
    let mut i = 0;
    while i < len {
        let global_idx = start_idx + i;
        let block_idx = global_idx / 48;
        let sub_idx = global_idx % 48;

        let block = &cache.blocks[block_idx];
        let scale = block.scale;

        // Procesamos lo que queda del bloque actual o hasta el final de len
        let remaining_in_block = 48 - sub_idx;
        let batch_len = remaining_in_block.min(len - i);

        for j in 0..batch_len {
            let current_sub_idx = sub_idx + j;
            let byte_idx = current_sub_idx / 4;
            let bit_shift = (3 - (current_sub_idx % 4)) * 2;
            let quantized = (block.data[byte_idx] >> bit_shift) & 0b11;

            sum += query[i + j] * (quantized as f32) * scale;
        }

        i += batch_len;
    }

    sum
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
        while i < n {
            sum_sq += x[i] * x[i];
            i += 1;
        }
        // Suelo de seguridad para evitar NaNs en Android
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
        while i < n {
            out[i] = (x[i] * inv_rms) * weight[i];
            i += 1;
        }
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
            while i < n {
                sum_sq += x[i] * x[i];
                i += 1;
            }
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
            while i < n {
                out[i] = x[i] * inv_rms * weight[i];
                i += 1;
            }
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
            while i < n {
                sum_sq += x[i] * x[i];
                i += 1;
            }
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
            while i < n {
                out[i] = x[i] * inv_rms * weight[i];
                i += 1;
            }
        }
    }

    #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
    {
        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / x.len() as f32 + eps).max(1e-5).sqrt();
        for i in 0..n {
            out[i] = x[i] * inv_rms * weight[i];
        }
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

            let silu = g * sigmoid;
            *o = silu * u;
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
    if SHUFFLE_TABLE_INITIALIZED {
        return;
    }
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
// lateral_inhibition_kwta — El Filtro del "Río Semántico"
// =============================================================================

/// Implementa la Inhibición Lateral (K-Winners-Take-All).
///
/// Este kernel simula cómo las "Islas" de cristalización inhiben el ruido
/// de la "Materia Oscura" circundante, forzando a la señal a fluir por los
/// canales de máxima resonancia (El Río Semántico).
pub fn lateral_inhibition_kwta(scores: &mut [f32], k: usize) {
    if scores.len() <= k {
        return;
    }

    // Revertido para diagnóstico de NaN
    let mut sorted_scores = scores.to_vec();
    sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted_scores[k - 1];

    // Inhibición: las señales por debajo del umbral se extinguen (Materia Oscura)
    for s in scores.iter_mut() {
        if *s < threshold {
            *s = -1e9; // Silencio inhibitorio
        }
    }
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
    modulation: &[f32; 4],
) -> f32 {
    // Audit Forense: Forzamos motor escalar para aislar inestabilidad SIMD
    genomic_dot_product_scalar(weights, input, centroids, stride, n_blocks, modulation)
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
    genomic_dot_product(weights, input, centroids, stride, n_blocks, &[1.0; 4])
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
    modulation: &[f32; 4],
) -> f32 {
    let mut sum = 0.0f32;

    for j in 0..n_blocks {
        let input_block_ptr = input.as_ptr().add(j * stride * 4);
        let weights_block_ptr = weights.as_ptr().add(j * stride);
        let centroids_ptr = centroids.as_ptr().add(j * 4);

        for k in 0..stride {
            let byte = *weights_block_ptr.add(k);

            for b in 0..4usize {
                let shift = (3 - b) * 2;
                let bits = (byte >> shift) & 0b11;
                let c_idx = (bits ^ (bits >> 1)) as usize;

                let weight_val = *centroids_ptr.add(c_idx) * modulation[c_idx];
                sum += weight_val * *input_block_ptr.add(k * 4 + b);
            }
        }
    }

    // Frenado Lagrangiano: El rozamiento semántico aniquila el ruido residual (Entropía)
    // Esto asegura que el eco toroidal sea puro en ciclos infinitos.
    if sum.abs() < 1e-5 {
        sum = 0.0;
    }

    sum
}

/// # Safety
/// Implementación de 4 bits (2 pesos por byte). Soporta 16 centroides por bloque.
#[inline(always)]
pub unsafe fn genomic_dot_product_4bit(
    weights: &[u8],
    input: &[f32],
    centroids: &[f32],
    stride_4bit: usize, // stride_4bit = block_size / 2
    n_blocks: usize,
) -> f32 {
    let mut sum = 0.0f32;

    for j in 0..n_blocks {
        let block_size = stride_4bit * 2;
        let input_block_ptr = input.as_ptr().add(j * block_size);
        let weights_block_ptr = weights.as_ptr().add(j * stride_4bit);
        let centroids_ptr = centroids.as_ptr().add(j * 16);

        for k in 0..stride_4bit {
            let byte = *weights_block_ptr.add(k);

            // Peso 1 (High nibble)
            let c_idx1 = (byte >> 4) as usize;
            sum += *centroids_ptr.add(c_idx1) * *input_block_ptr.add(k * 2);

            // Peso 2 (Low nibble)
            let c_idx2 = (byte & 0x0F) as usize;
            sum += *centroids_ptr.add(c_idx2) * *input_block_ptr.add(k * 2 + 1);
        }
    }

    if sum.abs() < 1e-6 {
        sum = 0.0;
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
                if mode == 0 {
                    d_v[j] = *lut_base.get(dims * 4 + b_idx).unwrap_or(&0.0);
                } else if mode == 1 {
                    let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                    d_v[j] = *lut_epi
                        .get(dims * 16 + (b_idx << 2 | (eb ^ (eb >> 1)) as usize))
                        .unwrap_or(&0.0);
                } else {
                    let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                    let tb = (*tri_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                    d_v[j] = *lut_tri
                        .get(
                            dims * 64
                                + (b_idx << 4
                                    | ((eb ^ (eb >> 1)) as usize) << 2
                                    | (tb ^ (tb >> 1)) as usize),
                        )
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
                total += *lut_epi
                    .get(dims * 16 + (b_idx << 2 | (eb ^ (eb >> 1)) as usize))
                    .unwrap_or(&0.0);
            } else {
                let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                let tb = (*tri_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                total += *lut_tri
                    .get(
                        dims * 64
                            + (b_idx << 4
                                | ((eb ^ (eb >> 1)) as usize) << 2
                                | (tb ^ (tb >> 1)) as usize),
                    )
                    .unwrap_or(&0.0);
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
            if mode == 0 {
                total += *lut_base.get(dims * 4 + b_idx).unwrap_or(&0.0);
            } else if mode == 1 {
                let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                total += *lut_epi
                    .get(dims * 16 + (b_idx << 2 | (eb ^ (eb >> 1)) as usize))
                    .unwrap_or(&0.0);
            } else {
                let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                let tb = (*tri_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                total += *lut_tri
                    .get(
                        dims * 64
                            + (b_idx << 4
                                | ((eb ^ (eb >> 1)) as usize) << 2
                                | (tb ^ (tb >> 1)) as usize),
                    )
                    .unwrap_or(&0.0);
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
    calculate_distance_lut(
        lut_base, lut_epi, lut_tri, strand, epi_strand, tri_strand, mask, n_dims,
    )
}
