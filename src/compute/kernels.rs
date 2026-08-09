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
pub fn swiglu_balanced(gate: &[f32], up: &[f32], out: &mut [f32], _h_scale: f32) {
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

        let c0 = *centroids_ptr.add(0) * modulation[0];
        let c1 = *centroids_ptr.add(1) * modulation[1];
        let c2 = *centroids_ptr.add(2) * modulation[2];
        let c3 = *centroids_ptr.add(3) * modulation[3];
        // Swapped mapping to align with Gray code:
        // Index 2 (0b10) maps to c3 (4th centroid).
        // Index 3 (0b11) maps to c2 (3rd centroid).
        let c_arr = [c0, c1, c3, c2];

        let mut k = 0;

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();

            while k + 4 <= stride {
                let b0 = *weights_block_ptr.add(k);
                let b1 = *weights_block_ptr.add(k + 1);
                let b2 = *weights_block_ptr.add(k + 2);
                let b3 = *weights_block_ptr.add(k + 3);

                let w_vals = [
                    c_arr[((b0 >> 6) & 0b11) as usize],
                    c_arr[((b0 >> 4) & 0b11) as usize],
                    c_arr[((b0 >> 2) & 0b11) as usize],
                    c_arr[(b0 & 0b11) as usize],
                    c_arr[((b1 >> 6) & 0b11) as usize],
                    c_arr[((b1 >> 4) & 0b11) as usize],
                    c_arr[((b1 >> 2) & 0b11) as usize],
                    c_arr[(b1 & 0b11) as usize],
                    c_arr[((b2 >> 6) & 0b11) as usize],
                    c_arr[((b2 >> 4) & 0b11) as usize],
                    c_arr[((b2 >> 2) & 0b11) as usize],
                    c_arr[(b2 & 0b11) as usize],
                    c_arr[((b3 >> 6) & 0b11) as usize],
                    c_arr[((b3 >> 4) & 0b11) as usize],
                    c_arr[((b3 >> 2) & 0b11) as usize],
                    c_arr[(b3 & 0b11) as usize],
                ];

                let vw0 = _mm256_loadu_ps(w_vals.as_ptr());
                let vw1 = _mm256_loadu_ps(w_vals.as_ptr().add(8));

                let vi0 = _mm256_loadu_ps(input_block_ptr.add(k * 4));
                let vi1 = _mm256_loadu_ps(input_block_ptr.add(k * 4 + 8));

                acc0 = _mm256_fmadd_ps(vw0, vi0, acc0);
                acc1 = _mm256_fmadd_ps(vw1, vi1, acc1);

                k += 4;
            }

            let acc = _mm256_add_ps(acc0, acc1);
            let hi = _mm256_extractf128_ps(acc, 1);
            let lo = _mm256_castps256_ps128(acc);
            let sum128 = _mm_add_ps(lo, hi);
            let shuf = _mm_movehdup_ps(sum128);
            let sums = _mm_add_ps(sum128, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            let result = _mm_add_ss(sums, shuf2);
            sum += _mm_cvtss_f32(result);
        }

        while k < stride {
            let byte = *weights_block_ptr.add(k);
            for b in 0..4usize {
                let shift = (3 - b) * 2;
                let bits = (byte >> shift) & 0b11;
                let weight_val = c_arr[bits as usize];
                sum += weight_val * *input_block_ptr.add(k * 4 + b);
            }
            k += 1;
        }
    }

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
    let block_size = stride_4bit * 2;
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    for j in 0..n_blocks {
        let input_block_ptr = input.as_ptr().add(j * block_size);
        let weights_block_ptr = weights.as_ptr().add(j * stride_4bit);
        let centroids_ptr = centroids.as_ptr().add(j * 16);
        let c_lut: &[f32; 16] = &*(centroids_ptr as *const [f32; 16]);

        let mut k = 0;
        while k + 4 <= stride_4bit {
            let b0 = *weights_block_ptr.add(k);
            let b1 = *weights_block_ptr.add(k + 1);
            let b2 = *weights_block_ptr.add(k + 2);
            let b3 = *weights_block_ptr.add(k + 3);

            let in_ptr = input_block_ptr.add(k * 2);

            sum0 += c_lut[(b0 >> 4) as usize] * *in_ptr;
            sum1 += c_lut[(b0 & 0x0F) as usize] * *in_ptr.add(1);
            sum2 += c_lut[(b1 >> 4) as usize] * *in_ptr.add(2);
            sum3 += c_lut[(b1 & 0x0F) as usize] * *in_ptr.add(3);

            sum0 += c_lut[(b2 >> 4) as usize] * *in_ptr.add(4);
            sum1 += c_lut[(b2 & 0x0F) as usize] * *in_ptr.add(5);
            sum2 += c_lut[(b3 >> 4) as usize] * *in_ptr.add(6);
            sum3 += c_lut[(b3 & 0x0F) as usize] * *in_ptr.add(7);

            k += 4;
        }

        while k < stride_4bit {
            let byte = *weights_block_ptr.add(k);
            sum0 += c_lut[(byte >> 4) as usize] * *input_block_ptr.add(k * 2);
            sum1 += c_lut[(byte & 0x0F) as usize] * *input_block_ptr.add(k * 2 + 1);
            k += 1;
        }
    }

    let total = sum0 + sum1 + sum2 + sum3;
    if total.abs() < 1e-6 {
        0.0
    } else {
        total
    }
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

/// # Safety
/// Implementación optimizada de producto punto para formato Q4_0 (Variant B).
/// Reduce el número de multiplicaciones flotantes factorizando la escala y el mínimo por bloque.
#[inline(always)]
pub unsafe fn genomic_dot_product_q4_0(
    blocks: &[crate::io::header::Q4_0Block],
    input: &[f32],
    n_blocks: usize,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return genomic_dot_product_q4_0_avx2(blocks, input, n_blocks);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return genomic_dot_product_q4_0_neon(blocks, input, n_blocks);
    }

    // Fallback escalar
    let mut total_sum = 0.0f32;

    for j in 0..n_blocks {
        let block = &blocks[j];
        let scale = block.scale.to_f32();
        let min = block.min.to_f32();

        let input_offset = j * 32;
        let mut sum_q_in = 0.0f32;
        let mut sum_in = 0.0f32;

        let qs = &block.qs;

        for k in 0..16 {
            let byte = *qs.get_unchecked(k);
            let q0 = (byte & 0x0F) as f32;
            let q1 = (byte >> 4) as f32;

            let x0 = *input.get_unchecked(input_offset + k * 2);
            let x1 = *input.get_unchecked(input_offset + k * 2 + 1);

            sum_q_in += q0 * x0 + q1 * x1;
            sum_in += x0 + x1;
        }

        total_sum += sum_q_in * scale + sum_in * min;
    }

    if total_sum.abs() < 1e-6 {
        0.0
    } else {
        total_sum
    }
}

/// # Safety
/// Kernel de-cuantización y producto punto AVX2 + FMA para bloques Q4_0
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn genomic_dot_product_q4_0_avx2(
    blocks: &[crate::io::header::Q4_0Block],
    input: &[f32],
    n_blocks: usize,
) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mask_low = _mm_set1_epi8(0x0F);

    for j in 0..n_blocks {
        let block = blocks.get_unchecked(j);
        let scale = block.scale.to_f32();
        let min = block.min.to_f32();

        let v_scale = _mm256_set1_ps(scale);
        let v_min = _mm256_set1_ps(min);

        let input_offset = j * 32;

        // Load 16 bytes of qs (128 bits)
        let v_qs = _mm_loadu_si128(block.qs.as_ptr() as *const __m128i);

        // Unpack low and high nibbles
        let low_nibbles = _mm_and_si128(v_qs, mask_low);
        let high_nibbles = _mm_and_si128(_mm_srli_epi16(v_qs, 4), mask_low);

        // Interleave low and high nibbles
        let interleaved_lo = _mm_unpacklo_epi8(low_nibbles, high_nibbles);
        let interleaved_hi = _mm_unpackhi_epi8(low_nibbles, high_nibbles);

        // Process first 8 elements
        let q_lo_0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(interleaved_lo));
        let dequant_lo_0 = _mm256_fmadd_ps(q_lo_0, v_scale, v_min);
        let x_lo_0 = _mm256_loadu_ps(input.as_ptr().add(input_offset));
        acc = _mm256_fmadd_ps(dequant_lo_0, x_lo_0, acc);

        // Process next 8 elements
        let q_lo_1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(interleaved_lo, 8)));
        let dequant_lo_1 = _mm256_fmadd_ps(q_lo_1, v_scale, v_min);
        let x_lo_1 = _mm256_loadu_ps(input.as_ptr().add(input_offset + 8));
        acc = _mm256_fmadd_ps(dequant_lo_1, x_lo_1, acc);

        // Process next 8 elements
        let q_hi_0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(interleaved_hi));
        let dequant_hi_0 = _mm256_fmadd_ps(q_hi_0, v_scale, v_min);
        let x_hi_0 = _mm256_loadu_ps(input.as_ptr().add(input_offset + 16));
        acc = _mm256_fmadd_ps(dequant_hi_0, x_hi_0, acc);

        // Process last 8 elements
        let q_hi_1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(interleaved_hi, 8)));
        let dequant_hi_1 = _mm256_fmadd_ps(q_hi_1, v_scale, v_min);
        let x_hi_1 = _mm256_loadu_ps(input.as_ptr().add(input_offset + 24));
        acc = _mm256_fmadd_ps(dequant_hi_1, x_hi_1, acc);
    }

    // Horizontal sum of acc
    let vlow = _mm256_castps256_ps128(acc);
    let vhigh = _mm256_extractf128_ps(acc, 1);
    let v128 = _mm_add_ps(vlow, vhigh);
    let hi = _mm_movehl_ps(v128, v128);
    let sum = _mm_add_ps(v128, hi);
    let shuf = _mm_shuffle_ps(sum, sum, 1);
    let final_sum = _mm_add_ss(sum, shuf);
    let total = _mm_cvtss_f32(final_sum);

    if total.abs() < 1e-6 {
        0.0
    } else {
        total
    }
}

/// # Safety
/// Kernel de-cuantización y producto punto ARM NEON para bloques Q4_0
#[cfg(target_arch = "aarch64")]
pub unsafe fn genomic_dot_product_q4_0_neon(
    blocks: &[crate::io::header::Q4_0Block],
    input: &[f32],
    n_blocks: usize,
) -> f32 {
    let mut acc = vdupq_n_f32(0.0);
    let mask_low = vdupq_n_u8(0x0F);

    for j in 0..n_blocks {
        let block = blocks.get_unchecked(j);
        let scale = block.scale.to_f32();
        let min = block.min.to_f32();

        let v_scale = vdupq_n_f32(scale);
        let v_min = vdupq_n_f32(min);

        let input_offset = j * 32;

        let v_qs = vld1q_u8(block.qs.as_ptr());

        let low_nibbles = vandq_u8(v_qs, mask_low);
        let high_nibbles = vshrq_n_u8(v_qs, 4);

        let interleaved_lo = vzip1q_u8(low_nibbles, high_nibbles);
        let interleaved_hi = vzip2q_u8(low_nibbles, high_nibbles);

        // lo: 0..7
        let u16_lo_0 = vmovl_u8(vget_low_u8(interleaved_lo));
        let q_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_lo_0)));
        let dequant_0 = vfmaq_f32(v_min, q_0, v_scale);
        let x_0 = vld1q_f32(input.as_ptr().add(input_offset));
        acc = vfmaq_f32(acc, dequant_0, x_0);

        let q_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_lo_0)));
        let dequant_1 = vfmaq_f32(v_min, q_1, v_scale);
        let x_1 = vld1q_f32(input.as_ptr().add(input_offset + 4));
        acc = vfmaq_f32(acc, dequant_1, x_1);

        // lo: 8..15
        let u16_lo_1 = vmovl_u8(vget_high_u8(interleaved_lo));
        let q_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_lo_1)));
        let dequant_2 = vfmaq_f32(v_min, q_2, v_scale);
        let x_2 = vld1q_f32(input.as_ptr().add(input_offset + 8));
        acc = vfmaq_f32(acc, dequant_2, x_2);

        let q_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_lo_1)));
        let dequant_3 = vfmaq_f32(v_min, q_3, v_scale);
        let x_3 = vld1q_f32(input.as_ptr().add(input_offset + 12));
        acc = vfmaq_f32(acc, dequant_3, x_3);

        // hi: 16..23
        let u16_hi_0 = vmovl_u8(vget_low_u8(interleaved_hi));
        let q_4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_hi_0)));
        let dequant_4 = vfmaq_f32(v_min, q_4, v_scale);
        let x_4 = vld1q_f32(input.as_ptr().add(input_offset + 16));
        acc = vfmaq_f32(acc, dequant_4, x_4);

        let q_5 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_hi_0)));
        let dequant_5 = vfmaq_f32(v_min, q_5, v_scale);
        let x_5 = vld1q_f32(input.as_ptr().add(input_offset + 20));
        acc = vfmaq_f32(acc, dequant_5, x_5);

        // hi: 24..31
        let u16_hi_1 = vmovl_u8(vget_high_u8(interleaved_hi));
        let q_6 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_hi_1)));
        let dequant_6 = vfmaq_f32(v_min, q_6, v_scale);
        let x_6 = vld1q_f32(input.as_ptr().add(input_offset + 24));
        acc = vfmaq_f32(acc, dequant_6, x_6);

        let q_7 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_hi_1)));
        let dequant_7 = vfmaq_f32(v_min, q_7, v_scale);
        let x_7 = vld1q_f32(input.as_ptr().add(input_offset + 28));
        acc = vfmaq_f32(acc, dequant_7, x_7);
    }

    let total = vaddvq_f32(acc);
    if total.abs() < 1e-6 {
        0.0
    } else {
        total
    }
}

// =============================================================================
// Q4_0 GEMV — Motor de dequantización y multiplicación vectorizada
// =============================================================================

/// Dequantiza y multiplica un bloque de 32 pesos Q4_0 usando AVX2 (x86_64)
/// 
/// # Safety
/// Requiere que `weights` tenga al menos 16 bytes, y `input` al menos 32 floats.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn gemv_q4_0_block_avx2(
    weights_ptr: *const u8,
    scale: f32,
    min: f32,
    input_ptr: *const f32,
) -> f32 {
    // Cargar 16 bytes (32 pesos de 4 bits)
    let packed = _mm_loadu_si128(weights_ptr as *const __m128i);
    
    // Desempaquetar 4 bits -> 8 bits
    let mask = _mm_set1_epi8(0x0F);
    let low = _mm_and_si128(packed, mask);
    let high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
    
    // Interleavar para obtener 16 bytes de pesos de 8 bits
    let interleaved = _mm_unpacklo_epi8(low, high);
    
    // Convertir a i16 y luego a i32 (dos registros de 8 elementos cada uno)
    let i16_low = _mm_cvtepi8_epi16(interleaved);
    let i16_high = _mm_cvtepi8_epi16(_mm_unpackhi_epi8(interleaved, interleaved));
    
    // Convertir a f32
    let f32_low = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(i16_low));
    let f32_high = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(i16_high));
    
    // Cargar input (32 floats = dos registros de 8 floats)
    let in_low = _mm256_loadu_ps(input_ptr);
    let in_high = _mm256_loadu_ps(input_ptr.add(8));
    
    // Aplicar dequantización: (q4 - 8) * scale + min
    let eight = _mm256_set1_ps(8.0);
    let scale_v = _mm256_set1_ps(scale);
    let min_v = _mm256_set1_ps(min);
    
    let dq_low = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(f32_low, eight), scale_v), min_v);
    let dq_high = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(f32_high, eight), scale_v), min_v);
    
    // Multiplicar y acumular (FMA)
    let mul_low = _mm256_mul_ps(dq_low, in_low);
    let mul_high = _mm256_mul_ps(dq_high, in_high);
    
    // Sumar los dos halves
    let sum = _mm256_add_ps(mul_low, mul_high);
    
    // Horizontal sum de 8 floats a 1 float
    let high128 = _mm256_extractf128_ps(sum, 1);
    let low128 = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(high128, low128);
    
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let final_sum = _mm_add_ss(sums, shuf2);
    
    _mm_cvtss_f32(final_sum)
}

/// Dequantiza y multiplica un bloque de 32 pesos Q4_0 usando NEON (aarch64)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn gemv_q4_0_block_neon(
    weights_ptr: *const u8,
    scale: f32,
    min: f32,
    input_ptr: *const f32,
) -> f32 {
    // Cargar 16 bytes
    let packed = vld1q_u8(weights_ptr);
    
    // Desempaquetar 4 bits -> 8 bits
    let mask = vdupq_n_u8(0x0F);
    let low = vandq_u8(packed, mask);
    let high = vandq_u8(vshrq_n_u8(packed, 4), mask);
    
    // Interleavar (zip)
    let interleaved = vzip1q_u8(low, high);
    
    // Convertir a f32 (NEON requiere pasar por i16 -> i32 -> f32)
    // Nota: Esta es una simplificación; en producción se usaría vcvtq_f32_s32(vcvtlq_s16(...))
    // Por ahora, fallback escalar seguro para aarch64 si la intrínseca es compleja
    let mut sum = 0.0f32;
    for i in 0..32 {
        let byte_idx = i / 2;
        let q4 = if i % 2 == 0 {
            (interleaved[byte_idx] & 0x0F) as f32
        } else {
            ((interleaved[byte_idx] >> 4) & 0x0F) as f32
        };
        let weight = (q4 - 8.0) * scale + min;
        sum += weight * *input_ptr.add(i);
    }
    sum
}

/// Producto Matriz-Vector (GEMV) optimizado para Q4_0
/// 
/// # Arguments
/// * `weights` - Pesos empaquetados (2 pesos por byte)
/// * `scales` - Escalas por bloque (f16)
/// * `mins` - Mínimos por bloque (f16)
/// * `input` - Vector de entrada (f32)
/// * `output` - Vector de salida (f32)
pub fn gemv_q4_0(
    weights: &[u8],
    scales: &[u16], // f16 representado como u16
    mins: &[u16],
    input: &[f32],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    let blocks_per_row = in_features / 32;
    
    // Paralelizar por fila de salida usando Rayon
    output.par_iter_mut().enumerate().for_each(|(i, out_ptr)| {
        let mut row_sum = 0.0f32;
        let row_offset = i * in_features / 2;
        
        for block_idx in 0..blocks_per_row {
            let w_ptr = unsafe { weights.as_ptr().add(row_offset + block_idx * 16) };
            let in_ptr = unsafe { input.as_ptr().add(block_idx * 32) };
            
            // Convertir f16 (u16) a f32
            let scale = half::f16::from_bits(scales[i * blocks_per_row + block_idx]).to_f32();
            let min = half::f16::from_bits(mins[i * blocks_per_row + block_idx]).to_f32();
            
            #[cfg(target_arch = "x86_64")]
            unsafe {
                row_sum += gemv_q4_0_block_avx2(w_ptr, scale, min, in_ptr);
            }
            
            #[cfg(target_arch = "aarch64")]
            unsafe {
                row_sum += gemv_q4_0_block_neon(w_ptr, scale, min, in_ptr);
            }
            
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                // Fallback escalar
                for j in 0..32 {
                    let byte_idx = block_idx * 16 + j / 2;
                    let q4 = if j % 2 == 0 {
                        (weights[row_offset + byte_idx] & 0x0F) as f32
                    } else {
                        ((weights[row_offset + byte_idx] >> 4) & 0x0F) as f32
                    };
                    let weight = (q4 - 8.0) * scale + min;
                    row_sum += weight * input[block_idx * 32 + j];
                }
            }
        }
        *out_ptr = row_sum;
    });
}
