// =============================================================================
// dot.rs — Productos punto SIMD multiplataforma
//
// aarch64  → ARM NEON (Android/Termux)
// x86_64   → AVX2 + FMA (Windows/Linux PC)
// fallback → escalar puro (cualquier otro target)
// =============================================================================

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::compute::kv_cache::CompressedKVCache;

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
