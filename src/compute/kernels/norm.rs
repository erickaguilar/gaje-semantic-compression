// =============================================================================
// norm.rs — Normalización RMS vectorizada multiplataforma
// =============================================================================

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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