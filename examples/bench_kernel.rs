use std::arch::x86_64::*;
use std::hint::black_box;
use std::time::Instant;

// 1. Escalar Puro de referencia
unsafe fn scalar_dot_product_4bit(
    weights: &[u8],
    input: &[f32],
    centroids: &[f32],
    stride_4bit: usize,
    n_blocks: usize,
) -> f32 {
    let mut scalar_sum = 0.0f32;
    let block_size = stride_4bit * 2;
    for j in 0..n_blocks {
        let input_block_ptr = input.as_ptr().add(j * block_size);
        let weights_block_ptr = weights.as_ptr().add(j * stride_4bit);
        let centroids_ptr = centroids.as_ptr().add(j * 16);
        let c_lut: &[f32; 16] = &*(centroids_ptr as *const [f32; 16]);

        for k in 0..stride_4bit {
            let byte = *weights_block_ptr.add(k);
            let c_idx1 = (byte >> 4) as usize;
            scalar_sum += c_lut[c_idx1] * *input_block_ptr.add(k * 2);

            let c_idx2 = (byte & 0x0F) as usize;
            scalar_sum += c_lut[c_idx2] * *input_block_ptr.add(k * 2 + 1);
        }
    }
    if scalar_sum.abs() < 1e-6 {
        0.0
    } else {
        scalar_sum
    }
}

// 2. AVX2 Stack Spill (Versión previa)
unsafe fn avx2_stack_spill_dot_product_4bit(
    weights: &[u8],
    input: &[f32],
    centroids: &[f32],
    stride_4bit: usize,
    n_blocks: usize,
) -> f32 {
    let block_size = stride_4bit * 2;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    for j in 0..n_blocks {
        let input_block_ptr = input.as_ptr().add(j * block_size);
        let weights_block_ptr = weights.as_ptr().add(j * stride_4bit);
        let centroids_ptr = centroids.as_ptr().add(j * 16);
        let c_lut: &[f32; 16] = &*(centroids_ptr as *const [f32; 16]);

        let mut k = 0;
        while k + 8 <= stride_4bit {
            let b0 = *weights_block_ptr.add(k);
            let b1 = *weights_block_ptr.add(k + 1);
            let b2 = *weights_block_ptr.add(k + 2);
            let b3 = *weights_block_ptr.add(k + 3);
            let b4 = *weights_block_ptr.add(k + 4);
            let b5 = *weights_block_ptr.add(k + 5);
            let b6 = *weights_block_ptr.add(k + 6);
            let b7 = *weights_block_ptr.add(k + 7);

            let w_vals0 = [
                c_lut[(b0 >> 4) as usize],
                c_lut[(b0 & 0x0F) as usize],
                c_lut[(b1 >> 4) as usize],
                c_lut[(b1 & 0x0F) as usize],
                c_lut[(b2 >> 4) as usize],
                c_lut[(b2 & 0x0F) as usize],
                c_lut[(b3 >> 4) as usize],
                c_lut[(b3 & 0x0F) as usize],
            ];
            let w_vals1 = [
                c_lut[(b4 >> 4) as usize],
                c_lut[(b4 & 0x0F) as usize],
                c_lut[(b5 >> 4) as usize],
                c_lut[(b5 & 0x0F) as usize],
                c_lut[(b6 >> 4) as usize],
                c_lut[(b6 & 0x0F) as usize],
                c_lut[(b7 >> 4) as usize],
                c_lut[(b7 & 0x0F) as usize],
            ];

            let vw0 = _mm256_loadu_ps(w_vals0.as_ptr());
            let vw1 = _mm256_loadu_ps(w_vals1.as_ptr());
            let vi0 = _mm256_loadu_ps(input_block_ptr.add(k * 2));
            let vi1 = _mm256_loadu_ps(input_block_ptr.add(k * 2 + 8));

            acc0 = _mm256_fmadd_ps(vw0, vi0, acc0);
            acc1 = _mm256_fmadd_ps(vw1, vi1, acc1);

            k += 8;
        }

        while k < stride_4bit {
            let byte = *weights_block_ptr.add(k);
            let c1 = c_lut[(byte >> 4) as usize];
            let c2 = c_lut[(byte & 0x0F) as usize];
            let in1 = *input_block_ptr.add(k * 2);
            let in2 = *input_block_ptr.add(k * 2 + 1);
            let v_tail = _mm256_setr_ps(c1 * in1 + c2 * in2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            acc0 = _mm256_add_ps(acc0, v_tail);
            k += 1;
        }
    }

    let acc = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let res_f32 = _mm_cvtss_f32(result);
    if res_f32.abs() < 1e-6 {
        0.0
    } else {
        res_f32
    }
}

// 3. Unrolled 4-Way FMA Accumulator (Kernel Optimizado sin Stack Spill)
unsafe fn unrolled_fma_dot_product_4bit(
    weights: &[u8],
    input: &[f32],
    centroids: &[f32],
    stride_4bit: usize,
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

fn main() {
    println!("================================================================");
    println!("🧪 ISOLATED BENCHMARK 4-BIT (Preventing Dead Code Elimination)");
    println!("================================================================");

    let in_dim = 896;
    let out_dim = 4864;
    let block_size = 64;
    let stride_4bit = block_size / 2;
    let n_blocks = in_dim / block_size;

    let mut input = vec![0.5f32; in_dim];
    for i in 0..in_dim {
        input[i] = (i as f32 * 0.01).sin();
    }

    let n_weights_bytes = out_dim * (in_dim / 2);
    let mut weights = vec![0u8; n_weights_bytes];
    for i in 0..n_weights_bytes {
        weights[i] = ((i * 17 + 3) % 256) as u8;
    }

    let n_centroids = out_dim * n_blocks * 16;
    let mut centroids = vec![0.0f32; n_centroids];
    for i in 0..n_centroids {
        centroids[i] = (i as f32 * 0.05).cos();
    }

    let iterations = 1000;
    let mut dummy_sink = 0.0f32;

    // 1. Escalar Simple
    let t0 = Instant::now();
    for _ in 0..iterations {
        for r in 0..out_dim {
            let w = &weights[r * (in_dim / 2)..(r + 1) * (in_dim / 2)];
            let c = &centroids[r * n_blocks * 16..(r + 1) * n_blocks * 16];
            let v = unsafe { scalar_dot_product_4bit(w, &input, c, stride_4bit, n_blocks) };
            dummy_sink += black_box(v);
        }
    }
    let dur_scalar = t0.elapsed().as_secs_f64() * 1000.0;

    // 2. AVX2 Stack Spill (Código previo)
    let t0 = Instant::now();
    for _ in 0..iterations {
        for r in 0..out_dim {
            let w = &weights[r * (in_dim / 2)..(r + 1) * (in_dim / 2)];
            let c = &centroids[r * n_blocks * 16..(r + 1) * n_blocks * 16];
            let v =
                unsafe { avx2_stack_spill_dot_product_4bit(w, &input, c, stride_4bit, n_blocks) };
            dummy_sink += black_box(v);
        }
    }
    let dur_avx2_stack = t0.elapsed().as_secs_f64() * 1000.0;

    // 3. Unrolled 4-Way FMA Accumulators (Nueva solución)
    let t0 = Instant::now();
    for _ in 0..iterations {
        for r in 0..out_dim {
            let w = &weights[r * (in_dim / 2)..(r + 1) * (in_dim / 2)];
            let c = &centroids[r * n_blocks * 16..(r + 1) * n_blocks * 16];
            let v = unsafe { unrolled_fma_dot_product_4bit(w, &input, c, stride_4bit, n_blocks) };
            dummy_sink += black_box(v);
        }
    }
    let dur_unrolled = t0.elapsed().as_secs_f64() * 1000.0;

    black_box(dummy_sink);

    let us_scalar = dur_scalar * 1000.0 / iterations as f64;
    let us_avx2_stack = dur_avx2_stack * 1000.0 / iterations as f64;
    let us_unrolled = dur_unrolled * 1000.0 / iterations as f64;

    println!("📊 RESULTADOS PRECISOS (µs por FFN GEMV):");
    println!(
        "  1. Escalar Simple Base    : {:.2} ms ({:.2} µs/gemv) [Baseline 1.00x]",
        dur_scalar, us_scalar
    );
    println!(
        "  2. AVX2 Stack Spill (Viejo): {:.2} ms ({:.2} µs/gemv) [{:.2}x]",
        dur_avx2_stack,
        us_avx2_stack,
        dur_scalar / dur_avx2_stack
    );
    println!(
        "  3. Unrolled 4-Way (NUEVO)  : {:.2} ms ({:.2} µs/gemv) [{:.2}x]",
        dur_unrolled,
        us_unrolled,
        dur_scalar / dur_unrolled
    );
    println!("================================================================");
    println!(
        "🚀 Aceleración del Nuevo Kernel vs AVX2 Viejo: {:.2}x más rápido",
        dur_avx2_stack / dur_unrolled
    );
    println!("================================================================");
}
