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

// 4. AVX2 Vectorized Nibble Unpack + Gather (vpshufb SIMD)
unsafe fn avx2_vpshufb_gather_dot_product_4bit(
    weights: &[u8],
    input: &[f32],
    centroids: &[f32],
    stride_4bit: usize,
    n_blocks: usize,
) -> f32 {
    let block_size = stride_4bit * 2;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let mask_low = _mm_set1_epi8(0x0F);

    for j in 0..n_blocks {
        let input_block_ptr = input.as_ptr().add(j * block_size);
        let weights_block_ptr = weights.as_ptr().add(j * stride_4bit);
        let centroids_ptr = centroids.as_ptr().add(j * 16);

        let mut k = 0;
        while k + 16 <= stride_4bit {
            let v_bytes = _mm_loadu_si128(weights_block_ptr.add(k) as *const __m128i);

            let n_low = _mm_and_si128(v_bytes, mask_low);
            let n_high = _mm_and_si128(_mm_srli_epi16(v_bytes, 4), mask_low);

            let idxs_0_15 = _mm_unpacklo_epi8(n_high, n_low);
            let idxs_16_31 = _mm_unpackhi_epi8(n_high, n_low);

            let i_0_7 = _mm256_cvtepu8_epi32(idxs_0_15);
            let i_8_15 = _mm256_cvtepu8_epi32(_mm_srli_si128(idxs_0_15, 8));
            let i_16_23 = _mm256_cvtepu8_epi32(idxs_16_31);
            let i_24_31 = _mm256_cvtepu8_epi32(_mm_srli_si128(idxs_16_31, 8));

            let w0 = _mm256_i32gather_ps::<4>(centroids_ptr, i_0_7);
            let w1 = _mm256_i32gather_ps::<4>(centroids_ptr, i_8_15);
            let w2 = _mm256_i32gather_ps::<4>(centroids_ptr, i_16_23);
            let w3 = _mm256_i32gather_ps::<4>(centroids_ptr, i_24_31);

            let in0 = _mm256_loadu_ps(input_block_ptr.add(k * 2));
            let in1 = _mm256_loadu_ps(input_block_ptr.add(k * 2 + 8));
            let in2 = _mm256_loadu_ps(input_block_ptr.add(k * 2 + 16));
            let in3 = _mm256_loadu_ps(input_block_ptr.add(k * 2 + 24));

            acc0 = _mm256_fmadd_ps(w0, in0, acc0);
            acc1 = _mm256_fmadd_ps(w1, in1, acc1);
            acc2 = _mm256_fmadd_ps(w2, in2, acc2);
            acc3 = _mm256_fmadd_ps(w3, in3, acc3);

            k += 16;
        }

        let c_lut: &[f32; 16] = &*(centroids_ptr as *const [f32; 16]);
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

    let acc_a = _mm256_add_ps(acc0, acc1);
    let acc_b = _mm256_add_ps(acc2, acc3);
    let acc = _mm256_add_ps(acc_a, acc_b);

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

// 5. AVX2 vpermps (1-cycle register permute) 4-bit Kernel
unsafe fn avx2_vpermps_dot_product_4bit(
    weights: &[u8],
    input: &[f32],
    centroids: &[f32],
    stride_4bit: usize,
    n_blocks: usize,
) -> f32 {
    let block_size = stride_4bit * 2;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let mask_low = _mm_set1_epi8(0x0F);
    let c7 = _mm256_set1_epi32(7);
    let c8 = _mm256_set1_epi32(8);

    for j in 0..n_blocks {
        let input_block_ptr = input.as_ptr().add(j * block_size);
        let weights_block_ptr = weights.as_ptr().add(j * stride_4bit);
        let centroids_ptr = centroids.as_ptr().add(j * 16);

        // Cargar los 16 centroides del bloque en 2 registros YMM (8 floats cada uno)
        let c_low = _mm256_loadu_ps(centroids_ptr);
        let c_high = _mm256_loadu_ps(centroids_ptr.add(8));

        let mut k = 0;
        while k + 16 <= stride_4bit {
            let v_bytes = _mm_loadu_si128(weights_block_ptr.add(k) as *const __m128i);

            let n_low = _mm_and_si128(v_bytes, mask_low);
            let n_high = _mm_and_si128(_mm_srli_epi16(v_bytes, 4), mask_low);

            let idxs_0_15 = _mm_unpacklo_epi8(n_high, n_low);
            let idxs_16_31 = _mm_unpackhi_epi8(n_high, n_low);

            let i0 = _mm256_cvtepu8_epi32(idxs_0_15);
            let i1 = _mm256_cvtepu8_epi32(_mm_srli_si128(idxs_0_15, 8));
            let i2 = _mm256_cvtepu8_epi32(idxs_16_31);
            let i3 = _mm256_cvtepu8_epi32(_mm_srli_si128(idxs_16_31, 8));

            // Permutar c_low y c_high en 1 ciclo usando vpermps
            let m0 = _mm256_cmpgt_epi32(i0, c7);
            let w0_l = _mm256_permutevar8x32_ps(c_low, i0);
            let w0_h = _mm256_permutevar8x32_ps(c_high, _mm256_sub_epi32(i0, c8));
            let w0 = _mm256_blendv_ps(w0_l, w0_h, _mm256_castsi256_ps(m0));

            let m1 = _mm256_cmpgt_epi32(i1, c7);
            let w1_l = _mm256_permutevar8x32_ps(c_low, i1);
            let w1_h = _mm256_permutevar8x32_ps(c_high, _mm256_sub_epi32(i1, c8));
            let w1 = _mm256_blendv_ps(w1_l, w1_h, _mm256_castsi256_ps(m1));

            let m2 = _mm256_cmpgt_epi32(i2, c7);
            let w2_l = _mm256_permutevar8x32_ps(c_low, i2);
            let w2_h = _mm256_permutevar8x32_ps(c_high, _mm256_sub_epi32(i2, c8));
            let w2 = _mm256_blendv_ps(w2_l, w2_h, _mm256_castsi256_ps(m2));

            let m3 = _mm256_cmpgt_epi32(i3, c7);
            let w3_l = _mm256_permutevar8x32_ps(c_low, i3);
            let w3_h = _mm256_permutevar8x32_ps(c_high, _mm256_sub_epi32(i3, c8));
            let w3 = _mm256_blendv_ps(w3_l, w3_h, _mm256_castsi256_ps(m3));

            let in0 = _mm256_loadu_ps(input_block_ptr.add(k * 2));
            let in1 = _mm256_loadu_ps(input_block_ptr.add(k * 2 + 8));
            let in2 = _mm256_loadu_ps(input_block_ptr.add(k * 2 + 16));
            let in3 = _mm256_loadu_ps(input_block_ptr.add(k * 2 + 24));

            acc0 = _mm256_fmadd_ps(w0, in0, acc0);
            acc1 = _mm256_fmadd_ps(w1, in1, acc1);
            acc2 = _mm256_fmadd_ps(w2, in2, acc2);
            acc3 = _mm256_fmadd_ps(w3, in3, acc3);

            k += 16;
        }

        let c_lut: &[f32; 16] = &*(centroids_ptr as *const [f32; 16]);
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

    let acc_a = _mm256_add_ps(acc0, acc1);
    let acc_b = _mm256_add_ps(acc2, acc3);
    let acc = _mm256_add_ps(acc_a, acc_b);

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

// 6. Unrolled 8-Way Scalar LUT (Maximum Out-of-Order Execution)
unsafe fn unrolled_8way_dot_product_4bit(
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
    let mut sum4 = 0.0f32;
    let mut sum5 = 0.0f32;
    let mut sum6 = 0.0f32;
    let mut sum7 = 0.0f32;

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

            let in_ptr = input_block_ptr.add(k * 2);

            sum0 += c_lut[(b0 >> 4) as usize] * *in_ptr;
            sum1 += c_lut[(b0 & 0x0F) as usize] * *in_ptr.add(1);
            sum2 += c_lut[(b1 >> 4) as usize] * *in_ptr.add(2);
            sum3 += c_lut[(b1 & 0x0F) as usize] * *in_ptr.add(3);

            sum4 += c_lut[(b2 >> 4) as usize] * *in_ptr.add(4);
            sum5 += c_lut[(b2 & 0x0F) as usize] * *in_ptr.add(5);
            sum6 += c_lut[(b3 >> 4) as usize] * *in_ptr.add(6);
            sum7 += c_lut[(b3 & 0x0F) as usize] * *in_ptr.add(7);

            sum0 += c_lut[(b4 >> 4) as usize] * *in_ptr.add(8);
            sum1 += c_lut[(b4 & 0x0F) as usize] * *in_ptr.add(9);
            sum2 += c_lut[(b5 >> 4) as usize] * *in_ptr.add(10);
            sum3 += c_lut[(b5 & 0x0F) as usize] * *in_ptr.add(11);

            sum4 += c_lut[(b6 >> 4) as usize] * *in_ptr.add(12);
            sum5 += c_lut[(b6 & 0x0F) as usize] * *in_ptr.add(13);
            sum6 += c_lut[(b7 >> 4) as usize] * *in_ptr.add(14);
            sum7 += c_lut[(b7 & 0x0F) as usize] * *in_ptr.add(15);

            k += 8;
        }

        while k < stride_4bit {
            let byte = *weights_block_ptr.add(k);
            sum0 += c_lut[(byte >> 4) as usize] * *input_block_ptr.add(k * 2);
            sum1 += c_lut[(byte & 0x0F) as usize] * *input_block_ptr.add(k * 2 + 1);
            k += 1;
        }
    }

    let total = (sum0 + sum1 + sum2 + sum3) + (sum4 + sum5 + sum6 + sum7);
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

    // 4. AVX2 Vectorized Nibble Unpack + Gather (vpshufb)
    let t0 = Instant::now();
    for _ in 0..iterations {
        for r in 0..out_dim {
            let w = &weights[r * (in_dim / 2)..(r + 1) * (in_dim / 2)];
            let c = &centroids[r * n_blocks * 16..(r + 1) * n_blocks * 16];
            let v = unsafe {
                avx2_vpshufb_gather_dot_product_4bit(w, &input, c, stride_4bit, n_blocks)
            };
            dummy_sink += black_box(v);
        }
    }
    let dur_gather = t0.elapsed().as_secs_f64() * 1000.0;

    // 5. AVX2 vpermps (1-cycle register permute)
    let t0 = Instant::now();
    for _ in 0..iterations {
        for r in 0..out_dim {
            let w = &weights[r * (in_dim / 2)..(r + 1) * (in_dim / 2)];
            let c = &centroids[r * n_blocks * 16..(r + 1) * n_blocks * 16];
            let v = unsafe { avx2_vpermps_dot_product_4bit(w, &input, c, stride_4bit, n_blocks) };
            dummy_sink += black_box(v);
        }
    }
    let dur_permps = t0.elapsed().as_secs_f64() * 1000.0;

    // 6. Unrolled 8-Way Scalar LUT
    let t0 = Instant::now();
    for _ in 0..iterations {
        for r in 0..out_dim {
            let w = &weights[r * (in_dim / 2)..(r + 1) * (in_dim / 2)];
            let c = &centroids[r * n_blocks * 16..(r + 1) * n_blocks * 16];
            let v = unsafe { unrolled_8way_dot_product_4bit(w, &input, c, stride_4bit, n_blocks) };
            dummy_sink += black_box(v);
        }
    }
    let dur_8way = t0.elapsed().as_secs_f64() * 1000.0;

    black_box(dummy_sink);

    let us_scalar = dur_scalar * 1000.0 / iterations as f64;
    let us_avx2_stack = dur_avx2_stack * 1000.0 / iterations as f64;
    let us_unrolled = dur_unrolled * 1000.0 / iterations as f64;
    let us_gather = dur_gather * 1000.0 / iterations as f64;
    let us_permps = dur_permps * 1000.0 / iterations as f64;
    let us_8way = dur_8way * 1000.0 / iterations as f64;

    println!("📊 RESULTADOS PRECISOS (µs por FFN GEMV):");
    println!(
        "  1. Escalar Simple Base      : {:.2} ms ({:.2} µs/gemv) [Baseline 1.00x]",
        dur_scalar, us_scalar
    );
    println!(
        "  2. AVX2 Stack Spill (Viejo)  : {:.2} ms ({:.2} µs/gemv) [{:.2}x]",
        dur_avx2_stack,
        us_avx2_stack,
        dur_scalar / dur_avx2_stack
    );
    println!(
        "  3. Unrolled 4-Way           : {:.2} ms ({:.2} µs/gemv) [{:.2}x]",
        dur_unrolled,
        us_unrolled,
        dur_scalar / dur_unrolled
    );
    println!(
        "  4. VPSHUFB Gather (Slow)    : {:.2} ms ({:.2} µs/gemv) [{:.2}x]",
        dur_gather,
        us_gather,
        dur_scalar / dur_gather
    );
    println!(
        "  5. VPERMPS (1-Cycle SIMD)   : {:.2} ms ({:.2} µs/gemv) [{:.2}x]",
        dur_permps,
        us_permps,
        dur_scalar / dur_permps
    );
    println!(
        "  6. Unrolled 8-Way (Optimizado): {:.2} ms ({:.2} µs/gemv) [{:.2}x]",
        dur_8way,
        us_8way,
        dur_scalar / dur_8way
    );
    println!("================================================================");
    println!(
        "🚀 Speedup 8-Way vs Unrolled 4-Way: {:.2}x | vs Scalar: {:.2}x",
        dur_unrolled / dur_8way,
        dur_scalar / dur_8way
    );
    println!("================================================================");
}
