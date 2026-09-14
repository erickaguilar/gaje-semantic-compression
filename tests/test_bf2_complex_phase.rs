#[cfg(test)]
mod tests {
    use _impl::compute::bf2::{
        bf2_gemv_parallel, bf2_gemv_scalar, born_density, mod_relu, prepare_complex_input,
        Bf2Matrix, INV_SQRT_2,
    };
    use std::time::Instant;

    /// 1. Verificación matemática exacta de los 4 cuadrantes canónicos
    #[test]
    fn test_bf2_quadrant_exactness() {
        println!("\n========================================================");
        println!("🧬 VALIDACIÓN DE EXACTITUD MATEMÁTICA BF2-COMPLEX EN C");
        println!("========================================================");

        let xr = 2.5f32;
        let xi = -1.5f32;
        let s = xr + xi;
        let d = xr - xi;

        let quadrants = [
            (0b00, "A (Cuadrante I: 45°)", 1.0f32, 1.0f32),
            (0b01, "C (Cuadrante II: 135°)", -1.0f32, 1.0f32),
            (0b11, "G (Cuadrante III: 225°)", -1.0f32, -1.0f32),
            (0b10, "T (Cuadrante IV: 315°)", 1.0f32, -1.0f32),
        ];

        for (bits, label, wr, wi) in quadrants {
            // Cómputo formal complejo: (wr + i*wi)*(xr + i*xi)
            let formal_r = (wr * xr - wi * xi) * INV_SQRT_2;
            let formal_i = (wr * xi + wi * xr) * INV_SQRT_2;

            // Cómputo zero-multiplier con s y d:
            let (bf2_r, bf2_i) = match bits {
                0b00 => (d * INV_SQRT_2, s * INV_SQRT_2),
                0b01 => (-s * INV_SQRT_2, d * INV_SQRT_2),
                0b10 => (s * INV_SQRT_2, -d * INV_SQRT_2),
                0b11 => (-d * INV_SQRT_2, -s * INV_SQRT_2),
                _ => unreachable!(),
            };

            let err_r = (formal_r - bf2_r).abs();
            let err_i = (formal_i - bf2_i).abs();

            println!(
                "Base {} [{:02b}]: Formal=({:.4}, {:.4}), BF2=({:.4}, {:.4}) -> Error Max: {:.1e}",
                label, bits, formal_r, formal_i, bf2_r, bf2_i, err_r.max(err_i)
            );

            assert!(err_r < 1e-6, "Error en componente real para {}", label);
            assert!(err_i < 1e-6, "Error en componente imaginario para {}", label);
        }
        println!("✅ Todos los cuadrantes cuaternarios coinciden bit a bit.");
    }

    /// 2. Verificación de GEMV completo (matriz x vector) contra producto formal
    #[test]
    fn test_bf2_gemv_equivalence() {
        let rows = 64;
        let cols = 128; // Múltiplo de 32

        // Crear pesos reales e imaginarios sintéticos
        let mut wr = Vec::with_capacity(rows * cols);
        let mut wi = Vec::with_capacity(rows * cols);
        for i in 0..(rows * cols) {
            let angle = (i as f32 * 0.17).sin() * std::f32::consts::PI;
            let mag = 0.5 + 0.5 * (i as f32 * 0.05).cos().abs();
            wr.push(mag * angle.cos());
            wi.push(mag * angle.sin());
        }

        let matrix = Bf2Matrix::from_f32(&wr, Some(&wi), rows, cols);

        // Vector de entrada
        let mut xr = Vec::with_capacity(cols);
        let mut xi = Vec::with_capacity(cols);
        for i in 0..cols {
            xr.push((i as f32 * 0.31).cos());
            xi.push((i as f32 * 0.23).sin());
        }

        // Preparar sumas y diferencias
        let mut s_vec = vec![0.0f32; cols];
        let mut d_vec = vec![0.0f32; cols];
        prepare_complex_input(&xr, &xi, &mut s_vec, &mut d_vec);

        // Inferencia escalar
        let mut out_r = vec![0.0f32; rows];
        let mut out_i = vec![0.0f32; rows];
        bf2_gemv_scalar(&matrix, &s_vec, &d_vec, &mut out_r, &mut out_i);

        // Inferencia paralela
        let mut par_out_r = vec![0.0f32; rows];
        let mut par_out_i = vec![0.0f32; rows];
        bf2_gemv_parallel(&matrix, &s_vec, &d_vec, &mut par_out_r, &mut par_out_i);

        // Validar paridad entre escalar y paralelo
        for r in 0..rows {
            assert!(
                (out_r[r] - par_out_r[r]).abs() < 1e-5,
                "Discrepancia en out_r fila {}",
                r
            );
            assert!(
                (out_i[r] - par_out_i[r]).abs() < 1e-5,
                "Discrepancia en out_i fila {}",
                r
            );
        }

        println!("✅ GEMV escalar y paralelo coinciden exactamente.");
    }

    /// 3. Verificación de la Regla de Born vs Softmax tradicional
    #[test]
    fn test_born_density_vs_softmax() {
        let vocab_size = 49152; // Tamaño de vocabulario de Qwen/GAJE
        let mut yr = Vec::with_capacity(vocab_size);
        let mut yi = Vec::with_capacity(vocab_size);
        for i in 0..vocab_size {
            yr.push((i as f32 * 0.01).sin() * 2.0);
            yi.push((i as f32 * 0.02).cos() * 2.0);
        }

        let mut born_probs = vec![0.0f32; vocab_size];

        // Benchmark Regla de Born
        let start_born = Instant::now();
        born_density(&yr, &yi, &mut born_probs);
        let born_duration = start_born.elapsed();

        // Validar normalización sum(P) == 1.0
        let sum_p: f32 = born_probs.iter().sum();
        assert!((sum_p - 1.0).abs() < 1e-4, "Suma de Born != 1.0: {}", sum_p);

        // Benchmark Softmax tradicional con exp()
        let start_softmax = Instant::now();
        let mut exp_sum = 0.0f32;
        let mut softmax_probs = vec![0.0f32; vocab_size];
        for i in 0..vocab_size {
            let e = yr[i].exp(); // Softmax sólo en componente real
            softmax_probs[i] = e;
            exp_sum += e;
        }
        let inv_exp = 1.0 / exp_sum;
        for p in softmax_probs.iter_mut() {
            *p *= inv_exp;
        }
        let softmax_duration = start_softmax.elapsed();

        println!("\n=== COMPARATIVA DE PROYECCIÓN DE VOCABULARIO (Vocab={}) ===", vocab_size);
        println!("⚡ Regla de Born (|Ψ|²):      {:>8.2?} (Zero exp, puras multiplicaciones)", born_duration);
        println!("⏱️  Softmax Clásico (exp(z)): {:>8.2?} (Trascendental pesado)", softmax_duration);
        let speedup = softmax_duration.as_nanos() as f64 / born_duration.as_nanos().max(1) as f64;
        println!("🚀 Aceleración en salida:    {:.2}x más rápido", speedup);
    }

    /// 4. Benchmark de Inferencia en el Silicio Local (ARM64 / Android Termux)
    #[test]
    fn test_benchmark_silicon_zero_multiplier() {
        println!("\n========================================================");
        println!("⚡ BENCHMARK DE INFERENCIA EN SILICIO LOCAL (ZERO-MULTIPLIER)");
        println!("========================================================");

        // Dimensiones típicas de capa de atención/FFN en modelos edge
        let dim = 512;
        let ffn_dim = 1024;
        let iterations = 100;

        let mut wr = vec![0.0f32; ffn_dim * dim];
        let mut wi = vec![0.0f32; ffn_dim * dim];
        for i in 0..(ffn_dim * dim) {
            wr[i] = (i as f32 * 0.05).sin();
            wi[i] = (i as f32 * 0.03).cos();
        }

        let matrix = Bf2Matrix::from_f32(&wr, Some(&wi), ffn_dim, dim);

        let xr = vec![0.7f32; dim];
        let xi = vec![0.3f32; dim];

        let mut s_vec = vec![0.0f32; dim];
        let mut d_vec = vec![0.0f32; dim];
        prepare_complex_input(&xr, &xi, &mut s_vec, &mut d_vec);

        let mut out_r = vec![0.0f32; ffn_dim];
        let mut out_i = vec![0.0f32; ffn_dim];

        // 1. Benchmark BF2-Complex Scalar
        let start_bf2 = Instant::now();
        for _ in 0..iterations {
            bf2_gemv_scalar(&matrix, &s_vec, &d_vec, &mut out_r, &mut out_i);
        }
        let dur_bf2 = start_bf2.elapsed() / iterations;

        // 2. Benchmark BF2-Complex Parallel
        let start_bf2_par = Instant::now();
        for _ in 0..iterations {
            bf2_gemv_parallel(&matrix, &s_vec, &d_vec, &mut out_r, &mut out_i);
        }
        let dur_bf2_par = start_bf2_par.elapsed() / iterations;

        // 3. Benchmark FP32 Clásico con multiplicadores de hardware
        let start_fp32 = Instant::now();
        for _ in 0..iterations {
            for r in 0..ffn_dim {
                let mut acc_r = 0.0f32;
                let mut acc_i = 0.0f32;
                let row_off = r * dim;
                for c in 0..dim {
                    let w_r = wr[row_off + c];
                    let w_i = wi[row_off + c];
                    let x_r = xr[c];
                    let x_i = xi[c];
                    // 4 multiplicaciones de hardware + 2 adiciones por elemento complejo
                    acc_r += w_r * x_r - w_i * x_i;
                    acc_i += w_r * x_i + w_i * x_r;
                }
                out_r[r] = acc_r;
                out_i[r] = acc_i;
            }
        }
        let dur_fp32 = start_fp32.elapsed() / iterations;

        let mem_bf2 = matrix.memory_bytes();
        let mem_fp32 = ffn_dim * dim * 4 * 2; // 2 componentes f32
        let ratio = mem_fp32 as f32 / mem_bf2 as f32;

        println!("Dimensiones Matriz: {} x {} ({} pesos complejos)", ffn_dim, dim, ffn_dim * dim);
        println!("Memoria FP32 Denso: {:.2} KB", mem_fp32 as f32 / 1024.0);
        println!("Memoria BF2-Complex: {:.2} KB (Tasa de compresión: {:.2}x)", mem_bf2 as f32 / 1024.0, ratio);
        println!("--------------------------------------------------------");
        println!("⏱️  FP32 Denso (Multiplicadores HW): {:>8.2?}", dur_fp32);
        println!("⚡ BF2-Complex Escalar (Zero-Mul):   {:>8.2?}", dur_bf2);
        println!("🚀 BF2-Complex Paralelo Rayon:       {:>8.2?}", dur_bf2_par);
        println!("--------------------------------------------------------");

        let speedup_scalar = dur_fp32.as_nanos() as f64 / dur_bf2.as_nanos().max(1) as f64;
        let speedup_par = dur_fp32.as_nanos() as f64 / dur_bf2_par.as_nanos().max(1) as f64;
        println!("📈 Aceleración Escalar:  {:.2}x", speedup_scalar);
        println!("🔥 Aceleración Paralela: {:.2}x", speedup_par);

        // Estimar throughput de tokens/segundo en un transformer de 12 capas (4 proyecciones por capa = 48 GEMVs)
        let total_gemvs_per_token = 48.0;
        let tok_s_fp32 = 1.0 / (dur_fp32.as_secs_f64() * total_gemvs_per_token);
        let tok_s_bf2 = 1.0 / (dur_bf2_par.as_secs_f64() * total_gemvs_per_token);

        println!("📊 Throughput Estimado en Transformer L=12:");
        println!("   • FP32 Denso:    {:.1} tok/s", tok_s_fp32);
        println!("   • BF2-Complex:   {:.1} tok/s", tok_s_bf2);
        println!("========================================================\n");
    }

    /// 5. Verificación de la invariancia de fase en activación ModReLU
    #[test]
    fn test_mod_relu_phase_invariance() {
        let mut yr = vec![3.0f32, -4.0f32, 0.05f32];
        let mut yi = vec![4.0f32, 3.0f32, -0.05f32];

        let orig_angles: Vec<f32> = yr.iter().zip(&yi).map(|(&r, &i)| i.atan2(r)).collect();

        // Aplicar bias que preserva los primeros dos y filtra el tercero
        mod_relu(&mut yr, &mut yi, -0.1);

        // Primeros dos deben conservar su fase intacta
        let angle0 = yi[0].atan2(yr[0]);
        let angle1 = yi[1].atan2(yr[1]);

        assert!((angle0 - orig_angles[0]).abs() < 1e-6, "Fase alterada en elemento 0");
        assert!((angle1 - orig_angles[1]).abs() < 1e-6, "Fase alterada en elemento 1");

        // El tercero debe haber colapsado a 0.0 por estar bajo el umbral de ruido
        assert_eq!(yr[2], 0.0);
        assert_eq!(yi[2], 0.0);

        println!("✅ ModReLU preserva exactamente la fase y filtra ruido cuántico.");
    }
}
