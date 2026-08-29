//! 🩺 Módulo de Diagnóstico de Hardware y Entorno (gaje-cli doctor)

use std::time::Instant;

#[derive(Debug, Clone)]
pub struct DoctorReport {
    pub cpu_cores: usize,
    pub simd_avx2: bool,
    pub simd_avx512f: bool,
    pub simd_fma: bool,
    pub simd_sse41: bool,
    pub simd_neon: bool,
    pub gpu_backend: String,
    pub mmap_speed_gb_s: f64,
    pub os_info: String,
    pub is_optimal: bool,
}

pub fn run_doctor() -> DoctorReport {
    let cpu_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    #[cfg(target_arch = "x86_64")]
    let (simd_avx2, simd_avx512f, simd_fma, simd_sse41, simd_neon) = (
        is_x86_feature_detected!("avx2"),
        is_x86_feature_detected!("avx512f"),
        is_x86_feature_detected!("fma"),
        is_x86_feature_detected!("sse4.1"),
        false,
    );

    #[cfg(target_arch = "aarch64")]
    let (simd_avx2, simd_avx512f, simd_fma, simd_sse41, simd_neon) = (
        false, false, false, false, true, // NEON siempre activo en aarch64
    );

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let (simd_avx2, simd_avx512f, simd_fma, simd_sse41, simd_neon) =
        (false, false, false, false, false);

    // Detección de GPU
    let gpu_backend = {
        #[cfg(feature = "gpu")]
        {
            if let Some(ref ctx) = *crate::compute::gpu::context::GLOBAL_GPU_CONTEXT {
                format!("{} ({})", ctx.info.device_name, ctx.info.backend)
            } else {
                "WGPU / Vulkan (Sin adaptador detectado)".to_string()
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            "Modo CPU Soberano (SIMD AVX2/NEON)".to_string()
        }
    };

    // Benchmark sintético de ancho de banda de memoria
    let mmap_speed_gb_s = {
        let size_bytes = 64 * 1024 * 1024; // 64 MB
        let buffer = vec![0x5Au8; size_bytes];
        let t0 = Instant::now();
        let mut sum: u64 = 0;
        for chunk in buffer.chunks_exact(8) {
            sum = sum.wrapping_add(u64::from_ne_bytes(chunk.try_into().unwrap()));
        }
        std::hint::black_box(sum);
        let elapsed = t0.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            (size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / elapsed
        } else {
            0.0
        }
    };

    let os_info = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    let is_optimal = (simd_avx2 && simd_fma) || simd_neon || simd_avx512f;

    DoctorReport {
        cpu_cores,
        simd_avx2,
        simd_avx512f,
        simd_fma,
        simd_sse41,
        simd_neon,
        gpu_backend,
        mmap_speed_gb_s,
        os_info,
        is_optimal,
    }
}

pub fn print_doctor_report(report: &DoctorReport) {
    println!("\n🧬 ========================================================");
    println!("🩺 GAJE HELIX — Diagnóstico de Entorno y Hardware (Doctor)");
    println!("========================================================\n");

    println!("💻 Sistema Operativo: {}", report.os_info);
    println!(
        "⚙️  Núcleos de CPU Disponibles: {} hilos lógicos\n",
        report.cpu_cores
    );

    println!("⚡ Conjunto de Instrucciones SIMD:");
    println!(
        "   • AVX2 (256-bit):       {}",
        if report.simd_avx2 {
            "🟢 Soportado (Óptimo)"
        } else {
            "🔴 No detectado"
        }
    );
    println!(
        "   • FMA (Fused Multiply): {}",
        if report.simd_fma {
            "🟢 Soportado (Aceleración gemv)"
        } else {
            "🔴 No detectado"
        }
    );
    println!(
        "   • AVX-512F (512-bit):   {}",
        if report.simd_avx512f {
            "🟢 Soportado (Ultra)"
        } else {
            "⚪ No disponible (No requerido)"
        }
    );
    println!(
        "   • SSE 4.1:              {}",
        if report.simd_sse41 {
            "🟢 Soportado"
        } else {
            "🔴 No detectado"
        }
    );
    println!(
        "   • ARM NEON:             {}",
        if report.simd_neon {
            "🟢 Soportado"
        } else {
            "⚪ N/A (x86_64)"
        }
    );

    println!("\n🎮 Aceleración por Hardware:");
    println!("   • Backend GPU:          {}", report.gpu_backend);

    println!("\n🧠 Ancho de Banda de Memoria Zero-Copy:");
    println!(
        "   • Rendimiento Medido:   {:.2} GB/s",
        report.mmap_speed_gb_s
    );

    println!("\n--------------------------------------------------------");
    if report.is_optimal {
        println!("🏆 ESTADO: 🟢 ÓPTIMO — El sistema está certificado para inferencia genómica de alta velocidad.");
    } else {
        println!("⚠️ ESTADO: 🟡 COMPATIBLE — Inferencia funcional en modo escalar estándar.");
    }
    println!("========================================================\n");
}
