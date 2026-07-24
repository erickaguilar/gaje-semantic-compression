use _impl::compute::power::{CpuCluster, PowerManager};
use std::time::Instant;

fn main() {
    println!("🔋 GAJE-Flow: Demo de Gestión de Energía Consciente (big.LITTLE)");

    let pm = PowerManager::detect();
    println!("{}", pm.summary());

    // 1. Tarea en núcleos de eficiencia (LITTLE)
    match pm.set_thread_affinity(CpuCluster::Little) {
        Ok(_) => {
            println!("[LITTLE] Iniciando tarea de fondo (Cores de eficiencia)...");
            let start = Instant::now();
            let mut sum = 0u64;
            for i in 0..10_000_000 {
                sum = sum.wrapping_add(i);
            }
            println!(
                "[LITTLE] Tarea completada en {:?} (Resultado: {})",
                start.elapsed(),
                sum
            );
        }
        Err(e) => println!("[!] No se pudo asignar cluster LITTLE: {}", e),
    }

    // 2. Tarea en núcleos de máximo rendimiento (Big)
    match pm.set_thread_affinity(CpuCluster::Big) {
        Ok(_) => {
            println!("\n[BIG] Iniciando tarea de alta prioridad (Cores de rendimiento)...");
            let start = Instant::now();
            let mut sum = 0u64;
            for i in 0..10_000_000 {
                sum = sum.wrapping_add(i);
            }
            println!(
                "[BIG] Tarea completada en {:?} (Resultado: {})",
                start.elapsed(),
                sum
            );
        }
        Err(e) => println!("\n[!] No se pudo asignar cluster BIG: {}. Esto es común si el núcleo está dormido por ahorro de energía.", e),
    }

    println!(
        "\n✅ Gestión de energía validada. El sistema maneja las restricciones de afinidad de CPU."
    );
}
