use _impl::compute::power::{PowerManager, CpuCluster};
use std::thread;
use std::time::Instant;

fn main() {
    println!("🔋 GAJE-Flow: Demo de Gestión de Energía Consciente (big.LITTLE)");
    
    let pm = PowerManager::detect();
    println!("{}", pm.summary());

    // 1. Tarea en núcleos de eficiencia (LITTLE)
    pm.set_thread_affinity(CpuCluster::Little).expect("Error al asignar LITTLE");
    println!("[LITTLE] Iniciando tarea de fondo (Cores de eficiencia)...");
    
    let start = Instant::now();
    let mut sum = 0u64;
    for i in 0..10_000_000 {
        sum = sum.wrapping_add(i);
    }
    println!("[LITTLE] Tarea completada en {:?} (Resultado: {})", start.elapsed(), sum);

    // 2. Tarea en núcleos de máximo rendimiento (Big)
    pm.set_thread_affinity(CpuCluster::Big).expect("Error al asignar Big");
    println!("\n[BIG] Iniciando tarea de alta prioridad (Cores de rendimiento)...");
    
    let start = Instant::now();
    let mut sum = 0u64;
    for i in 0..10_000_000 {
        sum = sum.wrapping_add(i);
    }
    println!("[BIG] Tarea completada en {:?} (Resultado: {})", start.elapsed(), sum);

    println!("\n✅ Gestión de energía validada. El sistema puede conmutar hilos entre clusters dinámicamente.");
}
