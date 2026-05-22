use crate::nn::spiking::layer::GajeNeuromorphicLayer;
use crate::compute::scheduler::NeuromorphicScheduler;
use std::time::Instant;

/// Benchmark Industrial para evaluar el rendimiento de la arquitectura SoA y Timing Wheel.
pub fn run_context_benchmark(
    dim: usize,
    num_layers: usize,
    context_length: usize,
    sparsity: f32, 
) {
    println!("🚀 Iniciando Benchmark Neuromórfico Industrial (SoA + Timing Wheel):");
    println!("   Dim: {}, Capas: {}, Contexto: {}, Sparsity: {:.2}", dim, num_layers, context_length, sparsity);

    let centroides = [-1.0, -0.2, 0.2, 1.0];
    let mut scheduler = NeuromorphicScheduler::new(centroides, 1);
    
    // 1. Inicializar Red SoA
    let mut layers = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        // En diseño SoA, una capa contiene todas las neuronas contiguas
        let layer = GajeNeuromorphicLayer::new(dim, dim, 0.5, 0.95);
        layers.push(layer);
    }

    // 2. Inyectar estímulos (simulando contexto masivo)
    let num_spikes = (context_length as f32 * sparsity) as usize;
    for i in 0..num_spikes {
        scheduler.inject_spike(0, i % dim, (i as u64) % 1024, 0, 1.0);
    }

    println!("   Eventos programados (SoA): {}", num_spikes);

    // 3. Medir tiempo de simulación
    let start = Instant::now();
    let outputs = scheduler.run_to_completion(&mut layers);
    let duration = start.elapsed();

    println!("✅ Benchmark Completado:");
    println!("   Tiempo total: {:?}", duration);
    println!("   Spikes de salida generados: {}", outputs.len());
    println!("   Throughput: {:.2} eventos/segundo", (num_spikes as f64) / duration.as_secs_f64());
    
    if context_length >= 1_000_000 {
        println!("🔥 ¡Hito Industrial alcanzado! Contexto de 1,000,000 procesado con arquitectura SoA.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soa_benchmark() {
        run_context_benchmark(128, 2, 10_000, 0.05);
    }
}
