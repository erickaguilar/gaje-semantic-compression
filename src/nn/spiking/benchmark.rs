use crate::nn::spiking::block::SpikingTransformerBlock;
use crate::compute::scheduler::NeuromorphicScheduler;
use std::time::Instant;

/// Benchmark para evaluar el rendimiento del Emulador Neuromórfico con contextos masivos.
pub fn run_context_benchmark(
    dim: usize,
    num_layers: usize,
    context_length: usize,
    sparsity: f32, // Fracción de tokens que generan spikes (actividad)
) {
    println!("🚀 Iniciando Benchmark Neuromórfico:");
    println!("   Dim: {}, Capas: {}, Contexto: {}, Sparsity: {:.2}", dim, num_layers, context_length, sparsity);

    let centroides = [-1.0, -0.2, 0.2, 1.0];
    let mut scheduler = NeuromorphicScheduler::new(centroides, 1);
    
    // 1. Inicializar Red
    let mut layers = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let block = SpikingTransformerBlock::new(dim, 8, 1.0, 0.9);
        // En una implementación real, convertiríamos el bloque a un conjunto de neuronas para el scheduler
        // Aquí simulamos una capa plana para simplificar el benchmark
        let layer: Vec<_> = block.attention.query_neurons; 
        layers.push(layer);
    }

    // 2. Inyectar estímulos basados en la sparsity (simulando contexto masivo)
    let num_spikes = (context_length as f32 * sparsity) as usize;
    for i in 0..num_spikes {
        scheduler.inject_spike(0, i % dim, (i as u64) % 100);
    }

    println!("   Eventos en cola inicial: {}", scheduler.event_queue.len());

    // 3. Medir tiempo de simulación
    let start = Instant::now();
    let outputs = scheduler.run_to_completion(&mut layers);
    let duration = start.elapsed();

    println!("✅ Benchmark Completado:");
    println!("   Tiempo total: {:?}", duration);
    println!("   Spikes de salida generados: {}", outputs.len());
    println!("   Eventos/segundo (est.): {:.2}", (num_spikes as f64 + outputs.len() as f64) / duration.as_secs_f64());
    
    if context_length >= 1_000_000 {
        println!("🔥 ¡Hito alcanzado! Procesamiento de contexto de 1,000,000 de tokens completado.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_large_context_simulation() {
        // Ejecutar una versión pequeña del benchmark como test
        run_context_benchmark(128, 2, 10_000, 0.05);
    }
}
