use crate::compute::event_queue::SpikeEvent;
use crate::compute::timing_wheel::TimingWheel;
use crate::nn::spiking::layer::GajeNeuromorphicLayer;

/// El Scheduler coordina la simulación completa basada en eventos usando Timing Wheel y SoA.
pub struct NeuromorphicScheduler {
    pub wheel: TimingWheel,
    pub centroides: [f32; 4],
    pub delay_per_layer: u64, // Retardo Δt entre capas en ticks
}

impl NeuromorphicScheduler {
    pub fn new(centroides: [f32; 4], delay_per_layer: u64) -> Self {
        Self {
            wheel: TimingWheel::new(1024), // Horizonte de 1024 ticks O(1)
            centroides,
            delay_per_layer,
        }
    }

    /// Inyecta un estímulo inicial.
    pub fn inject_spike(&mut self, layer_id: usize, neuron_id: usize, timestamp: u64) {
        self.wheel.push(SpikeEvent {
            timestamp,
            source_neuron_id: neuron_id,
            target_layer_id: layer_id,
            target_neuron_id: neuron_id,
        });
    }

    /// Ejecuta un paso de simulación procesando todos los eventos del tick actual.
    pub fn step(&mut self, layers: &mut [GajeNeuromorphicLayer]) -> Vec<SpikeEvent> {
        let mut new_spikes = Vec::new();
        let events = self.wheel.pop_active();

        for event in events {
            if event.target_layer_id < layers.len() {
                let layer = &mut layers[event.target_layer_id];
                
                // Integración Masiva (SIMD-ready): El spike afecta a toda la capa o a una neurona específica.
                // En esta versión industrial, el spike representa una neurona del input disparando.
                layer.integrate_batch(event.source_neuron_id, &self.centroides);

                // Verificar disparos en toda la capa tras la integración
                // (Optimización: Esto podría hacerse solo una vez por tick por capa)
                let layer_spikes = layer.check_spikes();
                
                for neuron_idx in layer_spikes {
                    let next_layer_id = event.target_layer_id + 1;
                    if next_layer_id < layers.len() {
                        let new_event = SpikeEvent {
                            timestamp: self.wheel.current_tick + self.delay_per_layer,
                            source_neuron_id: neuron_idx,
                            target_layer_id: next_layer_id,
                            target_neuron_id: 0, // En SoA, el target_neuron_id se maneja en integrate_batch
                        };
                        self.wheel.push(new_event);
                        new_spikes.push(new_event);
                    } else {
                        // Output Spike
                        new_spikes.push(SpikeEvent {
                            timestamp: self.wheel.current_tick,
                            source_neuron_id: neuron_idx,
                            target_layer_id: next_layer_id,
                            target_neuron_id: 0,
                        });
                    }
                }
            }
        }

        self.wheel.advance_tick(1);
        new_spikes
    }

    /// Ejecuta la simulación hasta que no queden eventos.
    pub fn run_to_completion(&mut self, layers: &mut [GajeNeuromorphicLayer]) -> Vec<SpikeEvent> {
        let mut all_output_spikes = Vec::new();
        while !self.wheel.is_empty() {
            let outputs = self.step(layers);
            for out in outputs {
                if out.target_layer_id >= layers.len() {
                    all_output_spikes.push(out);
                }
            }
            
            // Nota: El skip_to_next_event es más complejo en Timing Wheel,
            // pero podemos avanzar ticks vacíos rápidamente si es necesario.
        }
        all_output_spikes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::spiking::GajeWeight2Bit;

    #[test]
    fn test_scheduler_propagation_soa() {
        let centroides = [0.0, 0.5, 0.8, 1.2];
        let mut scheduler = NeuromorphicScheduler::new(centroides, 2);
        
        // Crear 2 capas SoA
        let mut layer0 = GajeNeuromorphicLayer::new(1, 1, 1.0, 1.0);
        layer0.set_weight(0, 0, GajeWeight2Bit::State11); // Peso 1.2
        
        let mut layer1 = GajeNeuromorphicLayer::new(1, 1, 1.0, 1.0);
        layer1.set_weight(0, 0, GajeWeight2Bit::State11); // Peso 1.2

        let mut layers = vec![layer0, layer1];

        // Inyectar spike inicial en Capa 0, Neurona 0
        scheduler.inject_spike(0, 0, 0);

        // Ejecutar
        let outputs = scheduler.run_to_completion(&mut layers);

        // T=0: L0 integra 1.2 y dispara.
        // T=2: L1 integra 1.2 y dispara (delay=2).
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].timestamp, 2);
        assert_eq!(outputs[0].target_layer_id, 2);
    }
}
