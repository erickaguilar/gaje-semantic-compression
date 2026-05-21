use crate::compute::event_queue::{EventQueue, SpikeEvent};
use crate::nn::spiking::neuron::SpikingNeuron;

/// El Scheduler coordina la simulación completa basada en eventos.
pub struct NeuromorphicScheduler {
    pub event_queue: EventQueue,
    pub centroides: [f32; 4],
    pub delay_per_layer: u64, // Retardo Δt entre capas en ticks
}

impl NeuromorphicScheduler {
    pub fn new(centroides: [f32; 4], delay_per_layer: u64) -> Self {
        Self {
            event_queue: EventQueue::new(),
            centroides,
            delay_per_layer,
        }
    }

    /// Inyecta un estímulo inicial (ej. un token codificado como spike).
    pub fn inject_spike(&mut self, layer_id: usize, neuron_id: usize, timestamp: u64) {
        self.event_queue.push(SpikeEvent {
            timestamp,
            source_neuron_id: 0, // 0 indica fuente externa/input
            target_layer_id: layer_id,
            target_neuron_id: neuron_id,
        });
    }

    /// Ejecuta un paso de simulación procesando todos los eventos actuales.
    /// Retorna una lista de nuevos disparos generados para ser procesados por capas superiores o externas.
    pub fn step(&mut self, layers: &mut [Vec<SpikingNeuron>]) -> Vec<SpikeEvent> {
        let mut new_spikes = Vec::new();

        // 1. Procesar todos los eventos programados para el tick actual
        while let Some(event) = self.event_queue.pop_active() {
            if event.target_layer_id < layers.len() {
                let layer = &mut layers[event.target_layer_id];
                if event.target_neuron_id < layer.len() {
                    let neuron = &mut layer[event.target_neuron_id];
                    
                    // Integrar el impulso entrante
                    // Nota: En un sistema completo, el input_index dependería de la topología
                    // Por ahora usamos source_neuron_id simplificado
                    neuron.integrate(event.source_neuron_id % neuron.num_weights, &self.centroides);

                    // Verificar si la neurona destino dispara
                    if neuron.check_spike() {
                        // Si dispara, creamos eventos para la siguiente capa
                        let next_layer_id = event.target_layer_id + 1;
                        if next_layer_id < layers.len() {
                            // En un Transformer real, un disparo afectaría a muchas neuronas de la sig. capa
                            // Aquí simulamos la propagación a la neurona con el mismo índice
                            let new_event = SpikeEvent {
                                timestamp: self.event_queue.current_tick + self.delay_per_layer,
                                source_neuron_id: event.target_neuron_id,
                                target_layer_id: next_layer_id,
                                target_neuron_id: event.target_neuron_id, 
                            };
                            self.event_queue.push(new_event);
                            new_spikes.push(new_event);
                        } else {
                            // Es un disparo de la última capa (Output Spike)
                            new_spikes.push(SpikeEvent {
                                timestamp: self.event_queue.current_tick,
                                source_neuron_id: event.target_neuron_id,
                                target_layer_id: next_layer_id,
                                target_neuron_id: 0,
                            });
                        }
                    }
                }
            }
        }

        // 2. Avanzar el reloj
        self.event_queue.advance_tick(1);
        
        new_spikes
    }

    /// Ejecuta la simulación hasta que no queden eventos.
    pub fn run_to_completion(&mut self, layers: &mut [Vec<SpikingNeuron>]) -> Vec<SpikeEvent> {
        let mut all_output_spikes = Vec::new();
        while !self.event_queue.is_empty() {
            let outputs = self.step(layers);
            for out in outputs {
                if out.target_layer_id >= layers.len() {
                    all_output_spikes.push(out);
                }
            }
            
            // Si no hay eventos en el tick actual, saltamos al siguiente evento
            if !self.event_queue.is_empty() {
                self.event_queue.skip_to_next_event();
            }
        }
        all_output_spikes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::spiking::GajeWeight2Bit;

    #[test]
    fn test_scheduler_propagation() {
        let centroides = [0.0, 0.5, 0.8, 1.2];
        let mut scheduler = NeuromorphicScheduler::new(centroides, 2);
        
        // Crear 2 capas con 1 neurona cada una
        let mut layer1_neuron = SpikingNeuron::new(1.0, 1.0, 1);
        layer1_neuron.set_weight(0, GajeWeight2Bit::State11); // Peso 1.2
        
        let mut layer2_neuron = SpikingNeuron::new(1.0, 1.0, 1);
        layer2_neuron.set_weight(0, GajeWeight2Bit::State11); // Peso 1.2

        let mut layers = vec![
            vec![layer1_neuron],
            vec![layer2_neuron]
        ];

        // Inyectar spike inicial en Capa 0
        scheduler.inject_spike(0, 0, 0);

        // Ejecutar
        let outputs = scheduler.run_to_completion(&mut layers);

        // Verificaciones:
        // T=0: Spike inyectado en L0. L0 integra 1.2 y dispara.
        // T=0+2: Spike llega a L1 (delay=2). L1 integra 1.2 y dispara.
        // T=2: Spike de salida de L1.
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].timestamp, 2);
        assert_eq!(outputs[0].target_layer_id, 2); // Capa de salida
    }
}
