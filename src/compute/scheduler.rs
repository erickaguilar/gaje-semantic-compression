use crate::compute::event_queue::SpikeEvent;
use crate::compute::timing_wheel::TimingWheel;
use crate::nn::spiking::layer::GajeNeuromorphicLayer;
use pyo3::prelude::*;
use std::collections::HashSet;

/// El Scheduler coordina la simulación completa basada en eventos usando Timing Wheel y SoA.
#[pyclass]
pub struct NeuromorphicScheduler {
    pub wheel: TimingWheel,
    pub centroides: [f32; 4],
    pub delay_per_layer: u64, // Retardo Δt entre capas en ticks
}

#[pymethods]
impl NeuromorphicScheduler {
    #[new]
    pub fn new(centroides: [f32; 4], delay_per_layer: u64) -> Self {
        Self {
            wheel: TimingWheel::new(1024), // Horizonte de 1024 ticks O(1)
            centroides,
            delay_per_layer,
        }
    }

    /// Inyecta un estímulo inicial con precisión de fase e intensidad.
    pub fn inject_spike(&mut self, layer_id: usize, neuron_id: usize, timestamp: u64, phase_offset: u8, intensity: f32) {
        self.wheel.push(SpikeEvent {
            timestamp,
            phase_offset,
            intensity,
            source_neuron_id: neuron_id,
            target_layer_id: layer_id,
            target_neuron_id: neuron_id,
        });
    }

    /// Pasarela para Python: ejecuta un paso.
    #[pyo3(name = "step")]
    pub fn step_py(&mut self, mut layers: Vec<PyRefMut<GajeNeuromorphicLayer>>) -> Vec<SpikeEvent> {
        let mut layers_mut: Vec<&mut GajeNeuromorphicLayer> = layers.iter_mut().map(|l| &mut **l).collect();
        self.step(&mut layers_mut)
    }

    /// Pasarela para Python: ejecuta hasta completar.
    #[pyo3(name = "run_to_completion")]
    pub fn run_to_completion_py(&mut self, mut layers: Vec<PyRefMut<GajeNeuromorphicLayer>>, max_ticks: u64) -> Vec<SpikeEvent> {
        let mut layers_mut: Vec<&mut GajeNeuromorphicLayer> = layers.iter_mut().map(|l| &mut **l).collect();
        let mut all_output_spikes = Vec::new();
        let mut ticks = 0;
        while !self.wheel.is_empty() && ticks < max_ticks {
            let outputs = self.step(&mut layers_mut);
            for out in outputs {
                if out.target_layer_id >= layers_mut.len() {
                    all_output_spikes.push(out);
                }
            }
            ticks += 1;
        }
        all_output_spikes
    }
}

impl NeuromorphicScheduler {
    /// Ejecuta un paso de simulación (Interno Rust).
    pub fn step(&mut self, layers: &mut [&mut GajeNeuromorphicLayer]) -> Vec<SpikeEvent> {
        let mut new_spikes = Vec::new();
        let events = self.wheel.pop_active();
        
        if events.is_empty() {
            self.wheel.advance_tick(1);
            return new_spikes;
        }

        // 1. Fase de Integración
        let mut affected_layers = HashSet::new();
        for event in &events {
            if event.target_layer_id < layers.len() {
                layers[event.target_layer_id].integrate_batch(event.source_neuron_id, self.centroides, event.intensity);
                affected_layers.insert(event.target_layer_id);
            }
        }

        // 2. Fase de Disparo e Inhibición Lateral (K-WTA)
        for layer_id in affected_layers {
            let layer = &mut *layers[layer_id];
            let mut layer_spikes = layer.check_spikes();
            
            if layer_spikes.is_empty() { continue; }

            layer_spikes.sort_by(|a, b| {
                let res = a.2.cmp(&b.2);
                if res == std::cmp::Ordering::Equal {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    res
                }
            });

            let num_winners = layer_spikes.len().min(layer.k_wta);
            
            for i in 0..num_winners {
                let (neuron_idx, intensity, phase) = layer_spikes[i];
                let next_layer_id = layer_id + 1;
                
                if next_layer_id < layers.len() {
                    let new_event = SpikeEvent {
                        timestamp: self.wheel.current_tick + self.delay_per_layer,
                        phase_offset: phase,
                        intensity,
                        source_neuron_id: neuron_idx,
                        target_layer_id: next_layer_id,
                        target_neuron_id: 0,
                    };
                    self.wheel.push(new_event);
                    new_spikes.push(new_event);
                } else {
                    new_spikes.push(SpikeEvent {
                        timestamp: self.wheel.current_tick,
                        phase_offset: phase,
                        intensity,
                        source_neuron_id: neuron_idx,
                        target_layer_id: next_layer_id,
                        target_neuron_id: 0,
                    });
                }
            }
        }

        self.wheel.advance_tick(1);
        new_spikes
    }

    /// Ejecuta la simulación hasta completar (Interno Rust).
    pub fn run_to_completion(&mut self, layers: &mut [GajeNeuromorphicLayer]) -> Vec<SpikeEvent> {
        let num_layers = layers.len();
        let mut layers_refs: Vec<&mut GajeNeuromorphicLayer> = layers.iter_mut().collect();
        let mut all_output_spikes = Vec::new();
        while !self.wheel.is_empty() {
            let outputs = self.step(&mut layers_refs);
            for out in outputs {
                if out.target_layer_id >= num_layers {
                    all_output_spikes.push(out);
                }
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
    fn test_scheduler_propagation_soa() {
        let centroides = [0.0, 0.5, 0.8, 1.2];
        let mut scheduler = NeuromorphicScheduler::new(centroides, 2);
        
        // Crear 2 capas SoA
        let mut layer0 = GajeNeuromorphicLayer::new(1, 1, 1.0, 1.0);
        layer0.set_weight(0, 0, GajeWeight2Bit::State11 as u8); // Peso 1.2
        
        let mut layer1 = GajeNeuromorphicLayer::new(1, 1, 1.0, 1.0);
        layer1.set_weight(0, 0, GajeWeight2Bit::State11 as u8); // Peso 1.2

        let mut layers = vec![layer0, layer1];

        // Inyectar spike inicial en Capa 0, Neurona 0 (Fase 0, Intensidad 1.0)
        scheduler.inject_spike(0, 0, 0, 0, 1.0);

        // Ejecutar
        let outputs = scheduler.run_to_completion(&mut layers);

        // T=0: L0 integra 1.2 y dispara.
        // T=2: L1 integra 1.2 y dispara (delay=2).
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].timestamp, 2);
        assert_eq!(outputs[0].target_layer_id, 2);
    }
}
