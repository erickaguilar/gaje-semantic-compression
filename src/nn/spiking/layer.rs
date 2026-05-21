use crate::nn::spiking::neuron::GajeWeight2Bit;

/// Estructura de Capa Neuromórfica Industrial (SoA - Structure of Arrays).
/// Optimizada para localidad de datos, caché y SIMD.
#[derive(Clone, Debug)]
pub struct GajeNeuromorphicLayer {
    pub membrane_potentials: Vec<f32>,
    pub thresholds: Vec<f32>,
    pub decays: Vec<f32>,
    
    /// Pesos empaquetados: 4 pesos de 2-bits por byte.
    /// Layout: [neurona0_pesos..., neurona1_pesos..., ...]
    pub packed_weights: Vec<u8>, 
    
    pub num_neurons: usize,
    pub weights_per_neuron: usize,
}

impl GajeNeuromorphicLayer {
    /// Crea una nueva capa neuromórfica con diseño SoA.
    pub fn new(num_neurons: usize, weights_per_neuron: usize, threshold: f32, decay: f32) -> Self {
        let total_weights = num_neurons * weights_per_neuron;
        let packed_size = (total_weights + 3) / 4;
        
        Self {
            membrane_potentials: vec![0.0; num_neurons],
            thresholds: vec![threshold; num_neurons],
            decays: vec![decay; num_neurons],
            packed_weights: vec![0; packed_size],
            num_neurons,
            weights_per_neuron,
        }
    }

    /// Integra un spike entrante en todas las neuronas de la capa de forma masiva.
    /// `input_index` es el índice de la neurona de la capa anterior que disparó.
    /// ¡Cero Multiplicaciones!: Solo sumas de centroides.
    pub fn integrate_batch(&mut self, input_index: usize, centroides: &[f32; 4]) {
        // Este bucle es un candidato perfecto para auto-vectorización SIMD
        for i in 0..self.num_neurons {
            let global_weight_index = i * self.weights_per_neuron + input_index;
            let byte_index = global_weight_index / 4;
            let bit_shift = (global_weight_index % 4) * 2;
            
            // Extracción rápida de 2-bits
            let weight_bits = (self.packed_weights[byte_index] >> bit_shift) & 0x03;
            
            // Suma del centroide al potencial de la neurona i
            self.membrane_potentials[i] += centroides[weight_bits as usize];
        }
    }

    /// Procesa el estado de todas las neuronas para verificar disparos.
    /// Retorna un vector de índices de neuronas que dispararon en este tick.
    pub fn check_spikes(&mut self) -> Vec<usize> {
        let mut spikes = Vec::new();
        
        for i in 0..self.num_neurons {
            if self.membrane_potentials[i] >= self.thresholds[i] {
                self.membrane_potentials[i] = 0.0; // Reset
                spikes.push(i);
            } else {
                // Aplicar fuga de energía (decay)
                if self.membrane_potentials[i] > 0.0 {
                    self.membrane_potentials[i] *= self.decays[i];
                }
            }
        }
        
        spikes
    }

    /// Helper para establecer un peso individual (usado en evolución/entrenamiento).
    pub fn set_weight(&mut self, neuron_idx: usize, input_idx: usize, weight: GajeWeight2Bit) {
        let global_idx = neuron_idx * self.weights_per_neuron + input_idx;
        let byte_idx = global_idx / 4;
        let bit_shift = (global_idx % 4) * 2;
        let val = weight as u8;
        
        self.packed_weights[byte_idx] &= !(0x03 << bit_shift);
        self.packed_weights[byte_idx] |= val << bit_shift;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soa_integration() {
        let centroides = [-1.0, -0.2, 0.2, 1.0];
        let mut layer = GajeNeuromorphicLayer::new(10, 5, 0.5, 0.9);
        
        // Configurar pesos para la neurona 2 desde el input 0
        layer.set_weight(2, 0, GajeWeight2Bit::State11); // 1.0
        
        // Integrar spike del input 0
        layer.integrate_batch(0, &centroides);
        
        assert_eq!(layer.membrane_potentials[2], 1.0);
        assert_eq!(layer.membrane_potentials[0], -1.0); // Default state 00 es centroide[0]
        
        let spikes = layer.check_spikes();
        assert!(spikes.contains(&2));
        assert_eq!(layer.membrane_potentials[2], 0.0);
    }
}
