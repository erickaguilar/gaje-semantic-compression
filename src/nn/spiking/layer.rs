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

    /// Integra un spike entrante con una intensidad específica.
    /// `intensity` modula el impacto del centroide (Graded Spiking).
    pub fn integrate_batch(&mut self, input_index: usize, centroides: &[f32; 4], intensity: f32) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            
            let n = self.num_neurons;
            let weights_per_neuron = self.weights_per_neuron;
            let potentials_ptr = self.membrane_potentials.as_mut_ptr();
            let intensity_v = vdupq_n_f32(intensity);
            
            let mut i = 0;
            while i + 4 <= n {
                let mut p_v = vld1q_f32(potentials_ptr.add(i));
                
                let mut c_v = [0.0f32; 4];
                for j in 0..4 {
                    let neuron_idx = i + j;
                    let global_idx = neuron_idx * weights_per_neuron + input_index;
                    let byte_idx = global_idx / 4;
                    let bit_shift = (global_idx % 4) * 2;
                    let weight_bits = (self.packed_weights[byte_idx] >> bit_shift) & 0x03;
                    c_v[j] = centroides[weight_bits as usize];
                }
                
                // Modular incremento por intensidad y sumar
                let inc_v = vmulq_f32(vld1q_f32(c_v.as_ptr()), intensity_v);
                p_v = vaddq_f32(p_v, inc_v);
                vst1q_f32(potentials_ptr.add(i), p_v);
                
                i += 4;
            }
            
            while i < n {
                let global_weight_index = i * weights_per_neuron + input_index;
                let byte_index = global_weight_index / 4;
                let bit_shift = (global_weight_index % 4) * 2;
                let weight_bits = (self.packed_weights[byte_index] >> bit_shift) & 0x03;
                self.membrane_potentials[i] += centroides[weight_bits as usize] * intensity;
                i += 1;
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            for i in 0..self.num_neurons {
                let global_weight_index = i * self.weights_per_neuron + input_index;
                let byte_index = global_weight_index / 4;
                let bit_shift = (global_weight_index % 4) * 2;
                let weight_bits = (self.packed_weights[byte_index] >> bit_shift) & 0x03;
                self.membrane_potentials[i] += centroides[weight_bits as usize] * intensity;
            }
        }
    }

    /// Procesa el estado de todas las neuronas para verificar disparos.
    /// Retorna un vector de (índice, intensidad) de las neuronas que dispararon.
    pub fn check_spikes(&mut self) -> Vec<(usize, f32)> {
        let mut spikes = Vec::new();
        
        for i in 0..self.num_neurons {
            let potential = self.membrane_potentials[i];
            let threshold = self.thresholds[i];
            
            if potential >= threshold {
                // Cálculo de intensidad graduada (residuo)
                // Al menos 1.0 de intensidad, escalado por el exceso de energía
                let intensity = 1.0 + (potential - threshold) / threshold;
                
                self.membrane_potentials[i] = 0.0; // Reset
                spikes.push((i, intensity));
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
        
        // Integrar spike del input 0 con intensidad 2.0
        layer.integrate_batch(0, &centroides, 2.0);
        
        // 1.0 (peso) * 2.0 (intensidad) = 2.0
        assert_eq!(layer.membrane_potentials[2], 2.0);
        
        let spikes = layer.check_spikes();
        assert_eq!(spikes.len(), 1);
        let (idx, intensity) = spikes[0];
        assert_eq!(idx, 2);
        // Umbral 0.5, Potencial 2.0 -> Intensidad = 1.0 + (2.0 - 0.5)/0.5 = 1.0 + 3.0 = 4.0
        assert_eq!(intensity, 4.0);
        assert_eq!(layer.membrane_potentials[2], 0.0);
    }
}
