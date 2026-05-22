use crate::nn::spiking::neuron::GajeWeight2Bit;

/// Estructura de Capa Neuromórfica Industrial (SoA - Structure of Arrays).
/// Optimizada para localidad de datos, caché y SIMD.
#[derive(Clone, Debug)]
pub struct GajeNeuromorphicLayer {
    pub membrane_potentials: Vec<f32>,
    pub thresholds: Vec<f32>,
    pub decays: Vec<f32>,
    
    /// Pesos empaquetados: 4 pesos de 2-bits por byte.
    /// Layout (Input-Major): [input0_pesos_todas_neuronas, input1_pesos_todas_neuronas, ...]
    /// Cada bloque de input ocupa (num_neurons + 3) / 4 bytes.
    pub packed_weights: Vec<u8>, 
    
    pub num_neurons: usize,
    pub weights_per_neuron: usize,
    pub k_wta: usize, // Límite de Ganadores (K-Winners-Take-All)
}

impl GajeNeuromorphicLayer {
    /// Crea una nueva capa neuromórfica con diseño SoA.
    pub fn new(num_neurons: usize, weights_per_neuron: usize, threshold: f32, decay: f32) -> Self {
        let row_size = (num_neurons + 3) / 4;
        let packed_size = weights_per_neuron * row_size;
        
        Self {
            membrane_potentials: vec![0.0; num_neurons],
            thresholds: vec![threshold; num_neurons],
            decays: vec![decay; num_neurons],
            packed_weights: vec![0; packed_size],
            num_neurons,
            weights_per_neuron,
            k_wta: (num_neurons / 10).max(1), // Por defecto, 10% de ganadores
        }
    }

    /// Integra un spike entrante con una intensidad específica.
    /// `intensity` modula el impacto del centroide (Graded Spiking).
    pub fn integrate_batch(&mut self, input_index: usize, centroides: &[f32; 4], intensity: f32) {
        let row_size = (self.num_neurons + 3) / 4;
        let start_byte = input_index * row_size;
        
        // Sesgo homeostático basal para facilitar disparos en capas profundas
        let homeostatic_bias = 0.01;

        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            
            let n = self.num_neurons;
            let potentials_ptr = self.membrane_potentials.as_mut_ptr();
            let weights_ptr = self.packed_weights.as_ptr().add(start_byte);
            let intensity_v = vdupq_n_f32(intensity);
            let bias_v = vdupq_n_f32(homeostatic_bias);
            
            // Tabla de centroides para vqtbl1q_u8 (4 f32 = 16 bytes)
            let table_v = vld1q_u8(centroides.as_ptr() as *const u8);
            
            // Offsets para generar índices de 4-bytes: [0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3]
            let b0123 = vcreate_u8(0x0302010003020100);
            let offsets = vcombine_u8(b0123, b0123);

            // Máscaras y desplazamientos para expandir 4 pesos (2-bits) a 16 índices
            let masks = vld1q_u8([
                0x03, 0x03, 0x03, 0x03, 
                0x0C, 0x0C, 0x0C, 0x0C, 
                0x30, 0x30, 0x30, 0x30, 
                0xC0, 0xC0, 0xC0, 0xC0
            ].as_ptr());
            
            let shifts = vld1q_s8([
                2, 2, 2, 2, 
                0, 0, 0, 0, 
                -2, -2, -2, -2, 
                -4, -4, -4, -4
            ].as_ptr());
            
            let mut i = 0;
            // Procesamos de a 4 neuronas (1 byte de pesos)
            while i + 4 <= n {
                let byte_idx = i / 4;
                let b = *weights_ptr.add(byte_idx);
                
                let v_b = vdupq_n_u8(b);
                
                // Aplicar máscaras y desplazamientos para obtener indices (peso * 4)
                let indices_base = vandq_u8(v_b, masks);
                let indices_scaled = vshlq_u8(indices_base, shifts);
                let indices = vaddq_u8(indices_scaled, offsets);
                
                // Lookup de 4 floats simultáneos (16 bytes)
                let lookup_res = vqtbl1q_u8(table_v, indices);
                let c_v: float32x4_t = std::mem::transmute(lookup_res);
                
                let mut p_v = vld1q_f32(potentials_ptr.add(i));
                p_v = vfmaq_f32(p_v, c_v, intensity_v); // p = p + c * intensity
                p_v = vaddq_f32(p_v, bias_v);           // + bias
                vst1q_f32(potentials_ptr.add(i), p_v);
                
                i += 4;
            }
            
            // Sobrante
            while i < n {
                let byte_idx = i / 4;
                let bit_shift = (i % 4) * 2;
                let weight_bits = (self.packed_weights[start_byte + byte_idx] >> bit_shift) & 0x03;
                self.membrane_potentials[i] += (centroides[weight_bits as usize] * intensity) + homeostatic_bias;
                i += 1;
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            for i in 0..self.num_neurons {
                let byte_idx = i / 4;
                let bit_shift = (i % 4) * 2;
                let weight_bits = (self.packed_weights[start_byte + byte_idx] >> bit_shift) & 0x03;
                self.membrane_potentials[i] += centroides[weight_bits as usize] * intensity;
            }
        }
    }

    /// Procesa el estado de todas las neuronas para verificar disparos.
    /// Retorna un vector de (índice, intensidad, fase) de las neuronas que dispararon.
    /// Implementa GenomicNorm (v1) para estabilizar la varianza de la señal.
    pub fn check_spikes(&mut self) -> Vec<(usize, f32, u8)> {
        let mut spikes = Vec::new();
        
        // 1. Calcular Varianza Global (para GenomicNorm)
        let n = self.num_neurons;
        let mut sum_sq = 0.0f32;
        for &p in &self.membrane_potentials {
            if p > 0.0 { sum_sq += p * p; }
        }
        let rms = (sum_sq / n as f32 + 1e-6).sqrt();
        
        // 2. Umbral Adaptativo y Disparo
        for i in 0..n {
            let potential = self.membrane_potentials[i];
            let threshold = self.thresholds[i];
            
            if potential >= threshold {
                // GenomicNorm: Si la energía global es muy alta, suavizamos el disparo
                // h_scale actúa como el factor de TEMPERANCIA
                let norm_factor = if rms > 1.0 { 1.0 / rms } else { 1.0 };
                
                // Intensidad graduada normalizada
                let intensity = (1.0 + (potential - threshold) / threshold) * norm_factor;
                
                // Codificación de Fase (Latencia Temporal)
                let excess_ratio = (potential - threshold) / threshold;
                let phase = if excess_ratio >= 1.0 {
                    0
                } else {
                    15 - (excess_ratio * 15.0) as u8
                };
                
                self.membrane_potentials[i] = 0.0; // Reset
                spikes.push((i, intensity, phase));
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
        let row_size = (self.num_neurons + 3) / 4;
        let start_byte = input_idx * row_size;
        let byte_idx = neuron_idx / 4;
        let bit_shift = (neuron_idx % 4) * 2;
        let val = weight as u8;
        
        self.packed_weights[start_byte + byte_idx] &= !(0x03 << bit_shift);
        self.packed_weights[start_byte + byte_idx] |= val << bit_shift;
    }

    /// Refinamiento Genómico Local (Entrenamiento Nativo).
    /// Ajusta los pesos de un conjunto de neuronas basándose en un error local (delta).
    /// - `input_index`: Índice del input que disparó.
    /// - `deltas`: Valor de ajuste para cada neurona (>0 para reforzar, <0 para inhibir).
    /// - `learning_rate`: Probabilidad de que el cambio de bit ocurra (0.0 a 1.0).
    pub fn refine_step(&mut self, input_index: usize, deltas: &[f32], learning_rate: f32) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let row_size = (self.num_neurons + 3) / 4;
        let start_byte = input_index * row_size;

        for (i, &delta) in deltas.iter().enumerate() {
            if i >= self.num_neurons || delta.abs() < 1e-5 {
                continue;
            }

            // Aplicar learning rate como probabilidad de mutación dirigida
            if rng.gen::<f32>() > learning_rate {
                continue;
            }

            let byte_idx = i / 4;
            let bit_shift = (i % 4) * 2;
            let current_byte = self.packed_weights[start_byte + byte_idx];
            let current_weight = (current_byte >> bit_shift) & 0x03;

            let mut new_weight = current_weight;
            if delta > 0.0 {
                if current_weight < 3 { new_weight += 1; }
            } else {
                if current_weight > 0 { new_weight -= 1; }
            }

            if new_weight != current_weight {
                self.packed_weights[start_byte + byte_idx] &= !(0x03 << bit_shift);
                self.packed_weights[start_byte + byte_idx] |= new_weight << bit_shift;
            }
        }
    }

    /// Mecanismo de Homeostasis Genómica para mitigar el olvido catastrófico.
    pub fn apply_homeostasis(&mut self, target_potential: f32) {
        for p in self.membrane_potentials.iter_mut() {
            if *p > target_potential {
                *p *= 0.95; 
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refine_step() {
        let mut layer = GajeNeuromorphicLayer::new(4, 1, 1.0, 0.9);
        let deltas = [1.0, 0.0, 1.0, 0.0];
        layer.refine_step(0, &deltas, 1.0);
        assert_eq!((layer.packed_weights[0] >> 0) & 0x03, 1);
        assert_eq!((layer.packed_weights[0] >> 4) & 0x03, 1);
        
        let deltas_neg = [-1.0, 0.0, 0.0, 0.0];
        layer.refine_step(0, &deltas_neg, 1.0);
        assert_eq!((layer.packed_weights[0] >> 0) & 0x03, 0);
    }

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
        let (idx, intensity, phase) = spikes[0];
        assert_eq!(idx, 2);
        // Umbral 0.5, Potencial 2.0 -> Intensidad = 4.0
        assert_eq!(intensity, 4.0);
        // Exceso = 1.5, Ratio = 1.5 / 0.5 = 3.0. Como Ratio >= 1.0, Phase = 0
        assert_eq!(phase, 0);
        assert_eq!(layer.membrane_potentials[2], 0.0);
    }
}
