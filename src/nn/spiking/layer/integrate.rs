// =============================================================================
// integrate — Integración de impulsos (batch y lagrangiana)
// =============================================================================
use crate::nn::spiking::layer::GajeNeuromorphicLayer;

impl GajeNeuromorphicLayer {
    pub fn integrate_batch(
        &mut self,
        input_index: usize,
        centroides_real: [f32; 4],
        centroides_imag: [f32; 4],
        intensity: f32,
    ) {
        let row_size = (self.num_neurons + 3) / 4;
        let start_byte = input_index * row_size;
        let homeostatic_bias = 0.01;

        /* Temporarily disabled SIMD for NaN diagnosis
        #[cfg(target_arch = "aarch64")]
        unsafe {
            // ... (SIMD code)
        }
        */

        // Fallback robusto (Scalar)
        for i in 0..self.num_neurons {
            let byte_idx = i / 4;
            let bit_shift = (i % 4) * 2;
            let weight_bits = (self.packed_weights[start_byte + byte_idx] >> bit_shift) & 0x03;
            let delta_r = centroides_real[weight_bits as usize] * intensity;
            let delta_im = centroides_imag[weight_bits as usize] * intensity;

            // Reactivando Homeostasis: el bias evita el colapso por silencio (Materia Oscura)
            self.membrane_potentials_real[i] += delta_r + homeostatic_bias;
            self.membrane_potentials_imag[i] += delta_im;

            // Clamping para evitar explosión numérica en inferencia prolongada
            if !self.membrane_potentials_real[i].is_finite() {
                self.membrane_potentials_real[i] = 0.0;
            }
            if !self.membrane_potentials_imag[i].is_finite() {
                self.membrane_potentials_imag[i] = 0.0;
            }
        }

        // Inyectar Anclas de Estabilidad (Residuales de alta precisión)
        // Solo inyectamos si el input_index (peso de entrada) tiene anclas registradas
        // Nota: En SoA, anchor_indices almacena el índice global del peso (neurona_idx * num_inputs + input_idx)
        // Pero para simplificar y dar estabilidad local, buscaremos anclas asociadas a este input_index
        // si el mapeo es (neurona, input).
        for i in 0..self.num_neurons {
            // Buscamos si hay un ancla para el par (i, input_index)
            // Para optimizar, las anclas están ordenadas por fila (neurona) en anchor_row_ptrs
            let a_s = self.anchor_row_ptrs[i];
            let a_e = self.anchor_row_ptrs[i + 1];
            for k in a_s..a_e {
                if self.anchor_indices[k] as usize == input_index {
                    self.membrane_potentials_real[i] += self.anchor_values[k].to_f32() * intensity;
                }
            }
        }
    }

    /// Integra un lote de impulsos usando el Principio de Mínima Acción.
    pub fn integrate_batch_lagrangian(
        &mut self,
        input_index: usize,
        centroides_real: [f32; 4],
        centroides_imag: [f32; 4],
        intensity: f32,
        semantic_resistance: f32,
    ) {
        let row_size = (self.num_neurons + 3) / 4;
        let start_byte = input_index * row_size;

        // Aceleración geodésica frena el avance si hay resistencia
        let acceleration = self
            .lagrangian
            .geodesic_acceleration(-semantic_resistance, false);

        for i in 0..self.num_neurons {
            let byte_idx = i / 4;
            let bit_shift = (i % 4) * 2;
            let weight_bits = (self.packed_weights[start_byte + byte_idx] >> bit_shift) & 0x03;

            let delta_r = centroides_real[weight_bits as usize];
            let delta_im = centroides_imag[weight_bits as usize];
            let velocity = (delta_r.powi(2) + delta_im.powi(2)).sqrt();

            let velocity_adjusted = (velocity + acceleration).max(0.0);
            if velocity > 0.0 {
                let scale = (velocity_adjusted / velocity) * intensity;
                self.membrane_potentials_real[i] += delta_r * scale;
                self.membrane_potentials_imag[i] += delta_im * scale;
            }
        }

        // Inyectar Anclas (siempre con máxima fidelidad)
        for i in 0..self.num_neurons {
            let a_s = self.anchor_row_ptrs[i];
            let a_e = self.anchor_row_ptrs[i + 1];
            for k in a_s..a_e {
                if self.anchor_indices[k] as usize == input_index {
                    self.membrane_potentials_real[i] += self.anchor_values[k].to_f32() * intensity;
                }
            }
        }
    }
}
