// =============================================================================
// spike — Disparo, refine, homeostasis e inhibición lateral
// =============================================================================
use crate::nn::spiking::layer::GajeNeuromorphicLayer;

impl GajeNeuromorphicLayer {
    pub fn check_spikes(&mut self) -> Vec<(usize, f32, u8)> {
        let mut spikes = Vec::new();
        let n = self.num_neurons;

        // Calcular RMS basado en magnitud compleja
        let mut sum_sq = 0.0f32;
        for i in 0..n {
            let r = self.membrane_potentials_real[i];
            let im = self.membrane_potentials_imag[i];
            let mag_sq = r * r + im * im;
            if mag_sq.is_finite() {
                sum_sq += mag_sq;
            }
        }

        let rms = (sum_sq / n as f32 + 1e-6).sqrt();
        let alpha = 0.15;
        if rms.is_finite() {
            self.rms_ema = (1.0 - alpha) * self.rms_ema + alpha * rms;
        }

        for i in 0..n {
            let r = self.membrane_potentials_real[i];
            let im = self.membrane_potentials_imag[i];
            let magnitude = (r * r + im * im).sqrt();
            let threshold = self.thresholds[i];

            if magnitude.is_finite() && magnitude >= threshold {
                let norm_factor = if self.rms_ema.is_finite() && self.rms_ema > 1.0 {
                    1.0 / self.rms_ema
                } else {
                    1.0
                };
                let intensity = (magnitude / threshold) * norm_factor;
                let excess_ratio = (magnitude - threshold) / threshold;
                // Phase coding (latencia): mayor magnitud dispara antes (phase_offset menor)
                let phase = if excess_ratio >= 1.0 {
                    0
                } else {
                    (15.0 * (1.0 - excess_ratio)) as u8
                };

                self.membrane_potentials_real[i] = 0.0;
                self.membrane_potentials_imag[i] = 0.0;
                spikes.push((i, intensity, phase));
            } else if magnitude > 0.0 {
                self.membrane_potentials_real[i] *= self.decays[i];
                self.membrane_potentials_imag[i] *= self.decays[i];

                // Limpieza de seguridad
                if !self.membrane_potentials_real[i].is_finite() {
                    self.membrane_potentials_real[i] = 0.0;
                }
                if !self.membrane_potentials_imag[i].is_finite() {
                    self.membrane_potentials_imag[i] = 0.0;
                }
            }
        }
        spikes
    }

    /// Verifica disparos aplicando el retraso Lagrangiano (fricción semántica).
    pub fn check_spikes_physical(&mut self, semantic_resistance: f32) -> Vec<(usize, f32, u8)> {
        let mut spikes = Vec::new();
        let n = self.num_neurons;

        // Retraso físico basado en la resistencia (Energía Potencial)
        let lagrangian_delay = self.lagrangian.calculate_timing_delay(semantic_resistance);
        let phase_delay_base = (lagrangian_delay * 16.0) as u8;

        for i in 0..n {
            let r = self.membrane_potentials_real[i];
            let im = self.membrane_potentials_imag[i];
            let magnitude = (r * r + im * im).sqrt();
            let threshold = self.thresholds[i];

            if magnitude >= threshold {
                let intensity = magnitude / threshold;
                let excess_ratio = (magnitude - threshold) / threshold;

                // Phase coding original (latencia por magnitud) + Retraso Lagrangiano
                let base_phase = if excess_ratio >= 1.0 {
                    0
                } else {
                    (15.0 * (1.0 - excess_ratio)) as u8
                };
                let phase = (base_phase + phase_delay_base).min(15);

                self.membrane_potentials_real[i] = 0.0;
                self.membrane_potentials_imag[i] = 0.0;
                spikes.push((i, intensity, phase));
            } else if magnitude > 0.0 {
                self.membrane_potentials_real[i] *= self.decays[i];
                self.membrane_potentials_imag[i] *= self.decays[i];
            }
        }
        spikes
    }

    pub fn set_weight(&mut self, neuron_idx: usize, input_idx: usize, val: u8) {
        let row_size = (self.num_neurons + 3) / 4;
        let start_byte = input_idx * row_size;
        let byte_idx = neuron_idx / 4;
        let bit_shift = (neuron_idx % 4) * 2;
        self.packed_weights[start_byte + byte_idx] &= !(0x03 << bit_shift);
        self.packed_weights[start_byte + byte_idx] |= (val & 0x03) << bit_shift;
    }

    pub fn refine_step(&mut self, input_index: usize, deltas: Vec<f32>, learning_rate: f32) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let row_size = (self.num_neurons + 3) / 4;
        let start_byte = input_index * row_size;

        if start_byte + row_size > self.packed_weights.len() {
            return;
        }

        for (i, &delta) in deltas.iter().enumerate() {
            if i >= self.num_neurons || delta.abs() < 1e-5 {
                continue;
            }
            if rng.gen::<f32>() > learning_rate {
                continue;
            }
            let byte_idx = i / 4;
            let bit_shift = (i % 4) * 2;
            let current_byte = self.packed_weights[start_byte + byte_idx];
            let current_weight = (current_byte >> bit_shift) & 0x03;
            let mut new_weight = current_weight;
            if delta > 0.0 {
                if current_weight < 3 {
                    new_weight += 1;
                }
            } else if current_weight > 0 {
                new_weight -= 1;
            }
            if new_weight != current_weight {
                self.packed_weights[start_byte + byte_idx] &= !(0x03 << bit_shift);
                self.packed_weights[start_byte + byte_idx] |= new_weight << bit_shift;
            }
        }
    }

    pub fn apply_homeostasis(&mut self, target_potential: f32) {
        for i in 0..self.num_neurons {
            let r = self.membrane_potentials_real[i];
            let im = self.membrane_potentials_imag[i];
            let mag = (r * r + im * im).sqrt();
            if mag > target_potential {
                self.membrane_potentials_real[i] *= 0.95;
                self.membrane_potentials_imag[i] *= 0.95;
            }
        }
    }

    /// Aplica Inhibición Lateral Temporal (K-WTA).
    /// Reduce el potencial de todas las neuronas basado en la intensidad de los ganadores.
    pub fn apply_lateral_inhibition(
        &mut self,
        winners: &[(usize, f32, u8)],
        inhibition_factor: f32,
    ) {
        if winners.is_empty() {
            return;
        }

        // Calcular la fuerza total de inhibición basada en el ganador más rápido/fuerte
        let total_inhibition = winners.iter().map(|w| w.1).sum::<f32>() * inhibition_factor;
        let decay = (1.0 - total_inhibition).max(0.1);

        for i in 0..self.num_neurons {
            // No inhibir a los ganadores (ya fueron reseteados o están en periodo refractario)
            let is_winner = winners.iter().any(|w| w.0 == i);
            if !is_winner {
                self.membrane_potentials_real[i] *= decay;
                self.membrane_potentials_imag[i] *= decay;
            }
        }
    }
}
