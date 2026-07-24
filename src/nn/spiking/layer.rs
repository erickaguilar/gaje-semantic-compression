#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::compute::lagrangian::LagrangianEngine;

/// Estructura de Capa Neuromórfica Industrial (SoA - Structure of Arrays).
/// Optimizada para localidad de datos, caché y SIMD.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Debug)]
pub struct GajeNeuromorphicLayer {
    pub membrane_potentials_real: Vec<f32>,
    pub membrane_potentials_imag: Vec<f32>,
    pub thresholds: Vec<f32>,
    pub decays: Vec<f32>,
    /// Pesos empaquetados: 4 pesos de 2-bits por byte.
    pub packed_weights: Vec<u8>,
    /// Anclas de Estabilidad: Pesos de alta precisión (F16) para núcleos semánticos.
    pub anchor_indices: Vec<u32>,
    pub anchor_values: Vec<half::f16>,
    pub anchor_row_ptrs: Vec<usize>,
    pub num_neurons: usize,
    pub weights_per_neuron: usize,
    pub k_wta: usize,
    pub rms_ema: f32,
    pub lagrangian: LagrangianEngine, // Motor de física semántica
}

#[cfg_attr(feature = "python", pymethods)]
impl GajeNeuromorphicLayer {
    #[cfg(feature = "python")]
    #[new]
    pub fn new_py(
        num_neurons: usize,
        weights_per_neuron: usize,
        threshold: f32,
        decay: f32,
    ) -> Self {
        Self::new(num_neurons, weights_per_neuron, threshold, decay)
    }

    #[cfg(feature = "python")]
    #[getter]
    pub fn get_membrane_potentials_real(&self) -> Vec<f32> {
        self.membrane_potentials_real.clone()
    }

    #[cfg(feature = "python")]
    #[getter]
    pub fn get_membrane_potentials_imag(&self) -> Vec<f32> {
        self.membrane_potentials_imag.clone()
    }

    #[cfg(feature = "python")]
    #[getter]
    pub fn get_packed_weights<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        Ok(pyo3::types::PyBytes::new(py, &self.packed_weights))
    }

    #[cfg(feature = "python")]
    pub fn load_packed_weights(&mut self, data: Vec<u8>) -> PyResult<()> {
        if data.len() != self.packed_weights.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Weight size mismatch: expected {}, got {}",
                self.packed_weights.len(),
                data.len()
            )));
        }
        self.packed_weights = data;
        Ok(())
    }

    pub fn reset_potentials(&mut self) {
        self.membrane_potentials_real.fill(0.0);
        self.membrane_potentials_imag.fill(0.0);
    }
}

impl GajeNeuromorphicLayer {
    pub fn new(num_neurons: usize, weights_per_neuron: usize, threshold: f32, decay: f32) -> Self {
        let row_size = (num_neurons + 3) / 4;
        let packed_size = weights_per_neuron * row_size;

        // Inicialización de alta entropía (Ruido blanco genómico)
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut packed_weights = vec![0u8; packed_size];
        for byte in packed_weights.iter_mut() {
            *byte = rng.gen();
        }

        Self {
            membrane_potentials_real: vec![0.0; num_neurons],
            membrane_potentials_imag: vec![0.0; num_neurons],
            thresholds: vec![threshold; num_neurons],
            decays: vec![decay; num_neurons],
            packed_weights,
            anchor_indices: Vec::new(),
            anchor_values: Vec::new(),
            anchor_row_ptrs: vec![0; num_neurons + 1],
            num_neurons,
            weights_per_neuron,
            k_wta: (num_neurons / 10).max(1),
            rms_ema: 1.0,
            lagrangian: LagrangianEngine::new(1.0),
        }
    }

    pub fn anchors_sparse_buffer(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GAJE");
        let count = self.anchor_indices.len();
        out.extend_from_slice(&(count as u32).to_le_bytes());
        for &idx in self.anchor_indices.iter() {
            out.extend_from_slice(&idx.to_le_bytes());
        }
        for &val in self.anchor_values.iter() {
            out.extend_from_slice(&val.to_le_bytes());
        }
        for &ptr in self.anchor_row_ptrs.iter() {
            out.extend_from_slice(&(ptr as u64).to_le_bytes());
        }
        out
    }

    pub fn load_anchors_from_u8(&mut self, anchors_u8: &[u8]) {
        if anchors_u8.len() >= 4 && &anchors_u8[0..4] == b"GAJE" {
            let count = u32::from_le_bytes(anchors_u8[4..8].try_into().unwrap()) as usize;
            let mut indices = Vec::with_capacity(count);
            let mut values = Vec::with_capacity(count);
            let mut row_ptrs = vec![0; self.num_neurons + 1];
            let idx_s = 8;
            let val_s = idx_s + count * 4;
            let ptr_s = val_s + count * 2;
            for i in 0..count {
                indices.push(u32::from_le_bytes(
                    anchors_u8[idx_s + i * 4..idx_s + i * 4 + 4]
                        .try_into()
                        .unwrap(),
                ));
                values.push(half::f16::from_le_bytes(
                    anchors_u8[val_s + i * 2..val_s + i * 2 + 2]
                        .try_into()
                        .unwrap(),
                ));
            }
            for i in 0..=self.num_neurons {
                row_ptrs[i] = u64::from_le_bytes(
                    anchors_u8[ptr_s + i * 8..ptr_s + i * 8 + 8]
                        .try_into()
                        .unwrap(),
                ) as usize;
            }
            self.anchor_indices = indices;
            self.anchor_values = values;
            self.anchor_row_ptrs = row_ptrs;
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::spiking::neuron::GajeWeight2Bit;

    #[test]
    fn test_refine_step() {
        let mut layer = GajeNeuromorphicLayer::new(4, 1, 1.0, 0.9);
        // Inicializar pesos a 0 para determinismo
        layer.packed_weights.fill(0);

        let deltas = vec![1.0, 0.0, 1.0, 0.0];
        layer.refine_step(0, deltas, 1.0);
        assert_eq!(layer.packed_weights[0] & 0x03, 1);
        assert_eq!((layer.packed_weights[0] >> 4) & 0x03, 1);
        let deltas_neg = vec![-1.0, 0.0, 0.0, 0.0];
        layer.refine_step(0, deltas_neg, 1.0);
        assert_eq!(layer.packed_weights[0] & 0x03, 0);
    }

    #[test]
    fn test_soa_integration() {
        let c_r = [0.0, -0.2, 0.2, 1.0]; // State 00 es neutral
        let c_im = [0.0, 0.0, 0.0, 0.0];
        let mut layer = GajeNeuromorphicLayer::new(10, 5, 0.5, 0.9);
        // Inicializar pesos a 0 para determinismo
        layer.packed_weights.fill(0);

        layer.set_weight(2, 0, GajeWeight2Bit::State11 as u8);
        layer.integrate_batch(0, c_r, c_im, 2.0);
        assert!(layer.membrane_potentials_real[2] > 2.0);
        let spikes = layer.check_spikes();
        assert_eq!(spikes.len(), 1);
        assert_eq!(spikes[0].0, 2);
        assert!(spikes[0].1 > 4.0);
        assert_eq!(spikes[0].2, 0);
        assert_eq!(layer.membrane_potentials_real[2], 0.0);
    }

    #[test]
    fn test_soa_lagrangian_integration() {
        let c_r = [0.0, 0.0, 0.0, 2.0];
        let c_im = [0.0, 0.0, 0.0, 0.0];
        let mut layer = GajeNeuromorphicLayer::new(10, 5, 0.5, 0.9);
        layer.packed_weights.fill(0);
        layer.set_weight(5, 0, GajeWeight2Bit::State11 as u8);

        // Sin resistencia
        layer.integrate_batch_lagrangian(0, c_r, c_im, 1.0, 0.0);
        assert!(layer.membrane_potentials_real[5] > 1.9); // 2.0 - epsilon

        // Con resistencia que bloquea
        layer.reset_potentials();
        layer.integrate_batch_lagrangian(0, c_r, c_im, 1.0, 2.1);
        assert_eq!(layer.membrane_potentials_real[5], 0.0);
    }
}
