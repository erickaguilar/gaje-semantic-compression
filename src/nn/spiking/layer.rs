#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Estructura de Capa Neuromórfica Industrial (SoA - Structure of Arrays).
/// Optimizada para localidad de datos, caché y SIMD.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Debug)]
pub struct GajeNeuromorphicLayer {
    pub membrane_potentials: Vec<f32>,
    pub thresholds: Vec<f32>,
    pub decays: Vec<f32>,
    /// Pesos empaquetados: 4 pesos de 2-bits por byte.
    pub packed_weights: Vec<u8>, 
    pub num_neurons: usize,
    pub weights_per_neuron: usize,
    pub k_wta: usize,
    pub rms_ema: f32,
}

#[cfg_attr(feature = "python", pymethods)]
impl GajeNeuromorphicLayer {
    #[cfg(feature = "python")]
    #[new]
    pub fn new_py(num_neurons: usize, weights_per_neuron: usize, threshold: f32, decay: f32) -> Self {
        Self::new(num_neurons, weights_per_neuron, threshold, decay)
    }

    #[cfg(feature = "python")]
    #[getter]
    pub fn get_membrane_potentials(&self) -> Vec<f32> { self.membrane_potentials.clone() }

    #[cfg(feature = "python")]
    #[setter]
    pub fn set_membrane_potentials(&mut self, values: Vec<f32>) -> PyResult<()> {
        if values.len() != self.num_neurons { return Err(pyo3::exceptions::PyValueError::new_err("Size mismatch")); }
        self.membrane_potentials = values; Ok(())
    }

    #[cfg(feature = "python")]
    #[getter]
    pub fn get_num_neurons(&self) -> usize { self.num_neurons }

    #[cfg(feature = "python")]
    #[getter]
    pub fn get_weights_per_neuron(&self) -> usize { self.weights_per_neuron }

    #[cfg(feature = "python")]
    #[getter]
    pub fn get_k_wta(&self) -> usize { self.k_wta }

    #[cfg(feature = "python")]
    #[setter]
    pub fn set_k_wta(&mut self, val: usize) { self.k_wta = val; }

    #[cfg(feature = "python")]
    #[getter]
    pub fn get_rms_ema(&self) -> f32 { self.rms_ema }

    pub fn reset_potentials(&mut self) {
        self.membrane_potentials.fill(0.0);
    }

    #[cfg(feature = "python")]
    #[getter]
    pub fn get_packed_weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        Ok(pyo3::types::PyBytes::new(py, &self.packed_weights))
    }

    #[cfg(feature = "python")]
    pub fn load_packed_weights(&mut self, data: Vec<u8>) -> PyResult<()> {
        if data.len() != self.packed_weights.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Weight size mismatch: expected {}, got {}", self.packed_weights.len(), data.len())));
        }
        self.packed_weights = data; Ok(())
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
            membrane_potentials: vec![0.0; num_neurons],
            thresholds: vec![threshold; num_neurons],
            decays: vec![decay; num_neurons],
            packed_weights,
            num_neurons,
            weights_per_neuron,
            k_wta: (num_neurons / 10).max(1),
            rms_ema: 1.0,
        }
    }

    pub fn integrate_batch(&mut self, input_index: usize, centroides: [f32; 4], intensity: f32) {
        let row_size = (self.num_neurons + 3) / 4;
        let start_byte = input_index * row_size;
        let homeostatic_bias = 0.01;

        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            let n = self.num_neurons;
            let potentials_ptr = self.membrane_potentials.as_mut_ptr();
            let weights_ptr = self.packed_weights.as_ptr().add(start_byte);
            let intensity_v = vdupq_n_f32(intensity);
            let bias_v = vdupq_n_f32(homeostatic_bias);
            let table_v = vld1q_u8(centroides.as_ptr() as *const u8);
            let b0123 = vcreate_u8(0x0302010003020100);
            let offsets = vcombine_u8(b0123, b0123);
            let masks = vld1q_u8([0x03, 0x03, 0x03, 0x03, 0x0C, 0x0C, 0x0C, 0x0C, 0x30, 0x30, 0x30, 0x30, 0xC0, 0xC0, 0xC0, 0xC0].as_ptr());
            let shifts = vld1q_s8([2, 2, 2, 2, 0, 0, 0, 0, -2, -2, -2, -2, -4, -4, -4, -4].as_ptr());
            let mut i = 0;
            while i + 4 <= n {
                let byte_idx = i / 4;
                let b = *weights_ptr.add(byte_idx);
                let v_b = vdupq_n_u8(b);
                let indices_base = vandq_u8(v_b, masks);
                let indices_scaled = vshlq_u8(indices_base, shifts);
                let indices = vaddq_u8(indices_scaled, offsets);
                let lookup_res = vqtbl1q_u8(table_v, indices);
                let c_v: float32x4_t = std::mem::transmute(lookup_res);
                let mut p_v = vld1q_f32(potentials_ptr.add(i));
                p_v = vfmaq_f32(p_v, c_v, intensity_v);
                p_v = vaddq_f32(p_v, bias_v);
                vst1q_f32(potentials_ptr.add(i), p_v);
                i += 4;
            }
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
                self.membrane_potentials[i] += (centroides[weight_bits as usize] * intensity) + homeostatic_bias;
            }
        }
    }

    pub fn check_spikes(&mut self) -> Vec<(usize, f32, u8)> {
        let mut spikes = Vec::new();
        let n = self.num_neurons;
        let mut sum_sq = 0.0f32;
        for &p in &self.membrane_potentials { if p > 0.0 { sum_sq += p * p; } }
        let rms = (sum_sq / n as f32 + 1e-6).sqrt();
        let alpha = 0.15;
        self.rms_ema = (1.0 - alpha) * self.rms_ema + alpha * rms;
        for i in 0..n {
            let potential = self.membrane_potentials[i];
            let threshold = self.thresholds[i];
            if potential >= threshold {
                let norm_factor = if self.rms_ema > 1.0 { 1.0 / self.rms_ema } else { 1.0 };
                let intensity = (1.0 + (potential - threshold) / threshold) * norm_factor;
                let excess_ratio = (potential - threshold) / threshold;
                let phase = if excess_ratio >= 1.0 { 0 } else { 15 - (excess_ratio * 15.0) as u8 };
                self.membrane_potentials[i] = 0.0;
                spikes.push((i, intensity, phase));
            } else if self.membrane_potentials[i] > 0.0 {
                self.membrane_potentials[i] *= self.decays[i];
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
        
        // Verificación de seguridad para evitar pánicos por OOB
        if start_byte + row_size > self.packed_weights.len() {
            return;
        }

        for (i, &delta) in deltas.iter().enumerate() {
            if i >= self.num_neurons || delta.abs() < 1e-5 { continue; }
            if rng.gen::<f32>() > learning_rate { continue; }
            let byte_idx = i / 4;
            let bit_shift = (i % 4) * 2;
            let current_byte = self.packed_weights[start_byte + byte_idx];
            let current_weight = (current_byte >> bit_shift) & 0x03;
            let mut new_weight = current_weight;
            if delta > 0.0 { if current_weight < 3 { new_weight += 1; } }
            else if current_weight > 0 { new_weight -= 1; }
            if new_weight != current_weight {
                self.packed_weights[start_byte + byte_idx] &= !(0x03 << bit_shift);
                self.packed_weights[start_byte + byte_idx] |= new_weight << bit_shift;
            }
        }
    }

    pub fn apply_homeostasis(&mut self, target_potential: f32) {
        for p in self.membrane_potentials.iter_mut() { if *p > target_potential { *p *= 0.95; } }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::spiking::neuron::GajeWeight2Bit;

    #[test]
    fn test_refine_step() {
        let mut layer = GajeNeuromorphicLayer::new(4, 1, 1.0, 0.9);
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
        let centroides = [-1.0, -0.2, 0.2, 1.0];
        let mut layer = GajeNeuromorphicLayer::new(10, 5, 0.5, 0.9);
        layer.set_weight(2, 0, GajeWeight2Bit::State11 as u8);
        layer.integrate_batch(0, centroides, 2.0);
        assert_eq!(layer.membrane_potentials[2], 2.01);
        let spikes = layer.check_spikes();
        assert_eq!(spikes.len(), 1);
        assert_eq!(spikes[0].0, 2);
        assert!(spikes[0].1 > 4.0);
        assert_eq!(spikes[0].2, 0);
        assert_eq!(layer.membrane_potentials[2], 0.0);
    }
}
