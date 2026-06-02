#[cfg(feature = "python")]
use pyo3::prelude::*;

use rand::Rng;
use std::cmp::Ordering;
use crate::compute::lagrangian::{LagrangianEngine, christoffel_connection};

/// # 🪐 Sampler Toroidal: Muestreo Consciente de la Fase
///
/// Este sampler implementa el Pilar 1 del Gran Salto, utilizando la topología 
/// toroidal y la física lagrangiana para guiar la generación de tokens.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct ToroidalSampler {
    pub engine: LagrangianEngine,
    pub last_phase: f32,
    pub last_velocity: f32,
    pub curvature: f32,
}

impl ToroidalSampler {
    pub fn new_core(mass: f32, curvature: f32) -> Self {
        Self {
            engine: LagrangianEngine::new(mass),
            last_phase: 0.0,
            last_velocity: 1.0,
            curvature,
        }
    }

    pub fn sample_core(
        &mut self,
        logits: Vec<f32>,
        temperature: f32,
        top_p: f32,
    ) -> Result<usize, String> {
        if logits.is_empty() {
            return Ok(0);
        }

        let n_tokens = logits.len();
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let threshold = (max_logit * 0.7).max(0.01);

        // 1. Calcular conexión de Christoffel para la fase actual
        let connection = christoffel_connection(self.last_phase, self.curvature);
        let expected_phase = (self.last_phase + connection) % 16.0;

        // 2. Evaluar candidatos con frenado Lagrangiano
        let mut candidates = Vec::with_capacity(n_tokens);
        for (i, &energy) in logits.iter().enumerate() {
            if energy >= threshold {
                let excess_ratio = (energy - threshold) / threshold.max(1e-6);
                let current_phase = if excess_ratio >= 1.0 {
                    0.0
                } else {
                    15.0 - (excess_ratio * 15.0)
                };

                let delta_phi = (current_phase - expected_phase).abs();
                let potential = delta_phi * self.curvature;
                let lagrangian = self.engine.calculate_lagrangian(self.last_velocity, potential);
                let adjusted_logit = energy + lagrangian.min(0.0);
                
                candidates.push((i, (adjusted_logit / temperature.max(1e-6)).exp(), current_phase));
            }
        }

        if candidates.is_empty() {
            return crate::compute::math::sample_top_p_core(logits, temperature, top_p);
        }

        let sum_exp: f32 = candidates.iter().map(|(_, p, _)| p).sum();
        if sum_exp <= 0.0 {
             return crate::compute::math::sample_top_p_core(logits, temperature, top_p);
        }

        for c in &mut candidates {
            c.1 /= sum_exp;
        }
        
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        
        let mut cumulative_prob = 0.0;
        let mut cutoff_idx = candidates.len();
        for (i, &(_, p, _)) in candidates.iter().enumerate() {
            cumulative_prob += p;
            if cumulative_prob > top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        candidates.truncate(cutoff_idx);
        
        let final_sum: f32 = candidates.iter().map(|(_, p, _)| p).sum();
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen::<f32>() * final_sum;
        
        let mut current_sum = 0.0;
        for &(id, p, phase) in &candidates {
            current_sum += p;
            if r <= current_sum {
                self.last_velocity = (1.0 - (phase - self.last_phase).abs() / 16.0).max(0.1);
                self.last_phase = phase;
                return Ok(id);
            }
        }

        let (final_id, _, final_phase) = candidates[0];
        self.last_phase = final_phase;
        Ok(final_id)
    }

    pub fn reset(&mut self) {
        self.last_phase = 0.0;
        self.last_velocity = 1.0;
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl ToroidalSampler {
    #[new]
    #[pyo3(signature = (mass=1.0, curvature=0.1))]
    pub fn py_new(mass: f32, curvature: f32) -> Self {
        Self::new_core(mass, curvature)
    }

    pub fn sample(
        &mut self,
        logits: Vec<f32>,
        temperature: f32,
        top_p: f32,
    ) -> PyResult<usize> {
        self.sample_core(logits, temperature, top_p).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    pub fn reset_py(&mut self) {
        self.reset();
    }
}
