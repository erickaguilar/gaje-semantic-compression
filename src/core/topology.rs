use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CentroidGraph {
    pub model_source: String,
    pub states: usize,
    pub topology: HashMap<String, Vec<Vec<f32>>>,
}

impl CentroidGraph {
    pub fn get_transition_matrix(&self, layer_idx: usize) -> Option<&Vec<Vec<f32>>> {
        self.topology.get(&layer_idx.to_string())
    }

    pub fn get_modulation_factors(&self, layer_idx: usize, current_state: usize, alpha: f32) -> [f32; 4] {
        let mut factors = [1.0f32; 4];
        if let Some(matrix) = self.get_transition_matrix(layer_idx) {
            if current_state < matrix.len() {
                let probs = &matrix[current_state];
                for (s, &p) in probs.iter().enumerate().take(4) {
                    // Modulación por estado: 
                    // 0, 1 -> Inhibición (0.9 a 1.0)
                    // 2, 3 -> Excitación (1.0 a 1.15)
                    let base = if s < 2 { -0.1 } else { 0.15 };
                    factors[s] = 1.0 + (base * p * alpha).clamp(-0.2, 0.3);
                }
            }
        }
        factors
    }

    pub fn apply_relational_bias(&self, layer_idx: usize, current_state: usize, hidden: &mut [f32], alpha: f32) {
        if let Some(matrix) = self.get_transition_matrix(layer_idx) {
            if current_state < matrix.len() {
                let probs = &matrix[current_state];
                
                // Encontrar el estado de destino más probable y su confianza (probabilidad)
                let mut best_state = 0;
                let mut max_p = 0.0;
                for (s, &p) in probs.iter().enumerate() {
                    if p > max_p {
                        max_p = p;
                        best_state = s;
                    }
                }

                // Modulación Relacional Refinada (No Lineal) con Estabilización
                let base_modulation = match best_state {
                    0 => -0.05, // Inhibición leve (reducido de -0.15 para estabilidad)
                    1 => -0.02, 
                    2 => 0.02,  
                    3 => 0.05,  // Excitación leve (reducido de 0.15)
                    _ => 0.0,
                };

                let confidence_factor = alpha * max_p;
                let final_bias = base_modulation * confidence_factor;

                // Calcular norma para evitar explosión
                let mut sum_sq = 0.0f32;
                for &val in hidden.iter() { sum_sq += val * val; }
                let norm = (sum_sq / hidden.len() as f32 + 1e-6).sqrt();

                for val in hidden.iter_mut() {
                    let current = *val;
                    if final_bias > 0.0 {
                        // Excitación controlada
                        *val = current + (current.abs() * final_bias).min(norm * 0.1);
                    } else {
                        // Inhibición controlada
                        *val = current * (1.0 + final_bias).max(0.9);
                    }
                    
                    // Clamping final de seguridad
                    if val.is_nan() { *val = 0.0; }
                    else if val.is_infinite() { *val = val.signum() * 10.0; }
                }
            }
        }
    }
}
