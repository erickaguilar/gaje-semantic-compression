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

                // Modulación Relacional Refinada (No Lineal)
                // En lugar de un multiplicador global, aplicamos una función que modula 
                // la intensidad de la señal basándose en la "certeza" de la topología.
                // best_state 0,1: Tendencia a la inhibición (estabilización)
                // best_state 2,3: Tendencia a la excitación (procesamiento activo)
                
                let base_modulation = match best_state {
                    0 => -0.15, // Inhibición fuerte
                    1 => -0.05, // Inhibición leve
                    2 => 0.05,  // Excitación leve
                    3 => 0.15,  // Excitación fuerte
                    _ => 0.0,
                };

                // Aplicar modulación con suavizado sigmoidal para evitar explosión de gradientes
                let confidence_factor = alpha * max_p;
                let final_bias = base_modulation * confidence_factor;

                for val in hidden.iter_mut() {
                    // Usamos una función de transferencia suave para aplicar el bias
                    // Esto permite que el grafo "guíe" la señal sin destruirla
                    let current = *val;
                    if final_bias > 0.0 {
                        // Excitación: aumenta los valores positivos, refuerza la señal
                        *val = current + (current.abs() * final_bias);
                    } else {
                        // Inhibición: atenúa la señal
                        *val = current * (1.0 + final_bias);
                    }
                }
            }
        }
    }
}
