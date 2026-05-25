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
                // Encontrar el estado de destino más probable
                let mut best_state = 0;
                let mut max_p = 0.0;
                for (s, &p) in probs.iter().enumerate() {
                    if p > max_p {
                        max_p = p;
                        best_state = s;
                    }
                }

                // Modificar el vector oculto para "empujarlo" hacia el centroide del estado más probable
                // Como no tenemos los centroides globales aquí, usamos el bias como una 
                // modulación de fase: si el estado es alto (2 o 3), aumentamos la amplitud de la señal.
                let multiplier = match best_state {
                    0 => 0.8, // Inhibición
                    1 => 0.9, // Neutro-bajo
                    2 => 1.1, // Excitación-media
                    3 => 1.3, // Excitación-fuerte
                    _ => 1.0,
                };

                for val in hidden.iter_mut() {
                    *val *= multiplier * (1.0 + alpha * max_p);
                }
            }
        }
    }
}
