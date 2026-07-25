//! # 🪐 Motor Lagrangiano: Optimización por Mínima Acción
//!
//! Este módulo implementa la física del Protocolo GAJE basada en las ecuaciones
//! de Euler-Lagrange y el Gradiente Natural Proximal.
//!
//! ## Principios:
//! * **L = T - V**: El Lagrangiano es la diferencia entre la energía cinética
//!   (precondicionada por la métrica de Fisher) y la energía potencial (Loss).
//! * **Métrica Heterogénea**: El espacio no es euclideo; la "masa" (inercia)
//!   varía según la importancia semántica de cada peso (anclas vs genoma).
//! * **Proximal Path**: La trayectoria minimiza la acción bajo restricciones
//!   de cuantización (ε-nets) y escasez (K-WTA).

#[derive(Clone, Debug)]
pub struct LagrangianEngine {
    pub mass_base: f32, // Inercia base para el genoma de 2 bits
    pub gamma: f32,     // Factor de conformalidad (stiffening) para las anclas
}

impl LagrangianEngine {
    pub fn new(mass_base: f32) -> Self {
        Self {
            mass_base,
            gamma: 8.0, // Default stiffening: factor 8x (proporcional a 16bit/2bit)
        }
    }

    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    /// Calcula el Lagrangiano para un estado neuronal dado.
    /// * `velocity`: Velocidad de disparo (gradiente natural).
    /// * `resistance`: Potencial semántico (Loss).
    /// * `is_anchor`: Si el parámetro es una ancla de alta precisión.
    pub fn calculate_lagrangian(&self, velocity: f32, resistance: f32, is_anchor: bool) -> f32 {
        let m = if is_anchor {
            self.mass_base * self.gamma
        } else {
            self.mass_base
        };
        let kinetic = 0.5 * m * velocity.powi(2);
        let potential = resistance;
        kinetic - potential
    }

    /// Calcula el paso de actualización (delta) basado en el precondicionador de Fisher.
    /// Actualización PNG: delta = -eta * M^-1 * grad
    pub fn calculate_step(&self, grad: f32, fisher_val: f32, is_anchor: bool, lr: f32) -> f32 {
        let m_conformal = if is_anchor { self.gamma } else { 1.0 };
        let metric = (fisher_val + 1e-6) * m_conformal;
        (lr / metric) * grad
    }

    /// Calcula la aceleración geodésica (fuerza/masa) respetando la métrica heterogénea.
    pub fn geodesic_acceleration(&self, force: f32, is_anchor: bool) -> f32 {
        let m = if is_anchor {
            self.mass_base * self.gamma
        } else {
            self.mass_base
        };
        force / m
    }

    /// Calcula el retraso temporal en la Rueda de Tiempo basado en la Acción.
    pub fn calculate_timing_delay(&self, potential_energy: f32) -> f32 {
        if potential_energy <= 1e-6 {
            0.0
        } else {
            // Retraso no lineal: los estados de alta energía (ruido) se hunden en el tiempo
            potential_energy.ln_1p() * 1.5
        }
    }
}

/// # Símbolos de Christoffel (Geometría del Toroide)
///
/// Define la conexión en la variedad semántica.
pub fn christoffel_connection(phase: f32, curvature: f32) -> f32 {
    phase.sin() * curvature
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_png_update_step() {
        let engine = LagrangianEngine::new(1.0);
        let grad = 0.5;
        let fisher_val = 0.1;
        let lr = 0.01;

        // Genomic (2-bit) step
        let step_genomic = engine.calculate_step(grad, fisher_val, false, lr);

        // Anchor (F16) step (should be 8x smaller due to gamma=8.0)
        let step_anchor = engine.calculate_step(grad, fisher_val, true, lr);

        assert!(step_genomic > step_anchor);
        assert!((step_genomic / step_anchor - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_geodesic_acceleration() {
        let engine = LagrangianEngine::new(2.0).with_gamma(10.0);
        let force = 10.0;

        // Genomic: a = F / m = 10 / 2 = 5
        let accel_genomic = engine.geodesic_acceleration(force, false);
        assert_eq!(accel_genomic, 5.0);

        // Anchor: a = F / (m * gamma) = 10 / (2 * 10) = 0.5
        let accel_anchor = engine.geodesic_acceleration(force, true);
        assert_eq!(accel_anchor, 0.5);
    }
}
