//! # ⚛️ Dinámica de Schrödinger y Evolución Simpléctica (GAJE Helix)
//!
//! Implementa la evolución unitaria de una función de onda semántica $\psi(\mathbf{x}, \tau) = \mathbf{u} + i\mathbf{v}$
//! gobernada por la ecuación de Schrödinger en el espacio complejo $\mathbb{C}^D$.
//! Utiliza un integrador simpléctico Leapfrog / Störmer-Verlet que conserva exactamente
//! la forma simpléctica $d\mathbf{u} \wedge d\mathbf{v}$ y el volumen del espacio de fases,
//! erradicando la disipación de gradientes y el colapso de fase en capas profundas.

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Representa el estado fasorial complejo $\psi = \mathbf{u} + i\mathbf{v}$ en $\mathbb{C}^D$.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Debug, PartialEq)]
pub struct FasorialState {
    /// Componente real $\mathbf{u}$ (coordenada de posición canónica en el espacio latente).
    pub u: Vec<f32>,
    /// Componente imaginaria $\mathbf{v}$ (momento conjugado canónico en el espacio latente).
    pub v: Vec<f32>,
}

#[cfg_attr(feature = "python", pymethods)]
impl FasorialState {
    #[cfg(feature = "python")]
    #[new]
    pub fn py_new(dim: usize) -> Self {
        Self::new(dim)
    }

    #[cfg(feature = "python")]
    #[getter]
    pub fn get_u(&self) -> Vec<f32> {
        self.u.clone()
    }

    #[cfg(feature = "python")]
    #[getter]
    pub fn get_v(&self) -> Vec<f32> {
        self.v.clone()
    }

    #[cfg(feature = "python")]
    #[pyo3(name = "born_probabilities")]
    pub fn py_born_probabilities(&self) -> Vec<f32> {
        self.born_probabilities()
    }
}

impl FasorialState {
    /// Crea un nuevo estado fasorial con la dimensión dada inicializado en cero.
    pub fn new(dim: usize) -> Self {
        Self {
            u: vec![0.0; dim],
            v: vec![0.0; dim],
        }
    }

    /// Inicializa el estado fasorial a partir de vectores reales e imaginarios dados.
    pub fn from_components(u: Vec<f32>, v: Vec<f32>) -> Result<Self, String> {
        if u.len() != v.len() {
            return Err(format!(
                "Dimensión incompatible: u tiene {} y v tiene {}",
                u.len(),
                v.len()
            ));
        }
        Ok(Self { u, v })
    }

    /// Dimensión del espacio de Hilbert complejo $\mathbb{C}^D$.
    #[inline]
    pub fn dim(&self) -> usize {
        self.u.len()
    }

    /// Paso simpléctico Leapfrog (Störmer-Verlet):
    /// Conserva la 2-forma simpléctica $d\mathbf{u}_{l+1} \wedge d\mathbf{v}_{l+1} = d\mathbf{u}_l \wedge d\mathbf{v}_l$.
    ///
    /// - `delta_tau`: Paso de profundidad de capa (típicamente $1.0 / L$).
    /// - `kinetic_grad`: Derivada del operador cinético $\hat{T}'(\mathbf{v})$ (frecuencia / difusión no local).
    /// - `potential_grad`: Derivada del operador potencial $\hat{V}'(\mathbf{u})$ (pozo atractor FFN / paisaje semántico).
    pub fn symplectic_step<F, G>(
        &mut self,
        delta_tau: f32,
        kinetic_grad: F,
        potential_grad: G,
    ) where
        F: Fn(&[f32], &mut [f32]),
        G: Fn(&[f32], &mut [f32]),
    {
        let dim = self.u.len();
        let mut t_grad = vec![0.0f32; dim];
        let mut v_grad = vec![0.0f32; dim];

        let half_dt = 0.5f32 * delta_tau;

        // 1. Medio paso cinético para posición u:
        //    u_{l + 1/2} = u_l + (dt / 2) * T'(v_l)
        kinetic_grad(&self.v, &mut t_grad);
        for i in 0..dim {
            self.u[i] += half_dt * t_grad[i];
        }

        // 2. Paso completo de potencial para momento v:
        //    v_{l + 1} = v_l - dt * V'(u_{l + 1/2})
        potential_grad(&self.u, &mut v_grad);
        for i in 0..dim {
            self.v[i] -= delta_tau * v_grad[i];
        }

        // 3. Medio paso cinético final para posición u:
        //    u_{l + 1} = u_{l + 1/2} + (dt / 2) * T'(v_{l + 1})
        kinetic_grad(&self.v, &mut t_grad);
        for i in 0..dim {
            self.u[i] += half_dt * t_grad[i];
        }
    }

    /// Calcula la norma del estado $\langle \psi \mid \psi \rangle = \sum (u_i^2 + v_i^2)$.
    pub fn norm_squared(&self) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..self.u.len() {
            sum += self.u[i] * self.u[i] + self.v[i] * self.v[i];
        }
        sum
    }

    /// Normaliza unitariamente la función de onda para proyectar sobre la esfera de Hilbert ($\langle \psi \mid \psi \rangle = 1$).
    pub fn normalize_unitary(&mut self) -> f32 {
        let norm_sq = self.norm_squared();
        if norm_sq > 1e-12 {
            let inv_norm = 1.0f32 / norm_sq.sqrt();
            for i in 0..self.u.len() {
                self.u[i] *= inv_norm;
                self.v[i] *= inv_norm;
            }
            norm_sq.sqrt()
        } else {
            0.0
        }
    }

    /// Calcula la distribución de probabilidad objetiva según la **Regla de Born**:
    /// $$P(i) = |\psi_i|^2 = u_i^2 + v_i^2$$
    pub fn born_probabilities(&self) -> Vec<f32> {
        let mut probs = Vec::with_capacity(self.u.len());
        let mut sum = 0.0f32;

        for i in 0..self.u.len() {
            let p = self.u[i] * self.u[i] + self.v[i] * self.v[i];
            probs.push(p);
            sum += p;
        }

        if sum > 1e-12 {
            let inv_sum = 1.0f32 / sum;
            for p in &mut probs {
                *p *= inv_sum;
            }
        }
        probs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fasorial_state_born_rule() {
        let u = vec![1.0, 0.0, 2.0];
        let v = vec![0.0, 1.0, 0.0];
        let state = FasorialState::from_components(u, v).unwrap();

        let probs = state.born_probabilities();
        assert_eq!(probs.len(), 3);
        // |psi_0|^2 = 1.0, |psi_1|^2 = 1.0, |psi_2|^2 = 4.0 -> Total = 6.0
        assert!((probs[0] - 1.0 / 6.0).abs() < 1e-5);
        assert!((probs[1] - 1.0 / 6.0).abs() < 1e-5);
        assert!((probs[2] - 4.0 / 6.0).abs() < 1e-5);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_symplectic_energy_conservation() {
        // Oscilador armónico simple: T(v) = 0.5 * v^2 -> T'(v) = v
        //                           V(u) = 0.5 * u^2 -> V'(u) = u
        let mut state = FasorialState::from_components(vec![1.0], vec![0.0]).unwrap();
        let dt = 0.1f32;

        let kinetic_grad = |v: &[f32], grad: &mut [f32]| grad[0] = v[0];
        let potential_grad = |u: &[f32], grad: &mut [f32]| grad[0] = u[0];

        // Ejecutar 100 pasos simplécticos
        for _ in 0..100 {
            state.symplectic_step(dt, kinetic_grad, potential_grad);
        }

        // La energía de un oscilador H = 0.5 * (u^2 + v^2) bajo integrador simpléctico
        // oscila alrededor del valor inicial exacto sin deriva secular exponencial
        let final_energy = 0.5 * (state.u[0] * state.u[0] + state.v[0] * state.v[0]);
        assert!(
            (final_energy - 0.5).abs() < 0.02,
            "Energía desviada: {}",
            final_energy
        );
    }
}
