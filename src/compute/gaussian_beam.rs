//! # 🔦 Óptica Paraxial Semántica y Haces Gaussianos (GAJE Helix)
//!
//! Modela el flujo de información a través de capas profundas como un haz óptico coherente
//! (Modo Fundamental $\text{TEM}_{00}$ de haz gaussiano).
//! Implementa la corrección del desfase anómalo de Gouy $\zeta(z) = \arctan(z / z_R)$,
//! el ensanchamiento del haz $w(z)$, la curvatura del frente de onda $R(z)$
//! y el criterio de estabilidad de resonador Fabry-Pérot $0 \le g_1 g_2 \le 1$.

#[cfg(feature = "python")]
use pyo3::prelude::*;

use std::f32::consts::PI;

/// Representa una cavidad de propagación paraxial de haz gaussiano.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianBeamCavity {
    /// Cintura mínima del haz semántico en el foco $z=0$ (radio de cuello de botella K-WTA).
    pub w0: f32,
    /// Longitud de onda semántica media de las activaciones.
    pub lambda: f32,
    /// Rango de Rayleigh: $z_R = \frac{\pi w_0^2}{\lambda}$.
    pub z_rayleigh: f32,
}

#[cfg_attr(feature = "python", pymethods)]
impl GaussianBeamCavity {
    #[cfg(feature = "python")]
    #[new]
    pub fn py_new(w0: f32, lambda: f32) -> Self {
        Self::new(w0, lambda)
    }

    #[cfg(feature = "python")]
    #[pyo3(name = "beam_width")]
    pub fn py_beam_width(&self, z: f32) -> f32 {
        self.beam_width(z)
    }

    #[cfg(feature = "python")]
    #[pyo3(name = "radius_of_curvature")]
    pub fn py_radius_of_curvature(&self, z: f32) -> f32 {
        self.radius_of_curvature(z)
    }

    #[cfg(feature = "python")]
    #[pyo3(name = "gouy_phase")]
    pub fn py_gouy_phase(&self, z: f32) -> f32 {
        self.gouy_phase(z)
    }

    #[cfg(feature = "python")]
    #[pyo3(name = "is_cavity_stable")]
    pub fn py_is_cavity_stable(&self, l_cav: f32, r1: f32, r2: f32) -> bool {
        self.is_cavity_stable(l_cav, r1, r2)
    }
}

impl GaussianBeamCavity {
    /// Inicializa la cavidad a partir del radio de cintura $w_0$ y longitud de onda $\lambda$.
    pub fn new(w0: f32, lambda: f32) -> Self {
        let z_rayleigh = (PI * w0 * w0) / (lambda.abs() + 1e-8);
        Self {
            w0,
            lambda,
            z_rayleigh,
        }
    }

    /// Calcula el semiancho transversal del haz $w(z) = w_0 \sqrt{1 + (z / z_R)^2}$ en la capa $z$.
    #[inline]
    pub fn beam_width(&self, z: f32) -> f32 {
        let ratio = z / self.z_rayleigh;
        self.w0 * (1.0 + ratio * ratio).sqrt()
    }

    /// Calcula el radio de curvatura de la superficie de fase $R(z) = z [1 + (z_R / z)^2]$.
    #[inline]
    pub fn radius_of_curvature(&self, z: f32) -> f32 {
        if z.abs() < 1e-6 {
            return f32::INFINITY; // Frente de onda plano en la cintura z = 0
        }
        let ratio = self.z_rayleigh / z;
        z * (1.0 + ratio * ratio)
    }

    /// Calcula el desfase de Gouy $\zeta(z) = \arctan(z / z_R)$ en radianes.
    /// Al cruzar la cintura de $-\infty$ a $+\infty$, acumula exactamente $\pi$ radianes ($180^\circ$).
    #[inline]
    pub fn gouy_phase(&self, z: f32) -> f32 {
        (z / self.z_rayleigh).atan()
    }

    /// Verifica la condición de estabilidad de Sylvester para cavidades ópticas / resonadores en memoria:
    /// $$0 \le g_1 g_2 \le 1, \quad \text{donde } g_i = 1 - \frac{L_{\text{cav}}}{R_i}$$
    pub fn is_cavity_stable(&self, l_cav: f32, r1: f32, r2: f32) -> bool {
        let g1 = if r1.is_finite() && r1.abs() > 1e-6 {
            1.0 - l_cav / r1
        } else {
            1.0 // Espejo plano
        };

        let g2 = if r2.is_finite() && r2.abs() > 1e-6 {
            1.0 - l_cav / r2
        } else {
            1.0 // Espejo plano
        };

        let g_prod = g1 * g2;
        (0.0..=1.0).contains(&g_prod)
    }

    /// Aplica compensación conforme de curvatura y fase de Gouy sobre el fasor $\psi = \mathbf{u} + i\mathbf{v}$:
    /// Modula el fasor por la contra-rotación unitaria $e^{-i \phi_{\text{corr}}}$, evitando la inversión
    /// de polaridad espuria $e^{i\pi} = -1$ al cruzar cuellos de botella de compresión.
    pub fn compensate_phase(&self, z: f32, r2: f32, u: &mut [f32], v: &mut [f32]) {
        let k = 2.0 * PI / (self.lambda.abs() + 1e-8);
        let r_curv = self.radius_of_curvature(z);
        let zeta = self.gouy_phase(z);

        // Fase paraxial total: phi = -zeta + (k * r^2) / (2 * R)
        let phase_corr = if r_curv.is_finite() {
            -zeta + (k * r2) / (2.0 * r_curv)
        } else {
            -zeta
        };

        let (sin_p, cos_p) = phase_corr.sin_cos();

        // Rotación unitaria exacta en C: (u + iv) * (cos + i sin)
        let len = u.len().min(v.len());
        for i in 0..len {
            let u_orig = u[i];
            let v_orig = v[i];
            u[i] = u_orig * cos_p - v_orig * sin_p;
            v[i] = u_orig * sin_p + v_orig * cos_p;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_beam_properties() {
        let cavity = GaussianBeamCavity::new(1.0, 1.0);
        // z_rayleigh = pi * 1^2 / 1 = pi ~ 3.14159
        assert!((cavity.z_rayleigh - PI).abs() < 1e-4);

        // En z=0: ancho = w0, curvatura = infinito, gouy = 0
        assert_eq!(cavity.beam_width(0.0), 1.0);
        assert!(cavity.radius_of_curvature(0.0).is_infinite());
        assert_eq!(cavity.gouy_phase(0.0), 0.0);

        // En z = z_R: ancho = w0 * sqrt(2), gouy = pi / 4
        let w_zr = cavity.beam_width(cavity.z_rayleigh);
        assert!((w_zr - 2.0f32.sqrt()).abs() < 1e-4);

        let zeta_zr = cavity.gouy_phase(cavity.z_rayleigh);
        assert!((zeta_zr - PI / 4.0).abs() < 1e-4);

        // En z -> infinito: gouy -> pi / 2
        let zeta_inf = cavity.gouy_phase(1e6);
        assert!((zeta_inf - PI / 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_phase_compensation_preserves_norm() {
        let cavity = GaussianBeamCavity::new(1.0, 1.0);
        let mut u = vec![0.6, 0.8];
        let mut v = vec![0.8, -0.6];

        let initial_norm_sq = u[0] * u[0] + v[0] * v[0];

        cavity.compensate_phase(2.0, 0.5, &mut u, &mut v);

        let final_norm_sq = u[0] * u[0] + v[0] * v[0];
        assert!(
            (initial_norm_sq - final_norm_sq).abs() < 1e-5,
            "La rotación unitaria debe preservar exactamente la norma euclidiana"
        );
    }

    #[test]
    fn test_cavity_stability() {
        let cavity = GaussianBeamCavity::new(1.0, 1.0);
        // Espejos simétricos con R1 = R2 = 10.0, longitud L = 5.0
        // g1 = 1 - 5/10 = 0.5, g2 = 0.5 -> g1 * g2 = 0.25 (Estable: 0 <= 0.25 <= 1)
        assert!(cavity.is_cavity_stable(5.0, 10.0, 10.0));

        // Cavidad inestable: L = 25.0 con R = 10.0 -> g1 = 1 - 2.5 = -1.5, g1*g2 = 2.25 > 1
        assert!(!cavity.is_cavity_stable(25.0, 10.0, 10.0));
    }
}
