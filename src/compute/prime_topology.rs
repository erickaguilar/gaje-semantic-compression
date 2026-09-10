//! # 🧬 Topología de Números Primos, Coprimalidad y Resonancia Aperiódica (GAJE Helix)
//!
//! Fundamentación teórica documentada en:
//! `docs/research/PRIME_TOPOLOGY_COPRIMALITY_AND_RESONANCE_FINDINGS.md` y
//! `docs/research/RNS_VECTOR_ARITHMETIC_AND_PRIME_FOUNDATIONS.md`.
//!
//! Rompe las simetrías nocivas de potencias de dos ($2^n$) en espacios cuantizados a 2 bits mediante:
//! 1. **RoPE Coprimo:** Frecuencias angulares basadas en inversos de primos $\theta_k = \frac{2\pi}{p_k}$,
//!    garantizando un ciclo aperiódico $\prod p_k \gg 10^{15}$ tokens por el Teorema Chino del Resto.
//! 2. **Particionado en Campo de Fermat $\mathbb{F}_{257}$:** Hash universal para `.gmem` con varianza mínima.
//! 3. **Secuencias de Halton Primas:** Muestreo cuasi-Monte Carlo para perturbaciones no colapsadas en pesos.

#[cfg(feature = "python")]
use pyo3::prelude::*;

use std::f32::consts::PI;

/// Primeros números primos precalculados para acceso ultrarrápido en inferencia y RoPE.
pub const CANONICAL_PRIMES: &[usize] = &[
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
    97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
    193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293,
    307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
    421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
];

/// Módulo de operaciones topológicas basadas en números primos y campos finitos.
#[cfg_attr(feature = "python", pyclass)]
pub struct PrimeTopology;

#[cfg_attr(feature = "python", pymethods)]
impl PrimeTopology {
    #[cfg(feature = "python")]
    #[staticmethod]
    #[pyo3(name = "get_primes")]
    pub fn py_get_primes(count: usize) -> Vec<usize> {
        Self::first_n_primes(count)
    }

    #[cfg(feature = "python")]
    #[staticmethod]
    #[pyo3(name = "coprime_rope_frequencies")]
    pub fn py_coprime_rope_frequencies(count: usize) -> Vec<f32> {
        Self::generate_coprime_frequencies(count)
    }

    #[cfg(feature = "python")]
    #[staticmethod]
    #[pyo3(name = "fermat_hash_257")]
    pub fn py_fermat_hash_257(vector: Vec<f32>, num_buckets: usize) -> usize {
        Self::fermat_hash_257(&vector, num_buckets)
    }
}

impl PrimeTopology {
    /// Obtiene los primeros $n$ números primos mediante tabla canónica o criba de Eratóstenes.
    pub fn first_n_primes(n: usize) -> Vec<usize> {
        if n <= CANONICAL_PRIMES.len() {
            return CANONICAL_PRIMES[..n].to_vec();
        }

        let mut primes = CANONICAL_PRIMES.to_vec();
        let mut candidate = primes.last().copied().unwrap_or(2) + 2;

        while primes.len() < n {
            let limit = (candidate as f64).sqrt() as usize;
            let mut is_p = true;
            for &p in &primes {
                if p > limit {
                    break;
                }
                if candidate % p == 0 {
                    is_p = false;
                    break;
                }
            }
            if is_p {
                primes.push(candidate);
            }
            candidate += 2;
        }

        primes
    }

    /// Genera las frecuencias angulares RoPE coprimas $\theta_k = \frac{2\pi}{p_k}$.
    /// Al ser todos los periodos $p_k$ primos relativos entre sí, el periodo conjunto del toroide
    /// es el producto acumulado $T = \prod p_k$, eliminando el aliasing armónico en secuencias largas.
    pub fn generate_coprime_frequencies(count: usize) -> Vec<f32> {
        let primes = Self::first_n_primes(count);
        primes
            .into_iter()
            .map(|p| (2.0f32 * PI) / (p as f32))
            .collect()
    }

    /// Particionado proyectivo uniforme sobre el campo finito primo de Fermat $\mathbb{F}_{257}$:
    /// $$h(\mathbf{v}) = \left( \left( \sum_{j=1}^{D} a_j \cdot \lfloor |v_j \cdot 1000| \rfloor + b \right) \bmod 257 \right) \bmod M$$
    /// Evita el agrupamiento patológico y los cubos vacíos observados en particionados $2^n$.
    pub fn fermat_hash_257(vector: &[f32], num_buckets: usize) -> usize {
        if num_buckets == 0 {
            return 0;
        }

        const P_FERMAT: u64 = 257;
        let mut acc: u64 = 17; // Semilla prima no nula en F_257*

        for (j, &val) in vector.iter().enumerate() {
            // Factor multiplicador primo periódico
            let prime_factor = (CANONICAL_PRIMES[j % CANONICAL_PRIMES.len()] as u64) % P_FERMAT;
            let val_scaled = (val.abs() * 1000.0) as u64;
            acc = (acc + (prime_factor * (val_scaled + 1))) % P_FERMAT;
        }

        (acc as usize) % num_buckets
    }
}

/// Generador cuasi-Monte Carlo de Secuencias de Halton multidimensionales
/// con bases primas $p_1, p_2, \dots, p_D$ para muestreo de baja discrepancia en DNI/SPSA.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Debug)]
pub struct HaltonSequence {
    pub bases: Vec<usize>,
    pub index: usize,
}

#[cfg_attr(feature = "python", pymethods)]
impl HaltonSequence {
    #[cfg(feature = "python")]
    #[new]
    pub fn py_new(dim: usize) -> Self {
        Self::new(dim)
    }

    #[cfg(feature = "python")]
    #[pyo3(name = "next_vector")]
    pub fn py_next_vector(&mut self) -> Vec<f32> {
        self.next_vector()
    }
}

impl HaltonSequence {
    /// Crea un generador de secuencias de Halton para un espacio de dimensión $D$,
    /// utilizando los primeros $D$ números primos como bases de inversión radical.
    pub fn new(dim: usize) -> Self {
        let bases = PrimeTopology::first_n_primes(dim);
        Self { bases, index: 1 }
    }

    /// Calcula la función de inversión radical de Van der Corput en base prima $p$:
    /// $$\phi_p(n) = \sum_{j=0}^M b_j p^{-(j+1)}$$
    #[inline]
    pub fn radical_inverse(mut n: usize, base: usize) -> f32 {
        let mut result = 0.0f32;
        let mut f = 1.0f32 / (base as f32);
        let base_f = base as f32;

        while n > 0 {
            let digit = n % base;
            result += (digit as f32) * f;
            f /= base_f;
            n /= base;
        }

        result
    }

    /// Genera el siguiente vector cuasi-aleatorio en $[0, 1)^D$.
    pub fn next_vector(&mut self) -> Vec<f32> {
        let n = self.index;
        self.index += 1;

        self.bases
            .iter()
            .map(|&p| Self::radical_inverse(n, p))
            .collect()
    }

    /// Genera un vector centrado en $[-1.0, +1.0]^D$ para perturbaciones de pesos en STE/SPSA.
    pub fn next_centered_perturbation(&mut self) -> Vec<f32> {
        self.next_vector()
            .into_iter()
            .map(|x| (x * 2.0) - 1.0)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_n_primes() {
        let primes = PrimeTopology::first_n_primes(10);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_coprime_rope_frequencies_aperiodicity() {
        let freqs = PrimeTopology::generate_coprime_frequencies(4);
        assert_eq!(freqs.len(), 4);

        // theta_0 = 2pi / 2 = pi
        // theta_1 = 2pi / 3
        // theta_2 = 2pi / 5
        // theta_3 = 2pi / 7
        assert!((freqs[0] - PI).abs() < 1e-5);
        assert!((freqs[1] - 2.0 * PI / 3.0).abs() < 1e-5);
        assert!((freqs[2] - 2.0 * PI / 5.0).abs() < 1e-5);
        assert!((freqs[3] - 2.0 * PI / 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_fermat_hash_257_distribution() {
        let v1 = vec![0.5, -0.2, 0.9, 0.1];
        let v2 = vec![0.5, -0.2, 0.9, 0.1]; // Idéntico -> mismo cubo
        let v3 = vec![0.1, 0.8, -0.4, 0.3]; // Distinto -> cubo proyectado

        let h1 = PrimeTopology::fermat_hash_257(&v1, 32);
        let h2 = PrimeTopology::fermat_hash_257(&v2, 32);
        let h3 = PrimeTopology::fermat_hash_257(&v3, 32);

        assert_eq!(h1, h2);
        assert!(h1 < 32);
        assert!(h3 < 32);
    }

    #[test]
    fn test_halton_sequence_bounds() {
        let mut halton = HaltonSequence::new(3);
        for _ in 0..50 {
            let vec = halton.next_vector();
            assert_eq!(vec.len(), 3);
            for &val in &vec {
                assert!(val >= 0.0 && val < 1.0);
            }
        }
    }
}
