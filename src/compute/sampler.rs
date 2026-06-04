#[cfg(feature = "python")]
use pyo3::prelude::*;

use rand::Rng;
use std::cmp::Ordering;
use crate::compute::lagrangian::{LagrangianEngine};

/// # 🧬 SintergicSampler: El Puente Transductor (Lattice -> Humano)
///
/// Implementa la Teoría de Transducción Sintergial de Jacobo Grinberg.
/// Este componente actúa como la "Aduana Dimensional" que traduce el espacio
/// de fase cuántico (2 bits) en texto coherente mediante la Latencia de Arribo.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct SintergicSampler {
    pub engine: LagrangianEngine,
    pub sintergy_threshold: f32, // Umbral de "Alta Sintergia" (Tick 0.1)
    pub last_phase: f32,
    pub last_velocity: f32,
}

impl SintergicSampler {
    pub fn new_core(mass: f32, sintergy_threshold: f32) -> Self {
        Self {
            engine: LagrangianEngine::new(mass),
            sintergy_threshold,
            last_phase: 0.0,
            last_velocity: 1.0,
        }
    }

    /// Realiza la transducción dimensional filtrando el ruido del "tiempo" (latencia tardía).
    pub fn sample_sintergic_core(
        &mut self,
        logits: Vec<f32>,
        temperature: f32,
        k_wta_factor: f32,
    ) -> Result<usize, String> {
        if logits.is_empty() {
            return Ok(0);
        }

        let n_tokens = logits.len();
        
        // 1. Transformar Logits en "Tiempo de Arribo" (Moneda Dimensional)
        // Cuanto más alto el logit, más cerca de la "Latencia Cero" (Inmovilidad/Luz).
        let mut temporal_arrivals = Vec::with_capacity(n_tokens);
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        for (i, &energy) in logits.iter().enumerate() {
            // Aplicamos una transformación exponencial para acentuar los picos de sintergia
            let logit_norm = (energy - max_logit) / temperature.max(1e-6);
            let resonance = logit_norm.exp();
            
            // Latencia: Inversa de la resonancia (Luz = 0.0, Caos = 1.0)
            let latency = 1.0 / (1.0 + resonance);
            temporal_arrivals.push((i, latency, resonance));
        }

        // 2. Colapso Sintergico (K-WTA por Latencia)
        // Solo permitimos que los impulsos que están en sintonía con las Anclas (F16) pasen.
        // Los impulsos que experimentan "tiempo" (latencia > threshold) son aniquilados.
        
        // Ordenamos por latencia (los más rápidos primero)
        temporal_arrivals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        
        // Filtrado K-WTA Dinámico: Tomamos solo los que están cerca del "Presente Puro"
        let min_latency = temporal_arrivals[0].1;
        let survival_threshold = min_latency + (self.sintergy_threshold * k_wta_factor);
        
        let mut survivors = Vec::new();
        for &candidate in &temporal_arrivals {
            if candidate.1 <= survival_threshold {
                survivors.push(candidate);
            } else {
                break; // El resto están disueltos en el caos temporal
            }
        }

        // 3. Muestreo de la Dimensión Humana
        // Si no hay supervivientes (colapso total), tomamos el más cercano a la luz.
        if survivors.is_empty() {
            return Ok(temporal_arrivals[0].0);
        }

        let mut rng = rand::thread_rng();
        let sum_resonance: f32 = survivors.iter().map(|(_, _, r)| r).sum();
        let r_val: f32 = rng.gen::<f32>() * sum_resonance;

        let mut current_sum = 0.0;
        for &(id, latency, resonance) in &survivors {
            current_sum += resonance;
            if r_val <= current_sum {
                // Actualizar estado del transductor
                self.last_velocity = 1.0 - latency;
                self.last_phase = (id % 16) as f32; // Aproximación de fase
                return Ok(id);
            }
        }

        Ok(survivors[0].0)
    }

    pub fn reset(&mut self) {
        self.last_phase = 0.0;
        self.last_velocity = 1.0;
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl SintergicSampler {
    #[new]
    #[pyo3(signature = (mass=1.0, sintergy_threshold=0.1))]
    pub fn py_new(mass: f32, sintergy_threshold: f32) -> Self {
        Self::new_core(mass, sintergy_threshold)
    }

    pub fn sample(
        &mut self,
        logits: Vec<f32>,
        temperature: f32,
        k_wta_factor: f32,
    ) -> PyResult<usize> {
        self.sample_sintergic_core(logits, temperature, k_wta_factor).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    pub fn reset_py(&mut self) {
        self.reset();
    }
}

/// # 🪐 Sampler Toroidal (Wrapper de SintergicSampler para Compatibilidad)
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct ToroidalSampler {
    pub sintergic: SintergicSampler,
    pub curvature: f32,
}

impl ToroidalSampler {
    pub fn new_core(mass: f32, curvature: f32) -> Self {
        Self {
            sintergic: SintergicSampler::new_core(mass, 0.1),
            curvature,
        }
    }

    pub fn sample_core(
        &mut self,
        logits: Vec<f32>,
        temperature: f32,
        top_p: f32,
    ) -> Result<usize, String> {
        // Mapeamos top_p a k_wta_factor de forma heurística para el puente transductor
        let k_wta_factor = (1.0 - top_p) * 2.0;
        self.sintergic.sample_sintergic_core(logits, temperature, k_wta_factor)
    }

    pub fn reset(&mut self) {
        self.sintergic.reset();
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
