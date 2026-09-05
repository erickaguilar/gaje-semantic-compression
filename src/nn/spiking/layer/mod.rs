// =============================================================================
// layer — Capa Neuromórfica Industrial (SoA) para GAJE
// =============================================================================
//
// `GajeNeuromorphicLayer` es una capa spiking orientada a datos (Structure of
// Arrays) con potenciales complejos, pesos de 2 bits, anclas de estabilidad y
// un motor lagrangiano de fricción semántica.
//
// Este módulo es un **hub de re-exportación** que preserva la API pública
// histórica (`crate::nn::spiking::layer::*`) mientras la implementación vive en
// submódulos cohesionados:
//
// - [`init`](crate::nn::spiking::layer::init): construcción, reset y anclas.
// - [`integrate`](crate::nn::spiking::layer::integrate): integración de impulsos (batch y lagrangiana).
// - [`spike`](crate::nn::spiking::layer::spike): disparo, refine, homeostasis e inhibición lateral.
// - [`python`](crate::nn::spiking::layer::python): bindings `#[pymethods]` (feature `python`).
// - [`tests`](crate::nn::spiking::layer::tests): tests unitarios.

pub mod init;
pub mod integrate;
pub mod python;
pub mod spike;
#[cfg(test)]
pub mod tests;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::compute::lagrangian::LagrangianEngine;

/// Estructura de Capa Neuromórfica Industrial (SoA - Structure of Arrays).
/// Optimizada para localidad de datos, caché y SIMD.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Debug)]
pub struct GajeNeuromorphicLayer {
    pub membrane_potentials_real: Vec<f32>,
    pub membrane_potentials_imag: Vec<f32>,
    pub thresholds: Vec<f32>,
    pub decays: Vec<f32>,
    /// Pesos empaquetados: 4 pesos de 2-bits por byte.
    pub packed_weights: Vec<u8>,
    /// Anclas de Estabilidad: Pesos de alta precisión (F16) para núcleos semánticos.
    pub anchor_indices: Vec<u32>,
    pub anchor_values: Vec<half::f16>,
    pub anchor_row_ptrs: Vec<usize>,
    pub num_neurons: usize,
    pub weights_per_neuron: usize,
    pub k_wta: usize,
    pub rms_ema: f32,
    pub lagrangian: LagrangianEngine, // Motor de física semántica
}
