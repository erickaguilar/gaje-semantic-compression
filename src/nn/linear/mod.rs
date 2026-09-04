// =============================================================================
// linear — Capa lineal genómica (LLM) para GAJE
// =============================================================================
//
// Implementa `GenomicLinear`: una capa densa cuyos pesos viven en una
// `WeightDatabase` comprimida (2/4-bit, Q4_0, Q8_0 o f32) con soporte opcional
// de sparse-anchors. Es el bloque primitivo usado por attention, block y llm.
//
// Este módulo es un **hub de re-exportación** que preserva la API pública
// histórica (`crate::nn::linear::*`) mientras la implementación vive en
// submódulos cohesionados:
//
// - [`storage`](crate::nn::linear::storage): `WeightStorage` y el trait `GenomicOperable`.
// - [`init`](crate::nn::linear::init): construcción (`new`/`empty`) y acceso crudo a pesos.
// - [`forward`](crate::nn::linear::forward): cómputo de filas y forwards (single/fused).
// - [`backward`](crate::nn::linear::backward): gradientes, refine con gradientes y mutaciones.
// - [`python`](crate::nn::linear::python): bindings `#[pymethods]` (feature `python`).
// - [`tests`](crate::nn::linear::tests): tests unitarios.

pub mod backward;
pub mod forward;
pub mod init;
pub mod python;
pub mod storage;
pub use storage as database;
#[cfg(test)]
pub mod tests;

use half::f16;
use std::sync::Arc;

#[cfg(feature = "python")]
use pyo3::prelude::*;

pub use crate::nn::linear::storage::*;

#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct GenomicLinear {
    pub weight_db: WeightStorage,
    pub epi_strands: Arc<Vec<u8>>,
    pub tri_strands: Arc<Vec<u8>>,
    pub epi_cols: Arc<Vec<(usize, usize)>>,
    pub tri_cols: Arc<Vec<(usize, usize)>>,
    pub anchor_indices: Arc<Vec<u32>>,
    pub anchor_values: Arc<Vec<f16>>,
    pub anchor_row_ptrs: Arc<Vec<usize>>,
    pub centroids: Vec<f32>,
    pub epigenetic_centroids: Vec<f32>,
    pub triplet_centroids: Vec<f32>,
    pub out_features: usize,
    pub in_features: usize,
    pub block_size: usize,
    pub rmsnorm_weight: Vec<f32>,
    pub eps: f32,
    pub bias: Vec<f32>,
    pub stride: usize,
}

