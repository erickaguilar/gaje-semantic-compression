// =============================================================================
// llm — Núcleo del Modelo de Lenguaje Genómico (Pure Rust)
// =============================================================================
//
// `GenomicLLM` orquesta embeddings + bloques transformer + LM head sobre
// pesos comprimidos. Es la entrada usada por el CLI, el SDK, los cargadores y
// los entrenadores.
//
// Este módulo es un **hub de re-exportación** que preserva la API pública
// histórica (`crate::nn::llm::*`) mientras la implementación vive en
// submódulos cohesionados:
//
// - [`forward`](crate::nn::llm::forward): forwards, entrenamiento y generación.
// - [`mutation`](crate::nn::llm::mutation): mutaciones por capa y homeostasis.
// - [`python`](crate::nn::llm::python): bindings `#[pymethods]` (feature `python`).

pub mod forward;
pub mod mutation;
pub mod python;
#[cfg(test)]
pub mod integration_tests;

use std::sync::Arc;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::compute::kernels::rms_norm;
use crate::core::quantum_codebook::QuantumEmbeddingTableNative;
use crate::core::topology::CentroidGraph;
use crate::nn::block::RustGenomicBlock;
use crate::nn::linear::GenomicLinear;

/// Núcleo del Modelo de Lenguaje Genómico (Pure Rust)
#[cfg(feature = "python")]
#[pyclass(name = "RustGenomicLLM")]
#[derive(Clone)]
pub struct GenomicLLM {
    #[pyo3(get, set)]
    pub embeddings: GenomicLinear,
    #[pyo3(get, set)]
    pub blocks: Vec<RustGenomicBlock>,
    #[pyo3(get, set)]
    pub output_norm: Vec<f32>,
    #[pyo3(get, set)]
    pub lm_head: GenomicLinear,
    #[pyo3(get, set)]
    pub eps: f32,
    pub k_wta_ratio: f32,
    pub topology: Option<Arc<CentroidGraph>>,
    pub quantum_embeddings: Option<Arc<QuantumEmbeddingTableNative>>,
}

#[cfg(not(feature = "python"))]
#[derive(Clone)]
pub struct GenomicLLM {
    pub embeddings: GenomicLinear,
    pub blocks: Vec<RustGenomicBlock>,
    pub output_norm: Vec<f32>,
    pub lm_head: GenomicLinear,
    pub eps: f32,
    pub k_wta_ratio: f32,
    pub topology: Option<Arc<CentroidGraph>>,
    pub quantum_embeddings: Option<Arc<QuantumEmbeddingTableNative>>,
}

pub use crate::nn::llm::forward::*;
pub use crate::nn::llm::mutation::*;
pub use crate::nn::llm::python::*;
