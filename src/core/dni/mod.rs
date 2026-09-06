// =============================================================================
// dni — DNIEngine: Motor de Ingestión Neuronal Directa para GAJE-Flow
// =============================================================================
//
// Permite la inyección granular de conocimiento en los pesos de 2 bits mediante
// evolución dirigida ultrarrápida (modelo de islas, enfriamiento termodinámico).
//
// Este módulo es un **hub de re-exportación** que preserva la API pública
// histórica (`crate::core::dni::*`) mientras la implementación vive en
// submódulos cohesionados:
//
// - [`evolution`](crate::core::dni::evolution): mutación dirigida, evaluación y perfilado.
// - [`merge`](crate::core::dni::merge): fusión y migración de conocimiento entre modelos.
// - [`python`](crate::core::dni::python): bindings `#[pymethods]` + métodos de ingestión.

pub mod evolution;
pub mod merge;
pub mod python;

use std::sync::Arc;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::core::tokenizer::GajeTokenizer;
use crate::nn::distiller::CouncilOfTeachers;
use crate::nn::llm::GenomicLLM;

/// # 🏝️ Island Model: Especialización por Nichos
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticNiche {
    General,
    Logic,
    Grammar,
    Memory,
}

#[cfg_attr(feature = "python", pyclass)]
pub struct DNIEngine {
    pub model: GenomicLLM,
    pub tokenizer: Arc<GajeTokenizer>,
    pub council: Option<Arc<CouncilOfTeachers>>,
    pub intensity: f32,
    pub target_layers: Vec<String>,
    pub validation_tokens: Vec<u32>,
    pub original_dna_hash: Vec<u64>,
    pub niche: SemanticNiche,
}
