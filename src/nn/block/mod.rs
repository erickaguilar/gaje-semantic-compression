// =============================================================================
// block — Bloque de Procesamiento Genómico (Pure Rust)
// =============================================================================
//
// `RustGenomicBlock` encapsula un transformer block completo: atención +
// FFN, todo sobre pesos comprimidos con modulación relacional y confinamiento
// K-WTA.
//
// Este módulo es un **hub de re-exportación** que preserva la API pública
// histórica (`crate::nn::block::*`) mientras la implementación vive en
// submódulos cohesionados:
//
// - [`init`](crate::nn::block::init): construcción (`new`) y homeostasis.
// - [`forward`](crate::nn::block::forward): forward del bloque y limpieza de caché.
// - [`refine`](crate::nn::block::refine): refine con gradientes (FFN y atención).
// - [`python`](crate::nn::block::python): bindings `#[pymethods]` (feature `python`).

pub mod cache;
pub mod forward;
pub mod init;
pub mod python;
pub mod refine;

use std::sync::Arc;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::core::topology::CentroidGraph;
use crate::nn::attention::GenomicAttention;
use crate::nn::linear::GenomicLinear;

/// Bloque de Procesamiento Genómico (Pure Rust)
#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone)]
pub struct RustGenomicBlock {
    #[pyo3(get, set)]
    pub idx: usize,
    #[pyo3(get, set)]
    pub attn: GenomicAttention,
    #[pyo3(get, set)]
    pub q_gen: GenomicLinear,
    #[pyo3(get, set)]
    pub k_gen: GenomicLinear,
    #[pyo3(get, set)]
    pub v_gen: GenomicLinear,
    #[pyo3(get, set)]
    pub w_o: GenomicLinear,
    #[pyo3(get, set)]
    pub gate_gen: GenomicLinear,
    #[pyo3(get, set)]
    pub up_gen: GenomicLinear,
    #[pyo3(get, set)]
    pub w_down: GenomicLinear,
    #[pyo3(get, set)]
    pub ffn_norm: Vec<f32>,
    #[pyo3(get, set)]
    pub eps: f32,
    #[pyo3(get, set)]
    pub act_fn: String,
    #[pyo3(get, set)]
    pub use_genomic_norm: bool,
    #[pyo3(get, set)]
    pub h_scale: f32,
    #[pyo3(get, set)]
    pub rna_threshold: f32,
    #[pyo3(get, set)]
    pub k_wta_ratio: f32,
    pub topology: Option<Arc<CentroidGraph>>,
    #[pyo3(get, set)]
    pub fused_qkv: Option<GenomicLinear>,
    #[pyo3(get, set)]
    pub fused_gate_up: Option<GenomicLinear>,
    pub mla: Option<crate::nn::attention::MlaAttention>,
    pub moe: Option<crate::nn::moe::MoeRouter>,
}

#[cfg(not(feature = "python"))]
#[derive(Clone)]
pub struct RustGenomicBlock {
    pub idx: usize,
    pub attn: GenomicAttention,
    pub q_gen: GenomicLinear,
    pub k_gen: GenomicLinear,
    pub v_gen: GenomicLinear,
    pub w_o: GenomicLinear,
    pub gate_gen: GenomicLinear,
    pub up_gen: GenomicLinear,
    pub w_down: GenomicLinear,
    pub ffn_norm: Vec<f32>,
    pub eps: f32,
    pub act_fn: String,
    pub use_genomic_norm: bool,
    pub h_scale: f32,
    pub rna_threshold: f32,
    pub k_wta_ratio: f32,
    pub topology: Option<Arc<CentroidGraph>>,
    pub fused_qkv: Option<GenomicLinear>,
    pub fused_gate_up: Option<GenomicLinear>,
    pub mla: Option<crate::nn::attention::MlaAttention>,
    pub moe: Option<crate::nn::moe::MoeRouter>,
}

pub use crate::nn::block::cache::*;
pub use crate::nn::block::forward::*;
pub use crate::nn::block::init::*;
pub use crate::nn::block::python::*;
pub use crate::nn::block::refine::*;
