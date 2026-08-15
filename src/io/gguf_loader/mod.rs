// =============================================================================
// gguf_loader — Carga de modelos GGUF en representación genómica
// =============================================================================
//
// Lee modelos GGUF y los convierte a la representación interna genómica
// (`GenomicLLM`) aplicando cuantización mixta, anclas y unpermute de pesos.
//
// Este módulo es un **hub de re-exportación** que preserva la API pública
// histórica (`crate::io::gguf_loader::*`) mientras la implementación vive en
// submódulos cohesionados:
//
// - [`metadata`](crate::io::gguf_loader::metadata): lecturas de metadatos y `infer_config`.
// - [`load`](crate::io::gguf_loader::load): `load_genomic_llm` (construcción del modelo).
// - [`tensors`](crate::io::gguf_loader::tensors): lectura/genomización de tensores y unpermute.

pub mod load;
pub mod metadata;
pub mod tensors;

use crate::io::gguf::GGUFReader;

/// Carga modelos GGUF y los traduce a la representación genómica.
pub struct GGUFLoader {
    pub reader: GGUFReader,
}

pub use crate::io::gguf_loader::load::*;
pub use crate::io::gguf_loader::metadata::*;
pub use crate::io::gguf_loader::tensors::*;
