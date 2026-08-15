// =============================================================================
// db_loader — Carga de modelos desde la base redb (NativeLoader)
// =============================================================================
//
// Lee modelos genómicos persistidos en una base `redb`: configuración,
// tensores (dna/centroids/anchors/bias), bloques del LLM, tokenizador y
// mutaciones. También es el punto de entrada para scripts Python.
//
// Este módulo es un **hub de re-exportación** que preserva la API pública
// histórica (`crate::io::db_loader::*`) mientras la implementación vive en
// submódulos cohesionados:
//
// - [`config`](crate::io::db_loader::config): apertura de la base y `load_config`.
// - [`llm`](crate::io::db_loader::llm): `load_llm` (construcción del modelo completo).
// - [`tensor`](crate::io::db_loader::tensor): helpers de lectura de tensores.
// - [`misc`](crate::io::db_loader::misc): `load_tokenizer` y `list_mutations`.

pub mod config;
pub mod llm;
pub mod misc;
pub mod tensor;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use std::sync::Arc;

#[cfg_attr(feature = "python", pyclass)]
pub struct NativeLoader {
    pub db: Arc<redb::Database>,
}

pub use crate::io::db_loader::config::*;
pub use crate::io::db_loader::llm::*;
pub use crate::io::db_loader::misc::*;
