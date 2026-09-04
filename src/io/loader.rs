//! # 🧬 Cargadores y Escritores del Motor GAJE
//!
//! Este módulo es un **hub de re-exportación** que mantiene la API pública histórica
//! (`crate::io::loader::*`) mientras la implementación vive en submódulos cohesionados:
//!
//! - [`config`](crate::io::config): estructuras de configuración (`ArchConfig`, `ModelConfig`).
//! - [`gguf_loader`](crate::io::gguf_loader): importación desde formato GGUF.
//! - [`db_loader`](crate::io::db_loader): carga/lectura desde la base `redb` (`NativeLoader`).
//! - [`flat_reader`](crate::io::flat_reader): lectura zero-copy del formato `.gaje.flat`.
//! - [`flat_writer`](crate::io::flat_writer): guardado e inicialización de modelos.
//! - [`ffi`](crate::io::ffi): wrappers PyO3 hacia el núcleo.

#[derive(Clone)]
pub struct NativeLoader {
    pub reader: std::sync::Arc<crate::io::flat_reader::GajeFlatFileReader>,
}

impl NativeLoader {
    pub fn new(path: &str) -> std::io::Result<Self> {
        let reader = crate::io::flat_reader::GajeFlatFileReader::open(path)?;
        Ok(Self {
            reader: std::sync::Arc::new(reader),
        })
    }

    pub fn load_llm(&self) -> std::io::Result<crate::nn::llm::GenomicLLM> {
        self.reader.load_genomic()
    }

    pub fn load_config(&self) -> std::io::Result<crate::io::config::ModelConfig> {
        self.reader.load_config()
    }

    pub fn load_tokenizer(&self) -> std::io::Result<crate::core::tokenizer::GajeTokenizer> {
        if let Some(gtok) = self.reader.get_embedded_gtok() {
            Ok(crate::core::tokenizer::GajeTokenizer::from_gtok(gtok))
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Tokenizer not found in GAJE flat model",
            ))
        }
    }
}

pub use crate::io::config::*;
pub use crate::io::flat_reader::*;
#[cfg(feature = "native")]
pub use crate::io::flat_writer::*;
#[cfg(feature = "native")]
pub use crate::io::gguf_loader::*;

pub fn load_topology(path: &str) -> std::io::Result<crate::core::topology::CentroidGraph> {
    let file = std::fs::File::open(path)?;
    let topo: crate::core::topology::CentroidGraph = serde_json::from_reader(file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    Ok(topo)
}
