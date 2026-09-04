// =============================================================================
// gguf — Lector del formato binario GGUF (metadatos y tensores)
// =============================================================================
//
// Parsea la cabecera de un archivo GGUF: versión, metadatos (key-value) e
// información de tensores, exponiendo acceso mmap a los datos de cada tensor.
//
// Este módulo es un **hub de re-exportación** que preserva la API pública
// histórica (`crate::io::gguf::*`) mientras la implementación vive en
// submódulos cohesionados:
//
// - [`types`](crate::io::gguf::types): tipos básicos (`GGUFValueType`, `GGUFValue`, `GGMLType`, `GGUFTensorInfo`).
// - [`reader`](crate::io::gguf::reader): `GGUFReader` (apertura y lectura binaria).
// - [`writer`](crate::io::gguf::writer): `GGUFWriter` (serialización binaria).

pub mod loader;
pub mod reader;
pub mod types;
pub mod writer;

pub use crate::io::gguf::loader::*;
pub use crate::io::gguf::reader::*;
pub use crate::io::gguf::types::*;
pub use crate::io::gguf::writer::*;
