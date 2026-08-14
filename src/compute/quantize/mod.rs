// =============================================================================
// quantize — Genomización y cuantización de pesos (Q4_0, Q8_0, f16, toroidal)
// =============================================================================
//
// Convierte tensores f32/f16 en bases de datos comprimidas ("DNA") con
// centroides y anclas sparse en formato "GAJE".
//
// Este módulo es un **hub de re-exportación** que preserva la API pública
// histórica (`crate::compute::quantize::*`, reexportado por `math`) mientras
// la implementación vive en submódulos cohesionados:
//
// - [`genomize`](crate::compute::quantize::genomize): genomización de f32/f16 y 4-bit + toroidal.
// - [`dequantize`](crate::compute::quantize::dequantize): reconstrucción de embeddings y bloques.
// - [`python`](crate::compute::quantize::python): wrappers `#[pyfunction]` (feature `python`).

pub mod dequantize;
pub mod genomize;
pub mod python;

pub use crate::compute::quantize::dequantize::*;
pub use crate::compute::quantize::genomize::*;
pub use crate::compute::quantize::python::*;
