// =============================================================================
// kernels — Motor SIMD multiplataforma para GAJE
//
// aarch64  → ARM NEON (Android/Termux)
// x86_64   → AVX2 + FMA (Windows/Linux PC)
// fallback → escalar puro (cualquier otro target)
// =============================================================================
//
// Este módulo es un **hub de re-exportación** que preserva la API pública histórica
// (`crate::compute::kernels::*`) mientras la implementación vive en submódulos cohesionados:
//
// - [`dot`](crate::compute::kernels::dot): productos punto SIMD (float + comprimido).
// - [`norm`](crate::compute::kernels::norm): normalización RMS.
// - [`activation`](crate::compute::kernels::activation): activaciones GLU (SwiGLU/GeGLU/ReLU).
// - [`genomic`](crate::compute::kernels::genomic): productos punto genómicos (2/4-bit, Q4_0, Q8_0) + GEMV.
// - [`lut`](crate::compute::kernels::lut): distancia LUT, tabla de shuffle e inhibición K-WTA.

pub mod activation;
pub mod dot;
pub mod genomic;
pub mod lut;
pub mod norm;

pub use crate::compute::kernels::activation::*;
pub use crate::compute::kernels::dot::*;
pub use crate::compute::kernels::genomic::*;
pub use crate::compute::kernels::lut::*;
pub use crate::compute::kernels::norm::*;
