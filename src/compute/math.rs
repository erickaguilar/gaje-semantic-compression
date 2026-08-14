//! # 🧬 Motor de Cómputo Genómico: Espacio de Fase Toroidal
//!
//! Este módulo es un **hub de re-exportación** que preserva la API pública histórica
//! (`crate::compute::math::*`) mientras la implementación vive en submódulos cohesionados:
//!
//! - [`quantize`](crate::compute::quantize): genomización y cuantización (Q4_0, Q8_0, f16, toroidal).
//! - [`phase`](crate::compute::phase): espacio de fase compleja y operaciones toroidales.
//! - [`sampling`](crate::compute::sampling): muestreo autoregresivo y utilidades de generación.
//! - [`metrics`](crate::compute::metrics): métricas de densidad informativa (MSE, entropías, RMS).
//! - [`search`](crate::compute::search): búsqueda de similitud y pruning de bases genómicas.

pub use crate::compute::metrics::*;
pub use crate::compute::phase::*;
pub use crate::compute::quantize::*;
pub use crate::compute::sampling::*;
pub use crate::compute::search::*;