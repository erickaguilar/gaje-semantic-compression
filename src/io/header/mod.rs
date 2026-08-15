// =============================================================================
// header — Cabecera del formato .flat v2 y bloques de cuantización group-wise
// =============================================================================
//
// Define la estructura binaria de cabecera (`FlatHeaderV2`) del formato .flat
// v2 junto con los bloques de cuantización `Q4_0Block`/`Q8_0Block` y sus tipos
// auxiliares (`QuantFormat`, `HeaderError`).
//
// Este módulo es un **hub de re-exportación** que preserva la API pública
// histórica (`crate::io::header::*`) mientras la implementación vive en
// submódulos cohesionados:
//
// - [`types`](crate::io::header::types): `QuantFormat` y `HeaderError`.
// - [`flat`](crate::io::header::flat): `FlatHeaderV2` (cabecera de 4096 bytes).
// - [`blocks`](crate::io::header::blocks): `Q4_0Block`/`Q8_0Block` (dequantización group-wise).

pub mod blocks;
pub mod flat;
pub mod types;

#[cfg(test)]
pub(crate) mod tests;

pub use crate::io::header::blocks::*;
pub use crate::io::header::flat::*;
pub use crate::io::header::types::*;
