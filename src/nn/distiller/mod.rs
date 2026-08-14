// =============================================================================
// distiller — Consejo de Profesores y destilación de conocimiento
// =============================================================================
//
// Implementa la destilación por consenso: uno o más `Teacher` (modelos GGUF)
// forman un `CouncilOfTeachers` cuyas probabilidades guían al estudiante a
// través de `GenomicDistiller`/`NativeGenomicDistiller`.
//
// Este módulo es un **hub de re-exportación** que preserva la API pública
// histórica (`crate::nn::distiller::*`) mientras la implementación vive en
// submódulos cohesionados:
//
// - [`teacher`](crate::nn::distiller::teacher): `Teacher` (maestro GGUF + mapeo de vocabulario).
// - [`council`](crate::nn::distiller::council): `CouncilOfTeachers` (consenso entre maestros).
// - [`distill`](crate::nn::distiller::distill): `GenomicDistiller`/`NativeGenomicDistiller` (ciclo de destilación).

pub mod council;
pub mod distill;
pub mod teacher;

pub use crate::nn::distiller::council::*;
pub use crate::nn::distiller::distill::*;
pub use crate::nn::distiller::teacher::*;
