// =============================================================================
// evolution_bitwise — Evolución por poblaciones paralelas (Island Model)
// =============================================================================
//
// Motor evolutivo que opera sobre organismos neuromórficos (capas spiking)
// o LLMs genómicos mediante mutación bitwise y crossover, organizados en
// islas paralelas con migración.
//
// Este módulo es un **hub de re-exportación** que preserva la API pública
// histórica (`crate::core::evolution_bitwise::*`) mientras la implementación
// vive en submódulos cohesionados:
//
// - [`organism`](crate::core::evolution_bitwise::organism): `NeuromorphicOrganism` (mutación y crossover).
// - [`engine`](crate::core::evolution_bitwise::engine): `SpikingEvolutionEngine` (población y evolución).
// - [`island`](crate::core::evolution_bitwise::island): `IslandModel` (islas paralelas y migración).

pub mod engine;
pub mod island;
pub mod organism;

pub use crate::core::evolution_bitwise::engine::*;
pub use crate::core::evolution_bitwise::island::*;
pub use crate::core::evolution_bitwise::organism::*;
