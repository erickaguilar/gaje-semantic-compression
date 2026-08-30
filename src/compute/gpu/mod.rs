// =============================================================================
// gpu/mod.rs — Módulo de Cómputo y Aceleración por GPU (Vulkan / WGPU)
// =============================================================================

pub mod context;
pub mod pipeline;
pub mod scheduler;

#[cfg(feature = "python")]
pub mod python;
