#![allow(
    dead_code,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::manual_checked_ops,
    clippy::non_canonical_partial_ord_impl,
    clippy::manual_div_ceil
)]
pub mod core;
pub mod compute;
pub mod io;
pub mod nn;
pub mod pyo3_shim;
pub mod ffi;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _impl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    unsafe { crate::compute::kernels::init_shuffle_table(); }
    m.add_class::<crate::core::index::GajeIndex>()?;
    m.add_class::<crate::core::dni::SemanticNiche>()?;
    m.add_class::<crate::core::session_memory::SessionBuffer>()?;
    m.add_class::<crate::core::dni::DNIEngine>()?;
    m.add_class::<crate::nn::attention::GenomicAttention>()?;
    m.add_class::<crate::nn::linear::GenomicLinear>()?;
    m.add_class::<crate::nn::spiking::layer::GajeNeuromorphicLayer>()?;
    m.add_class::<crate::compute::scheduler::NeuromorphicScheduler>()?;
    m.add_class::<crate::compute::event_queue::SpikeEvent>()?;
    m.add_class::<crate::nn::block::RustGenomicBlock>()?;
    m.add_class::<crate::nn::llm::GenomicLLM>()?;
    m.add_class::<crate::compute::sampler::ToroidalSampler>()?;
    m.add_class::<crate::compute::sampler::SintergicSampler>()?;
    m.add_class::<crate::io::loader::NativeLoader>()?;
    m.add_class::<crate::core::db::GajeDatabaseWriter>()?;
    m.add_class::<crate::core::db::GajeDatabaseReader>()?;
    m.add_class::<crate::io::loader::ModelConfig>()?;
    m.add_class::<crate::io::loader::ArchConfig>()?;
    m.add_class::<crate::nn::trainer::NativeGenomicTrainer>()?;
    m.add_class::<crate::nn::distiller::Teacher>()?;
    m.add_class::<crate::nn::distiller::CouncilOfTeachers>()?;
    m.add_class::<crate::nn::distiller::NativeGenomicDistiller>()?;
    m.add_class::<crate::nn::iqat::NativeIQATEngine>()?;
    m.add_function(wrap_pyfunction!(crate::io::loader::init_born_genomic_model_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::io::loader::save_genomic_model_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::nn::merger::merge_models_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::compute::math::dequantize_embedding_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::compute::math::quantize_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(crate::compute::math::quantize_pq, m)?)?;
    m.add_function(wrap_pyfunction!(crate::compute::math::dequantize_q8_0_native, m)?)?;
    m.add_function(wrap_pyfunction!(crate::compute::math::genomize_f32_native, m)?)?;
    m.add_function(wrap_pyfunction!(crate::compute::math::genomize_f16_native, m)?)?;
    m.add_function(wrap_pyfunction!(crate::compute::math::quantize_phase_native, m)?)?;
    m.add_function(wrap_pyfunction!(crate::compute::math::dequantize_phase_native, m)?)?;
    m.add_function(wrap_pyfunction!(crate::compute::math::calculate_shannon_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(crate::compute::math::calculate_genomic_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(crate::compute::math::apply_repetition_penalty, m)?)?;
    m.add_function(wrap_pyfunction!(crate::compute::math::sample_top_p, m)?)?;
    Ok(())
}
