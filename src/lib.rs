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

use crate::core::index::GajeIndex;
use crate::nn::{GenomicAttention, GenomicLinear, RustGenomicBlock, RustGenomicLLM};
use crate::core::db::{GajeDatabaseWriter, GajeDatabaseReader};
use crate::io::loader::{ModelConfig, ArchConfig};
use crate::compute::math::*;
use pyo3::prelude::*;

use crate::nn::spiking::layer::GajeNeuromorphicLayer;

#[pymodule]
fn _impl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    unsafe {
        crate::compute::kernels::init_shuffle_table();
    }
    m.add_class::<GajeIndex>()?;
    m.add_class::<GenomicAttention>()?;
    m.add_class::<GenomicLinear>()?;
    m.add_class::<GajeNeuromorphicLayer>()?;
    m.add_class::<RustGenomicBlock>()?;
    m.add_class::<RustGenomicLLM>()?;
    m.add_class::<GajeDatabaseWriter>()?;
    m.add_class::<GajeDatabaseReader>()?;
    m.add_class::<ModelConfig>()?;
    m.add_class::<ArchConfig>()?;
    m.add_function(wrap_pyfunction!(crate::io::loader::init_born_genomic_model_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::io::loader::save_genomic_model_py, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_pq, m)?)?;
    m.add_function(wrap_pyfunction!(dna_similarity_search_adc, m)?)?;
    m.add_function(wrap_pyfunction!(dna_similarity_search, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(apply_repetition_penalty, m)?)?;
    m.add_function(wrap_pyfunction!(genomize_f32_native, m)?)?;
    m.add_function(wrap_pyfunction!(genomize_f16_native, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_q8_0_native, m)?)?;
    m.add_function(wrap_pyfunction!(sample_top_p, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_shannon_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_mse_native, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_cosine_similarity_native, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_distribution_entropy_native, m)?)?;
    m.add_function(wrap_pyfunction!(prune_genomic_database, m)?)?;
    m.add_function(wrap_pyfunction!(generate_precision_mask_native, m)?)?;
    m.add_function(wrap_pyfunction!(get_active_dimensions_native, m)?)?;
    Ok(())
}
