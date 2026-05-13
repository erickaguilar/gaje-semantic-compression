mod kernels;
mod utils;
mod index;
pub mod nn;

use pyo3::prelude::*;
use crate::index::GajeIndex;
use crate::nn::{GenomicLinear, GenomicAttention, RustGenomicBlock, RustGenomicLLM};
use crate::utils::*;
#[pymodule]
fn _impl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    unsafe { crate::kernels::init_shuffle_table(); }
    m.add_class::<GajeIndex>()?;
    m.add_class::<GenomicAttention>()?;
    m.add_class::<GenomicLinear>()?;
    m.add_class::<RustGenomicBlock>()?;
    m.add_class::<RustGenomicLLM>()?;
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
    m.add_function(wrap_pyfunction!(prune_genomic_database, m)?)?;
    Ok(())
}
