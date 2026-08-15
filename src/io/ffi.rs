use crate::core::tokenizer::GajeTokenizer;
use crate::io::config::ModelConfig;
use crate::io::db_loader::NativeLoader;
use crate::io::flat_reader::load_genomic_auto;
use crate::io::flat_writer::{init_born_genomic_model, save_genomic_model};
use crate::nn::llm::GenomicLLM;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg_attr(feature = "python", pymethods)]
impl NativeLoader {
    #[cfg(feature = "python")]
    #[new]
    pub fn py_new(path: &str) -> PyResult<Self> {
        Self::new(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    #[cfg(feature = "python")]
    pub fn py_load_config(&self) -> PyResult<ModelConfig> {
        self.load_config()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[cfg(feature = "python")]
    pub fn py_load_llm(&self) -> PyResult<GenomicLLM> {
        self.load_llm()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "save_genomic_model", signature = (path, model, config, tokenizer_path=None))]
pub fn save_genomic_model_py(
    path: &str,
    model: &GenomicLLM,
    config: &ModelConfig,
    tokenizer_path: Option<&str>,
) -> PyResult<()> {
    let tok = if let Some(p) = tokenizer_path {
        Some(
            GajeTokenizer::from_file(p)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?,
        )
    } else {
        None
    };
    save_genomic_model(path, model, config, tok.as_ref())
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "init_born_genomic_model")]
pub fn init_born_genomic_model_py(
    path: &str,
    config: ModelConfig,
    vocab_size: usize,
) -> PyResult<GenomicLLM> {
    let inner = init_born_genomic_model(path, config, vocab_size)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(inner)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "load_genomic_auto")]
pub fn load_genomic_auto_py(path: &str) -> PyResult<GenomicLLM> {
    load_genomic_auto(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}
