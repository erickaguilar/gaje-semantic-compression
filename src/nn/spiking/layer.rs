#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct GajeNeuromorphicLayer {
    pub n_neurons: usize,
    pub threshold: f32,
}

impl GajeNeuromorphicLayer {
    pub fn new(n_neurons: usize, threshold: f32) -> Self {
        GajeNeuromorphicLayer { n_neurons, threshold }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl GajeNeuromorphicLayer {
    #[new]
    pub fn py_new(n_neurons: usize, threshold: f32) -> Self {
        Self::new(n_neurons, threshold)
    }
}
