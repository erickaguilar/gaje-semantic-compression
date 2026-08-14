// =============================================================================
// python — Bindings #[pymethods] de GajeNeuromorphicLayer (feature `python`)
// =============================================================================
#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::nn::spiking::layer::GajeNeuromorphicLayer;

#[cfg_attr(feature = "python", pymethods)]
impl GajeNeuromorphicLayer {
    #[cfg(feature = "python")]
    #[new]
    pub fn new_py(
        num_neurons: usize,
        weights_per_neuron: usize,
        threshold: f32,
        decay: f32,
    ) -> Self {
        Self::new(num_neurons, weights_per_neuron, threshold, decay)
    }

    #[cfg(feature = "python")]
    #[getter]
    pub fn get_membrane_potentials_real(&self) -> Vec<f32> {
        self.membrane_potentials_real.clone()
    }

    #[cfg(feature = "python")]
    #[getter]
    pub fn get_membrane_potentials_imag(&self) -> Vec<f32> {
        self.membrane_potentials_imag.clone()
    }

    #[cfg(feature = "python")]
    #[getter]
    pub fn get_packed_weights<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        Ok(pyo3::types::PyBytes::new(py, &self.packed_weights))
    }

    #[cfg(feature = "python")]
    pub fn load_packed_weights(&mut self, data: Vec<u8>) -> PyResult<()> {
        if data.len() != self.packed_weights.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Weight size mismatch: expected {}, got {}",
                self.packed_weights.len(),
                data.len()
            )));
        }
        self.packed_weights = data;
        Ok(())
    }
}
