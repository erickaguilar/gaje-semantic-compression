// =============================================================================
// python — Bindings #[pymethods] de GenomicLinear (feature `python`)
// =============================================================================
#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::nn::linear::GenomicLinear;

#[cfg(feature = "python")]
#[pymethods]
impl GenomicLinear {
    #[new]
    #[pyo3(signature = (database, anchors_u8, centroids, out_features, in_features, block_size, rmsnorm_weight=Vec::new(), eps=1e-6, precision_mask=Vec::new(), epigenetic_database=Vec::new(), epigenetic_centroids=Vec::new(), triplet_database=Vec::new(), triplet_centroids=Vec::new(), bias=Vec::new(), bit_depth=2))]
    pub fn py_new(
        database: Vec<u8>,
        anchors_u8: Vec<u8>,
        centroids: Vec<f32>,
        out_features: usize,
        in_features: usize,
        block_size: usize,
        rmsnorm_weight: Vec<f32>,
        eps: f32,
        precision_mask: Vec<u8>,
        epigenetic_database: Vec<u8>,
        epigenetic_centroids: Vec<f32>,
        triplet_database: Vec<u8>,
        triplet_centroids: Vec<f32>,
        bias: Vec<f32>,
        bit_depth: u8,
    ) -> Self {
        GenomicLinear::new(
            database,
            anchors_u8,
            centroids,
            out_features,
            in_features,
            block_size,
            rmsnorm_weight,
            eps,
            precision_mask,
            epigenetic_database,
            epigenetic_centroids,
            triplet_database,
            triplet_centroids,
            bias,
            bit_depth,
        )
    }
    pub fn forward(&self, input: Vec<f32>, activate_rna: bool) -> PyResult<Vec<f32>> {
        self.forward_core(input, None, activate_rna)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    pub fn get_row(&self, idx: usize) -> PyResult<Vec<f32>> {
        self.get_row_core(idx)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    pub fn backward(&self, d_output: Vec<f32>) -> PyResult<Vec<f32>> {
        self.backward_core(d_output)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    pub fn refine_with_grads(&mut self, input: Vec<f32>, grads: Vec<f32>, lr: f32) -> PyResult<()> {
        self.refine_with_grads_core(input, grads, lr)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    /// Etapa 3 IQAT: QAT de escala/min + re-cuantizacion de q con STE.
    pub fn refine_with_grads_ste(
        &mut self,
        input: Vec<f32>,
        grads: Vec<f32>,
        lr: f32,
    ) -> PyResult<()> {
        self.refine_with_grads_ste_core(input, grads, lr)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    pub fn recalibrate_centroids(&mut self, shift: f32) -> PyResult<()> {
        self.recalibrate_centroids_core(shift)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    pub fn apply_vector_equilibrium_alignment(&mut self, strength: f32) -> PyResult<()> {
        self.apply_vector_equilibrium_alignment_core(strength)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[getter]
    pub fn database(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            use pyo3::types::PyBytes;
            Ok(PyBytes::new(py, self.database_ref()).into())
        })
    }
    #[getter]
    pub fn centroids(&self) -> PyResult<Vec<f32>> {
        Ok(self.centroids.clone())
    }
}
