// =============================================================================
// python.rs — Bindings PyO3 para el backend de aceleración GPU
// =============================================================================

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
#[pyfunction]
pub fn is_gpu_available_py() -> bool {
    crate::compute::gpu::context::is_gpu_available()
}

#[cfg(feature = "python")]
#[pyfunction]
pub fn get_gpu_info_py(py: Python<'_>) -> PyResult<Option<PyObject>> {
    if let Some(info) = crate::compute::gpu::context::get_gpu_info() {
        let dict = PyDict::new(py);
        dict.set_item("device_name", &info.device_name)?;
        dict.set_item("backend", &info.backend)?;
        dict.set_item("device_type", &info.device_type)?;
        dict.set_item("is_unified_memory", info.is_unified_memory)?;
        dict.set_item("max_buffer_size_mb", info.max_buffer_size_mb)?;
        dict.set_item(
            "max_compute_workgroups_per_dim",
            info.max_compute_workgroups_per_dim.to_vec(),
        )?;
        Ok(Some(dict.into()))
    } else {
        Ok(None)
    }
}

#[cfg(feature = "python")]
#[pyfunction]
pub fn gpu_swiglu_py(gate: Vec<f32>, up: Vec<f32>, h_scale: f32) -> PyResult<Option<Vec<f32>>> {
    Ok(crate::compute::gpu::pipeline::gpu_swiglu(&gate, &up, h_scale))
}

#[cfg(feature = "python")]
#[pyfunction]
pub fn gpu_gemv_f32_py(
    weights: Vec<f32>,
    x: Vec<f32>,
    rows: usize,
    cols: usize,
) -> PyResult<Option<Vec<f32>>> {
    Ok(crate::compute::gpu::pipeline::gpu_gemv_f32(&weights, &x, rows, cols))
}
