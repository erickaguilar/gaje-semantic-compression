// =============================================================================
// python — Bindings #[pymethods] de RustGenomicBlock (feature `python`)
// =============================================================================
#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::nn::attention::GenomicAttention;
use crate::nn::block::RustGenomicBlock;
use crate::nn::linear::GenomicLinear;

#[cfg(feature = "python")]
#[pymethods]
impl RustGenomicBlock {
    #[new]
    #[pyo3(signature = (idx, attn, q_gen, k_gen, v_gen, w_o, gate_gen, up_gen, w_down, ffn_norm, eps, act_fn = "swiglu".to_string(), use_genomic_norm = false, h_scale = 1.0, rna_threshold = 0.5))]
    pub fn py_new(
        idx: usize,
        attn: GenomicAttention,
        q_gen: GenomicLinear,
        k_gen: GenomicLinear,
        v_gen: GenomicLinear,
        w_o: GenomicLinear,
        gate_gen: GenomicLinear,
        up_gen: GenomicLinear,
        w_down: GenomicLinear,
        ffn_norm: Vec<f32>,
        eps: f32,
        act_fn: String,
        use_genomic_norm: bool,
        h_scale: f32,
        rna_threshold: f32,
    ) -> Self {
        RustGenomicBlock {
            idx,
            attn,
            q_gen,
            k_gen,
            v_gen,
            w_o,
            gate_gen,
            up_gen,
            w_down,
            ffn_norm,
            eps,
            act_fn,
            use_genomic_norm,
            h_scale,
            rna_threshold,
            k_wta_ratio: 0.0,
            topology: None,
            fused_qkv: None,
            fused_gate_up: None,
        }
    }
    pub fn forward(&mut self, x: Vec<f32>, pos: usize) -> PyResult<Vec<f32>> {
        self.forward_core(x, pos)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    pub fn clear_cache(&mut self) -> PyResult<()> {
        self.clear_cache_core();
        Ok(())
    }
    pub fn refine_ffn(&mut self, x_norm: Vec<f32>, target: Vec<f32>, lr: f32) -> PyResult<()> {
        self.refine_ffn_core(x_norm, target, lr)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    pub fn refine_attention(
        &mut self,
        x_norm: Vec<f32>,
        target: Vec<f32>,
        pos: usize,
        lr: f32,
    ) -> PyResult<()> {
        self.refine_attention_core(x_norm, target, pos, lr)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}