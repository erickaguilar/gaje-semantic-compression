use pyo3::prelude::*;
use crate::nn::linear::GenomicLinear;
use crate::nn::attention::GenomicAttention;
use crate::kernels::rms_norm_neon;

#[pyclass]
#[derive(Clone)]
pub struct RustGenomicBlock {
    #[pyo3(get)]
    pub idx: usize,
    #[pyo3(get)]
    pub attn: GenomicAttention,
    #[pyo3(get)]
    pub q_gen: GenomicLinear,
    #[pyo3(get)]
    pub k_gen: GenomicLinear,
    #[pyo3(get)]
    pub v_gen: GenomicLinear,
    #[pyo3(get)]
    pub w_o: GenomicLinear,
    #[pyo3(get)]
    pub gate_gen: GenomicLinear,
    #[pyo3(get)]
    pub up_gen: GenomicLinear,
    #[pyo3(get)]
    pub w_down: GenomicLinear,
    #[pyo3(get)]
    pub ffn_norm: Vec<f32>,
    #[pyo3(get)]
    pub eps: f32,
}

#[pymethods]
impl RustGenomicBlock {
    #[new]
    pub fn new(
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
        }
    }

    pub fn forward(&mut self, x: Vec<f32>, pos: usize) -> PyResult<Vec<f32>> {
        // --- 1. Attention Block ---
        let x_norm = self.attn.apply_rmsnorm(x.clone())?;
        
        let q = self.q_gen.forward(x_norm.clone())?;
        let k = self.k_gen.forward(x_norm.clone())?;
        let v = self.v_gen.forward(x_norm)?;
        
        let attn_out = self.attn.forward_attention(q, k, v, pos)?;
        // DEBUG
        // println!("Rust inner attn_out: {:?}", &attn_out[0..5]);
        
        let projected_attn = self.w_o.forward(attn_out)?;
        // DEBUG
        // println!("Rust inner projected_attn: {:?}", &projected_attn[0..5]);
        
        // Residual connection
        let mut x_post_attn = x;
        for i in 0..x_post_attn.len() {
            x_post_attn[i] += projected_attn[i];
        }

        // --- 2. FeedForward Block (SwiGLU) ---
        let x_ffn_norm = unsafe { rms_norm_neon(&x_post_attn, &self.ffn_norm, self.eps) };
        
        let gate = self.gate_gen.forward(x_ffn_norm.clone())?;
        let up = self.up_gen.forward(x_ffn_norm)?;
        
        // Apply SiLU (Swish) to gate and multiply by up
        let mut swiglu_out = vec![0.0f32; gate.len()];
        for i in 0..gate.len() {
            let g = gate[i];
            let silu = g / (1.0 + (-g).exp());
            swiglu_out[i] = silu * up[i];
        }
        
        let projected_ffn = self.w_down.forward(swiglu_out)?;
        
        // Residual connection
        let mut final_out = x_post_attn;
        for i in 0..final_out.len() {
            final_out[i] += projected_ffn[i];
        }

        Ok(final_out)
    }

    pub fn clear_cache(&mut self) -> PyResult<()> {
        self.attn.clear_cache()
    }
}