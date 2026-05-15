use crate::kernels::rms_norm_neon;
use crate::nn::attention::GenomicAttention;
use crate::nn::linear::GenomicLinear;
use pyo3::prelude::*;

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

        let projected_attn = self.w_o.forward(attn_out)?;

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

    pub fn refine_swiglu(
        &mut self,
        x_ffn_norm: Vec<f32>,
        swiglu_target: Vec<f32>,
        lr: f32,
    ) -> PyResult<()> {
        let gate = self.gate_gen.forward(x_ffn_norm.clone())?;
        let up = self.up_gen.forward(x_ffn_norm.clone())?;

        let mut d_gate = vec![0.0f32; gate.len()];
        let mut d_up = vec![0.0f32; gate.len()];

        for i in 0..gate.len() {
            let g = gate[i];
            let u = up[i];
            let s = 1.0 / (1.0 + (-g).exp());
            let current_swiglu = g * s * u;
            let diff = current_swiglu - swiglu_target[i];

            // d(swiglu)/dg = silu'(g) * u
            // silu'(g) = s * (1 + g * (1 - s))
            let silu_p = s * (1.0 + g * (1.0 - s));
            d_gate[i] = diff * silu_p * u;

            // d(swiglu)/du = silu(g) = g * s
            d_up[i] = diff * (g * s);
        }

        self.gate_gen
            .refine_with_grads(x_ffn_norm.clone(), d_gate, lr)?;
        self.up_gen.refine_with_grads(x_ffn_norm, d_up, lr)?;

        Ok(())
    }

    pub fn refine_attention(
        &mut self,
        x_attn_norm: Vec<f32>,
        attn_target: Vec<f32>,
        pos: usize,
        lr: f32,
    ) -> PyResult<()> {
        let q = self.q_gen.forward(x_attn_norm.clone())?;
        let k = self.k_gen.forward(x_attn_norm.clone())?;
        let v = self.v_gen.forward(x_attn_norm.clone())?;

        // We only refine based on the immediate projected output drift.
        // Complex backprop through the full attention softmax is avoided for mobile efficiency;
        // instead, we treat it as an alignment problem for each projection.

        let current_attn_out = self
            .attn
            .forward_attention(q.clone(), k.clone(), v.clone(), pos)?;
        let projected_attn = self.w_o.forward(current_attn_out.clone())?;

        let mut d_attn_out = vec![0.0f32; current_attn_out.len()];
        for i in 0..projected_attn.len() {
            let diff = projected_attn[i] - attn_target[i];
            // Since w_o is linear, we'd need its weights for exact d_attn_out.
            // Simplified: we use the diff to guide the output projection refinement.
            d_attn_out[i] = diff;
        }

        // 1. Refine Output Projection (w_o)
        self.w_o
            .refine_centroids(current_attn_out, attn_target, lr)?;

        // 2. Refine Q, K, V Projections (linear targets from Maestro)
        // Note: These usually require explicit targets from the Maestro's Q, K, V activations.
        // For now, we assume Python will call refine_with_grads or refine_centroids on them directly
        // if exact matching is needed. This method serves as the entry point for block-level IQAT.

        Ok(())
    }
}
