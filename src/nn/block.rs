use crate::compute::kernels::rms_norm;
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
    #[pyo3(get)]
    pub act_fn: String,
    #[pyo3(get)]
    pub use_genomic_norm: bool,
}

#[pymethods]
impl RustGenomicBlock {
    #[new]
    #[pyo3(signature = (idx, attn, q_gen, k_gen, v_gen, w_o, gate_gen, up_gen, w_down, ffn_norm, eps, act_fn = "swiglu".to_string(), use_genomic_norm = false))]
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
        act_fn: String,
        use_genomic_norm: bool,
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

        // --- 2. FeedForward Block ---
        let x_ffn_norm = unsafe { rms_norm(&x_post_attn, &self.ffn_norm, self.eps) };

        let gate = self.gate_gen.forward(x_ffn_norm.clone())?;
        let up = self.up_gen.forward(x_ffn_norm)?;

        let mut ffn_out = vec![0.0f32; gate.len()];
        match self.act_fn.as_str() {
            "swiglu" => {
                for i in 0..gate.len() {
                    let g = gate[i];
                    let silu = g / (1.0 + (-g).exp());
                    ffn_out[i] = silu * up[i];
                }
            }
            "geglu" => {
                for i in 0..gate.len() {
                    let g = gate[i];
                    let gelu = 0.5 * g * (1.0 + ((0.79788456 * (g + 0.044715 * g * g * g)).tanh()));
                    ffn_out[i] = gelu * up[i];
                }
            }
            "relu" => {
                for i in 0..gate.len() {
                    ffn_out[i] = gate[i].max(0.0) * up[i];
                }
            }
            _ => {
                for i in 0..gate.len() {
                    let g = gate[i];
                    let silu = g / (1.0 + (-g).exp());
                    ffn_out[i] = silu * up[i];
                }
            }
        }

        // Optional GenomicNorm to stabilize variance before the final down projection
        if self.use_genomic_norm {
            let unit_weights = vec![1.0f32; ffn_out.len()];
            ffn_out = unsafe { rms_norm(&ffn_out, &unit_weights, self.eps) };
        }

        let projected_ffn = self.w_down.forward(ffn_out)?;

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

    pub fn refine_ffn(
        &mut self,
        x_ffn_norm: Vec<f32>,
        ffn_target: Vec<f32>,
        lr: f32,
    ) -> PyResult<()> {
        let gate = self.gate_gen.forward(x_ffn_norm.clone())?;
        let up = self.up_gen.forward(x_ffn_norm.clone())?;

        let mut d_gate = vec![0.0f32; gate.len()];
        let mut d_up = vec![0.0f32; gate.len()];

        for i in 0..gate.len() {
            let g = gate[i];
            let u = up[i];
            let diff;
            let silu_p;
            let gelu_p;

            match self.act_fn.as_str() {
                "swiglu" => {
                    let s = 1.0 / (1.0 + (-g).exp());
                    let current = g * s * u;
                    diff = current - ffn_target[i];
                    silu_p = s * (1.0 + g * (1.0 - s));
                    d_gate[i] = diff * silu_p * u;
                    d_up[i] = diff * (g * s);
                }
                "geglu" => {
                    // GELU approx derivative
                    let tanh_inner = 0.79788456 * (g + 0.044715 * g * g * g);
                    let tanh_val = tanh_inner.tanh();
                    let current = 0.5 * g * (1.0 + tanh_val) * u;
                    diff = current - ffn_target[i];
                    gelu_p = 0.5 * (1.0 + tanh_val) + 0.5 * g * (1.0 - tanh_val * tanh_val) * (0.79788456 * (1.0 + 3.0 * 0.044715 * g * g));
                    d_gate[i] = diff * gelu_p * u;
                    d_up[i] = diff * (0.5 * g * (1.0 + tanh_val));
                }
                "relu" => {
                    let current = gate[i].max(0.0) * u;
                    diff = current - ffn_target[i];
                    d_gate[i] = if g > 0.0 { diff * u } else { 0.0 };
                    d_up[i] = if g > 0.0 { diff * g } else { 0.0 };
                }
                _ => {
                    // SwiGLU default for refinement too
                    let s = 1.0 / (1.0 + (-g).exp());
                    let current = g * s * u;
                    diff = current - ffn_target[i];
                    silu_p = s * (1.0 + g * (1.0 - s));
                    d_gate[i] = diff * silu_p * u;
                    d_up[i] = diff * (g * s);
                }
            }
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
