use crate::compute::kernels::rms_norm;
use crate::nn::attention::GenomicAttention;
use crate::nn::linear::GenomicLinear;
use pyo3::prelude::*;
use rayon::prelude::*;

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
    #[pyo3(get)]
    pub h_scale: f32,
}

#[pymethods]
impl RustGenomicBlock {
    #[new]
    #[pyo3(signature = (idx, attn, q_gen, k_gen, v_gen, w_o, gate_gen, up_gen, w_down, ffn_norm, eps, act_fn = "swiglu".to_string(), use_genomic_norm = false, h_scale = 1.0))]
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
        h_scale: f32,
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

        // Vectorized residual connection
        let mut x_post_attn = x;
        x_post_attn.par_iter_mut()
            .zip(projected_attn.par_iter())
            .for_each(|(xi, &ai)| *xi += ai);

        // --- 2. FeedForward Block ---
        let x_ffn_norm = unsafe { rms_norm(&x_post_attn, &self.ffn_norm, self.eps) };

        let gate = self.gate_gen.forward(x_ffn_norm.clone())?;
        let up = self.up_gen.forward(x_ffn_norm)?;

        let mut ffn_out = vec![0.0f32; gate.len()];
        match self.act_fn.as_str() {
            "swiglu" => {
                ffn_out.par_iter_mut()
                    .zip(gate.par_iter())
                    .zip(up.par_iter())
                    .for_each(|((out, &g), &u)| {
                        // Estabilización de SwiGLU (Silu gating)
                        let g_safe = g.max(-88.0).min(88.0);
                        let sigmoid = if g_safe >= 0.0 {
                            1.0 / (1.0 + (-g_safe).exp())
                        } else {
                            let ex = g_safe.exp();
                            ex / (1.0 + ex)
                        };
                        let silu = g * sigmoid;
                        // Clamping para evitar deriva semántica destructiva en 2-bits
                        *out = (silu * u).clamp(-128.0, 128.0);
                    });
            }
            "geglu" => {
                ffn_out.par_iter_mut()
                    .zip(gate.par_iter())
                    .zip(up.par_iter())
                    .for_each(|((out, &g), &u)| {
                        // Estabilización de GeGLU
                        let g_safe = g.clamp(-20.0, 20.0);
                        let gelu = 0.5 * g_safe * (1.0 + ((0.79788456 * (g_safe + 0.044715 * g_safe * g_safe * g_safe)).tanh()));
                        *out = (gelu * u).clamp(-128.0, 128.0);
                    });
            }
            "relu" => {
                ffn_out.par_iter_mut()
                    .zip(gate.par_iter())
                    .zip(up.par_iter())
                    .for_each(|((out, &g), &u)| {
                        *out = (g.max(0.0) * u).clamp(-128.0, 128.0);
                    });
            }
            _ => {
                ffn_out.par_iter_mut()
                    .zip(gate.par_iter())
                    .zip(up.par_iter())
                    .for_each(|((out, &g), &u)| {
                        let g_safe = g.max(-88.0).min(88.0);
                        let sigmoid = if g_safe >= 0.0 {
                            1.0 / (1.0 + (-g_safe).exp())
                        } else {
                            let ex = g_safe.exp();
                            ex / (1.0 + ex)
                        };
                        let silu = g * sigmoid;
                        *out = (silu * u).clamp(-128.0, 128.0);
                    });
            }
        }

        // Optional GenomicNorm to stabilize variance before the final down projection
        if self.use_genomic_norm {
            let n = ffn_out.len();
            let sum_sq: f32 = ffn_out.par_iter().map(|&v| v * v).sum();
            let inv_rms = self.h_scale / (sum_sq / n as f32 + self.eps).sqrt();
            ffn_out.par_iter_mut().for_each(|out| *out *= inv_rms);
        }

        let projected_ffn = self.w_down.forward(ffn_out)?;

        // Vectorized residual connection
        let mut final_out = x_post_attn;
        final_out.par_iter_mut()
            .zip(projected_ffn.par_iter())
            .for_each(|(fi, &pi)| *fi += pi);

        Ok(final_out)
    }

    pub fn clear_cache(&mut self) -> PyResult<()> {
        self.attn.clear_cache()
    }

    pub fn mutate_homeostasis(&mut self, scale: f32) -> PyResult<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let delta = rng.gen_range(-scale..scale);
        self.h_scale += delta;
        self.h_scale = self.h_scale.max(0.01).min(10.0); // Límites de seguridad
        Ok(delta)
    }

    pub fn refine_ffn(
        &mut self,
        x_ffn_norm: Vec<f32>,
        ffn_target: Vec<f32>,
        lr: f32,
    ) -> PyResult<()> {
        let gate = self.gate_gen.forward(x_ffn_norm.clone())?;
        let up = self.up_gen.forward(x_ffn_norm.clone())?;

        let n = gate.len();
        // Vectorized refinement calculation
        let (d_gate, d_up): (Vec<f32>, Vec<f32>) = (0..n).into_par_iter().map(|i| {
            let g = gate[i];
            let u = up[i];
            let target = ffn_target[i];
            
            match self.act_fn.as_str() {
                "swiglu" => {
                    let g_safe = g.max(-88.0).min(88.0);
                    let s = if g_safe >= 0.0 {
                        1.0 / (1.0 + (-g_safe).exp())
                    } else {
                        let ex = g_safe.exp();
                        ex / (1.0 + ex)
                    };
                    let current = (g * s * u).clamp(-128.0, 128.0);
                    let diff = current - target;
                    let silu_p = s * (1.0 + g * (1.0 - s));
                    (diff * silu_p * u, diff * (g * s))
                }
                "geglu" => {
                    let tanh_inner = 0.79788456 * (g + 0.044715 * g * g * g);
                    let tanh_val = tanh_inner.tanh();
                    let current = 0.5 * g * (1.0 + tanh_val) * u;
                    let diff = current - target;
                    let gelu_p = 0.5 * (1.0 + tanh_val) + 0.5 * g * (1.0 - tanh_val * tanh_val) * (0.79788456 * (1.0 + 3.0 * 0.044715 * g * g));
                    (diff * gelu_p * u, diff * (0.5 * g * (1.0 + tanh_val)))
                }
                "relu" => {
                    let current = g.max(0.0) * u;
                    let diff = current - target;
                    if g > 0.0 { (diff * u, diff * g) } else { (0.0, 0.0) }
                }
                _ => {
                    let s = 1.0 / (1.0 + (-g).exp());
                    let current = g * s * u;
                    let diff = current - target;
                    let silu_p = s * (1.0 + g * (1.0 - s));
                    (diff * silu_p * u, diff * (g * s))
                }
            }
        }).unzip();

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
