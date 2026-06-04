use crate::compute::kernels::rms_norm;
use crate::core::topology::CentroidGraph;
use crate::nn::attention::GenomicAttention;
use crate::nn::linear::GenomicLinear;
use rayon::prelude::*;
use std::sync::Arc;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Bloque de Procesamiento Genómico (Pure Rust)
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct RustGenomicBlock {
    pub idx: usize,
    pub attn: GenomicAttention,
    pub q_gen: GenomicLinear,
    pub k_gen: GenomicLinear,
    pub v_gen: GenomicLinear,
    pub w_o: GenomicLinear,
    pub gate_gen: GenomicLinear,
    pub up_gen: GenomicLinear,
    pub w_down: GenomicLinear,
    pub ffn_norm: Vec<f32>,
    pub eps: f32,
    pub act_fn: String,
    pub use_genomic_norm: bool,
    pub h_scale: f32,
    pub rna_threshold: f32,
    pub topology: Option<Arc<CentroidGraph>>,
}

impl RustGenomicBlock {
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
            topology: None,
        }
    }

    pub fn forward_core(&mut self, x: Vec<f32>, pos: usize) -> Result<Vec<f32>, String> {
        // --- STAGE 1: ENTROPY ANALYSIS (Pilar 2) ---
        let entropy = crate::compute::math::calculate_activation_entropy(&x);
        let activate_rna = crate::compute::math::should_activate_rna(entropy, self.rna_threshold);

        // --- STAGE 4: DINAMIC STATE ESTIMATION ---
        let (current_state, modulation) = if let Some(ref topo) = self.topology {
            let mut sum_sq = 0.0f32;
            for &val in x.iter() {
                sum_sq += val * val;
            }
            let rms = (sum_sq / x.len() as f32 + 1e-6).sqrt();

            // Estimar estado basado en h_scale (homeostasis)
            let ratio = rms / self.h_scale.max(0.01);
            let state = if ratio < 0.5 {
                0
            } else if ratio < 1.0 {
                1
            } else if ratio < 1.5 {
                2
            } else {
                3
            };

            (
                state,
                Some(topo.get_modulation_factors(self.idx, state, 0.5)),
            )
        } else {
            (2, None)
        };

        let x_norm = if !self.attn.rmsnorm_weight.is_empty() {
            unsafe { rms_norm(&x, &self.attn.rmsnorm_weight, self.attn.eps) }
        } else {
            x.clone()
        };

        let q = self
            .q_gen
            .forward_core(x_norm.clone(), modulation, activate_rna)?;
        let k = self
            .k_gen
            .forward_core(x_norm.clone(), modulation, activate_rna)?;
        let v = self.v_gen.forward_core(x_norm, modulation, activate_rna)?;

        let attn_out = self.attn.forward_attention_core(q, k, v, pos)?;
        let projected_attn = self.w_o.forward_core(attn_out, modulation, activate_rna)?;

        let mut x_post = x;
        x_post
            .par_iter_mut()
            .zip(projected_attn.par_iter())
            .for_each(|(xi, &ai)| *xi += ai);
        let x_ffn_n = unsafe { rms_norm(&x_post, &self.ffn_norm, self.eps) };

        let gate = self
            .gate_gen
            .forward_core(x_ffn_n.clone(), modulation, activate_rna)?;
        let up = self
            .up_gen
            .forward_core(x_ffn_n, modulation, activate_rna)?;

        let mut ffn_out = vec![0.0f32; gate.len()];
        match self.act_fn.as_str() {
            "swiglu" => {
                crate::compute::kernels::swiglu_balanced(&gate, &up, &mut ffn_out, self.h_scale);
            }
            _ => {
                crate::compute::kernels::swiglu_balanced(&gate, &up, &mut ffn_out, self.h_scale);
            }
        }
        if self.use_genomic_norm {
            let rms = (ffn_out.par_iter().map(|&v| v * v).sum::<f32>() / ffn_out.len() as f32
                + self.eps)
                .sqrt();
            if rms > self.h_scale {
                let s = self.h_scale / rms;
                ffn_out.par_iter_mut().for_each(|out| *out *= s);
            }
        }
        let projected_ffn = self
            .w_down
            .forward_core(ffn_out, modulation, activate_rna)?;
        let mut final_out = x_post;
        final_out
            .par_iter_mut()
            .zip(projected_ffn.par_iter())
            .for_each(|(fi, &pi)| *fi += pi);

        // Inyectar Bias Relacional al final del bloque
        if let Some(ref topo) = self.topology {
            topo.apply_relational_bias(self.idx, current_state, &mut final_out, 0.5);
        }

        Ok(final_out)
    }

    pub fn clear_cache_core(&mut self) {
        self.attn.clear_cache_core();
    }

    pub fn refine_with_grads_core(
        &mut self,
        x: Vec<f32>,
        d_hidden: Vec<f32>,
        pos: usize,
        lr: f32,
    ) -> Result<Vec<f32>, String> {
        // --- STAGE 4: DINAMIC STATE ESTIMATION (Training consistency) ---
        let (_current_state, modulation) = if let Some(ref topo) = self.topology {
            let mut sum_sq = 0.0f32;
            for &val in x.iter() {
                sum_sq += val * val;
            }
            let rms = (sum_sq / x.len() as f32 + 1e-6).sqrt();
            let ratio = rms / self.h_scale.max(0.01);
            let state = if ratio < 0.5 {
                0
            } else if ratio < 1.0 {
                1
            } else if ratio < 1.5 {
                2
            } else {
                3
            };
            (
                state,
                Some(topo.get_modulation_factors(self.idx, state, 0.5)),
            )
        } else {
            (2, None)
        };

        let x_norm = if !self.attn.rmsnorm_weight.is_empty() {
            unsafe { rms_norm(&x, &self.attn.rmsnorm_weight, self.attn.eps) }
        } else {
            x.clone()
        };
        let q = self.q_gen.forward_core(x_norm.clone(), modulation, true)?;
        let k = self.k_gen.forward_core(x_norm.clone(), modulation, true)?;
        let v = self.v_gen.forward_core(x_norm.clone(), modulation, true)?;

        let attn_out = self.attn.forward_attention_core(q, k, v, pos)?;
        let proj_attn = self.w_o.forward_core(attn_out.clone(), modulation, true)?;

        let mut x_post_attn = x.clone();
        for i in 0..x.len() {
            x_post_attn[i] += proj_attn[i];
        }
        let x_ffn_n = unsafe { rms_norm(&x_post_attn, &self.ffn_norm, self.eps) };
        let gate = self
            .gate_gen
            .forward_core(x_ffn_n.clone(), modulation, true)?;
        let up = self
            .up_gen
            .forward_core(x_ffn_n.clone(), modulation, true)?;

        let d_ffn_out = self.w_down.backward_core(d_hidden.clone())?;
        let mut d_gate = vec![0.0f32; gate.len()];
        let mut d_up = vec![0.0f32; up.len()];
        for i in 0..gate.len() {
            let g = gate[i];
            let u = up[i];
            let s = 1.0 / (1.0 + (-g).exp());
            let silu_p = s * (1.0 + g * (1.0 - s));
            d_gate[i] = d_ffn_out[i] * silu_p * u;
            d_up[i] = d_ffn_out[i] * (g * s);
        }
        self.w_down
            .refine_with_grads_core(vec![0.0; gate.len()], d_hidden.clone(), lr)?;
        self.gate_gen
            .refine_with_grads_core(x_ffn_n.clone(), d_gate, lr)?;
        self.up_gen.refine_with_grads_core(x_ffn_n, d_up, lr)?;
        let d_ffn_in = self.gate_gen.backward_core(vec![0.0; gate.len()])?;
        let mut d_x_post = d_hidden;
        for i in 0..d_x_post.len() {
            d_x_post[i] += d_ffn_in[i];
        }
        let d_attn_out = self.w_o.backward_core(d_x_post.clone())?;
        self.w_o
            .refine_with_grads_core(attn_out, d_x_post.clone(), lr)?;
        let d_attn_in = self.v_gen.backward_core(d_attn_out)?;
        let mut d_x = d_x_post;
        for i in 0..d_x.len() {
            d_x[i] += d_attn_in[i];
        }
        Ok(d_x)
    }

    pub fn mutate_homeostasis_core(&mut self, scale: f32) -> Result<f32, String> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let delta = rng.gen_range(-scale..scale);
        self.h_scale += delta;
        self.h_scale = self.h_scale.clamp(0.01, 10.0);
        Ok(delta)
    }
}

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
            topology: None,
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
}
