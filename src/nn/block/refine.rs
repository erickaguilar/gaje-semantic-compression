// =============================================================================
// refine — Refine con gradientes (FFN y atención) de RustGenomicBlock
// =============================================================================
use crate::nn::block::RustGenomicBlock;

impl RustGenomicBlock {
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
            unsafe {
                crate::compute::kernels::rms_norm(&x, &self.attn.rmsnorm_weight, self.attn.eps)
            }
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
        let x_ffn_n =
            unsafe { crate::compute::kernels::rms_norm(&x_post_attn, &self.ffn_norm, self.eps) };
        let gate = self
            .gate_gen
            .forward_core(x_ffn_n.clone(), modulation, true)?;
        let up = self
            .up_gen
            .forward_core(x_ffn_n.clone(), modulation, true)?;

        let d_ffn_out = self.w_down.backward_core(d_hidden.clone())?;
        let mut d_gate = vec![0.0f32; gate.len()];
        let mut d_up = vec![0.0f32; up.len()];
        let mut ffn_out = vec![0.0f32; gate.len()];
        for i in 0..gate.len() {
            let g = gate[i];
            let u = up[i];
            let s = 1.0 / (1.0 + (-g).exp());
            let silu_val = g * s;
            let silu_p = s * (1.0 + g * (1.0 - s));
            ffn_out[i] = silu_val * u;
            d_gate[i] = d_ffn_out[i] * silu_p * u;
            d_up[i] = d_ffn_out[i] * silu_val;
        }
        self.w_down
            .refine_with_grads_core(ffn_out, d_hidden.clone(), lr)?;
        self.gate_gen
            .refine_with_grads_core(x_ffn_n.clone(), d_gate.clone(), lr)?;
        self.up_gen.refine_with_grads_core(x_ffn_n, d_up.clone(), lr)?;
        let d_ffn_gate = self.gate_gen.backward_core(d_gate)?;
        let d_ffn_up = self.up_gen.backward_core(d_up)?;
        let mut d_x_post = d_hidden;
        for i in 0..d_x_post.len() {
            d_x_post[i] += d_ffn_gate[i] + d_ffn_up[i];
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

    pub fn refine_ffn_core(
        &mut self,
        x_norm: Vec<f32>,
        target: Vec<f32>,
        lr: f32,
    ) -> Result<(), String> {
        let modulation = None;
        let gate = self
            .gate_gen
            .forward_core(x_norm.clone(), modulation, true)?;
        let up = self.up_gen.forward_core(x_norm.clone(), modulation, true)?;

        let mut d_gate = vec![0.0f32; gate.len()];
        let mut d_up = vec![0.0f32; up.len()];
        for i in 0..gate.len() {
            let g = gate[i];
            let u = up[i];
            let sig = 1.0 / (1.0 + (-g).exp());
            let silu_val = g * sig;
            let pred = silu_val * u;
            let diff = pred - target[i];

            let silu_p = sig * (1.0 + g * (1.0 - sig));
            d_gate[i] = diff * u * silu_p;
            d_up[i] = diff * silu_val;
        }

        self.gate_gen
            .refine_with_grads_core(x_norm.clone(), d_gate, lr)?;
        self.up_gen.refine_with_grads_core(x_norm, d_up, lr)?;
        Ok(())
    }

    pub fn refine_attention_core(
        &mut self,
        x_norm: Vec<f32>,
        target: Vec<f32>,
        pos: usize,
        lr: f32,
    ) -> Result<(), String> {
        let modulation = None;
        let q = self.q_gen.forward_core(x_norm.clone(), modulation, true)?;
        let k = self.k_gen.forward_core(x_norm.clone(), modulation, true)?;
        let v = self.v_gen.forward_core(x_norm.clone(), modulation, true)?;

        let attn_out = self.attn.forward_attention_core(q, k, v, pos)?;
        let pred = self.w_o.forward_core(attn_out.clone(), modulation, true)?;

        let mut grads = vec![0.0f32; pred.len()];
        for i in 0..pred.len() {
            grads[i] = pred[i] - target[i];
        }

        self.w_o.refine_with_grads_core(attn_out, grads, lr)?;
        Ok(())
    }
}
