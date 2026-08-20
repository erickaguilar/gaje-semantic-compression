// =============================================================================
// cache — ForwardCache: activaciones guardadas durante el forward original
// =============================================================================
//
// Este módulo implementa backprop estándar con caché de activaciones (el
// diseño que sustituye al doble-forward descartado en Fase 4). El forward
// guarda las activaciones intermedias de cada bloque y el backward las
// consume tal cual (SIN re-forward), garantizando que los gradientes
// correspondan al grafo computacional real que produjo la salida.
use crate::nn::block::RustGenomicBlock;
use crate::nn::linear::GenomicLinear;

/// Activaciones de un bloque necesarias para su backward.
pub struct BlockCache {
    /// Entrada al bloque (pre-norm).
    pub x: Vec<f32>,
    /// Salida del rmsnorm de atención (entrada a qkv).
    pub x_norm: Vec<f32>,
    /// q con RoPE aplicado (para el backward de atención).
    pub q_rope: Vec<f32>,
    /// Pesos softmax por (head, token) del paso actual.
    pub softmax_weights: Vec<f32>,
    /// Salida de atención (entrada a w_o).
    pub attn_out: Vec<f32>,
    /// x + proj_attn (residual de atención).
    pub x_post_attn: Vec<f32>,
    /// Salida del rmsnorm de FFN (entrada a gate/up).
    pub x_ffn_n: Vec<f32>,
    /// Activación gate (SwiGLU).
    pub gate: Vec<f32>,
    /// Activación up (SwiGLU).
    pub up: Vec<f32>,
    /// swiglu(gate, up), entrada a w_down.
    pub ffn_out: Vec<f32>,
}

impl RustGenomicBlock {
    /// Forward con caché de activaciones. Devuelve (salida, caché).
    pub fn forward_core_cached(
        &mut self,
        x: Vec<f32>,
        pos: usize,
    ) -> Result<(Vec<f32>, BlockCache), String> {
        let modulation = None;
        let x_norm = if !self.attn.rmsnorm_weight.is_empty() {
            unsafe {
                crate::compute::kernels::rms_norm(&x, &self.attn.rmsnorm_weight, self.attn.eps)
            }
        } else {
            x.clone()
        };
        let (q, k, v) = if let Some(ref qkv) = self.fused_qkv {
            let qkv_out = qkv.forward_core(x_norm.clone(), modulation, true)?;
            let q_dim = self.attn.n_head * self.attn.head_dim;
            let kv_dim = self.attn.n_head_kv * self.attn.head_dim;
            let q = qkv_out[0..q_dim].to_vec();
            let k = qkv_out[q_dim..q_dim + kv_dim].to_vec();
            let v = qkv_out[q_dim + kv_dim..q_dim + 2 * kv_dim].to_vec();
            (q, k, v)
        } else {
            GenomicLinear::forward_fused_3(&self.q_gen, &self.k_gen, &self.v_gen, &x_norm, modulation)?
        };
        let (attn_out, softmax_weights, q_rope) =
            self.attn.forward_attention_cached(q, k, v, pos)?;
        let proj_attn = self.w_o.forward_core(attn_out.clone(), modulation, true)?;

        let mut x_post_attn = x.clone();
        for i in 0..x.len() {
            x_post_attn[i] += proj_attn[i];
        }
        let x_ffn_n =
            unsafe { crate::compute::kernels::rms_norm(&x_post_attn, &self.ffn_norm, self.eps) };
        let (gate, up) = if let Some(ref gu) = self.fused_gate_up {
            let gu_out = gu.forward_core(x_ffn_n.clone(), modulation, true)?;
            let ffn_dim = gu_out.len() / 2;
            let gate = gu_out[0..ffn_dim].to_vec();
            let up = gu_out[ffn_dim..2 * ffn_dim].to_vec();
            (gate, up)
        } else {
            GenomicLinear::forward_fused_2(&self.gate_gen, &self.up_gen, &x_ffn_n, modulation)?
        };

        let mut ffn_out = vec![0.0f32; gate.len()];
        crate::compute::kernels::swiglu_balanced(&gate, &up, &mut ffn_out, self.h_scale);
        let proj_ffn = self.w_down.forward_core(ffn_out.clone(), modulation, true)?;

        let mut final_out = x_post_attn.clone();
        for i in 0..final_out.len() {
            final_out[i] += proj_ffn[i];
        }

        let cache = BlockCache {
            x,
            x_norm,
            q_rope,
            softmax_weights,
            attn_out,
            x_post_attn,
            x_ffn_n,
            gate,
            up,
            ffn_out,
        };
        Ok((final_out, cache))
    }

    /// Backward con caché de activaciones (sin re-forward). Devuelve dL/dx.
    /// Aplica grad-clipping global (`gclip`) y lr del bloque.
    pub fn backward_core_cached(
        &mut self,
        c: &BlockCache,
        mut d_out: Vec<f32>,
        lr: f32,
        gclip: f32,
    ) -> Result<Vec<f32>, String> {
        let clip = |v: &mut Vec<f32>| {
            if gclip > 0.0 {
                for x in v.iter_mut() {
                    *x = x.clamp(-gclip, gclip);
                }
            }
        };
        clip(&mut d_out);

        // ---- FFN (SwiGLU) ----
        let d_ffn_out = self.w_down.backward_core(d_out.clone())?;
        let mut d_gate = vec![0.0f32; c.gate.len()];
        let mut d_up = vec![0.0f32; c.up.len()];
        for i in 0..c.gate.len() {
            let g = c.gate[i];
            let u = c.up[i];
            let s = 1.0 / (1.0 + (-g).exp());
            let silu = g * s;
            let silu_p = s * (1.0 + g * (1.0 - s));
            d_gate[i] = d_ffn_out[i] * silu_p * u;
            d_up[i] = d_ffn_out[i] * silu;
        }
        clip(&mut d_gate);
        clip(&mut d_up);
        self.w_down
            .refine_with_grads_core(c.ffn_out.clone(), d_out.clone(), lr)?;
        if let Some(gu) = &mut self.fused_gate_up {
            let mut d_gup = d_gate.clone();
            d_gup.extend(&d_up);
            gu.refine_with_grads_core(c.x_ffn_n.clone(), d_gup, lr)?;
        } else {
            self.gate_gen
                .refine_with_grads_core(c.x_ffn_n.clone(), d_gate.clone(), lr)?;
            self.up_gen
                .refine_with_grads_core(c.x_ffn_n.clone(), d_up.clone(), lr)?;
        }

        // Gradiente w.r.t. x_ffn_n -> ffn_norm backward -> x_post_attn
        let mut d_x_ffn_n = if let Some(gu) = &mut self.fused_gate_up {
            let mut d_gup = d_gate.clone();
            d_gup.extend(&d_up);
            gu.backward_core(d_gup)?
        } else {
            let d_xffn_g = self.gate_gen.backward_core(d_gate)?;
            let d_xffn_u = self.up_gen.backward_core(d_up)?;
            let mut v = vec![0.0f32; c.x_ffn_n.len()];
            for i in 0..v.len() {
                v[i] = d_xffn_g[i] + d_xffn_u[i];
            }
            v
        };
        clip(&mut d_x_ffn_n);
        let d_x_post_attn = crate::compute::kernels::rms_norm_backward(
            &c.x_post_attn,
            &d_x_ffn_n,
            &self.ffn_norm,
            self.eps,
        );

        // final_out = x_post_attn + proj_ffn  =>  d_x_post = d_out + ffn_norm_backward
        let mut d_x_post = vec![0.0f32; c.x_post_attn.len()];
        for i in 0..d_x_post.len() {
            d_x_post[i] = d_out[i] + d_x_post_attn[i];
        }

        // ---- Atención ----
        let d_attn_out = self.w_o.backward_core(d_x_post.clone())?;
        self.w_o
            .refine_with_grads_core(c.attn_out.clone(), d_x_post.clone(), lr)?;
        let (d_q, d_k, d_v) =
            self.attn
                .backward_attention_core(&d_attn_out, &c.q_rope, &c.softmax_weights)?;
        let mut d_x_norm = if let Some(qkv) = &mut self.fused_qkv {
            let mut d_qkv = d_q.clone();
            d_qkv.extend(&d_k);
            d_qkv.extend(&d_v);
            qkv.backward_core(d_qkv)?
        } else {
            let d_xn_q = self.q_gen.backward_core(d_q)?;
            let d_xn_k = self.k_gen.backward_core(d_k)?;
            let d_xn_v = self.v_gen.backward_core(d_v)?;
            let mut v = vec![0.0f32; c.x.len()];
            for i in 0..v.len() {
                v[i] = d_xn_q[i] + d_xn_k[i] + d_xn_v[i];
            }
            v
        };
        clip(&mut d_x_norm);

        // d_x_norm es el gradiente w.r.t. x_norm (rmsnorm(x)); hay que
        // atravesar el backward del rmsnorm de atención para llegar a x.
        let d_x_attn = crate::compute::kernels::rms_norm_backward(
            &c.x,
            &d_x_norm,
            &self.attn.rmsnorm_weight,
            self.attn.eps,
        );
        let mut d_x = vec![0.0f32; c.x.len()];
        for i in 0..d_x.len() {
            d_x[i] = d_x_post[i] + d_x_attn[i];
        }
        Ok(d_x)
    }
}