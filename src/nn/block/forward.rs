// =============================================================================
// forward — Forward del bloque y limpieza de caché
// =============================================================================
use rayon::prelude::*;

use crate::nn::block::RustGenomicBlock;
use crate::nn::linear::GenomicLinear;

impl RustGenomicBlock {
    pub fn forward_core(&mut self, x: Vec<f32>, pos: usize) -> Result<Vec<f32>, String> {
        if x.iter().any(|v| v.is_nan()) {
            return Err("Input x is NaN".into());
        }
        // --- STAGE 1: ENTROPY ANALYSIS (Pilar 2) ---
        let entropy = crate::compute::math::calculate_activation_entropy(&x);
        let activate_rna = crate::compute::math::should_activate_rna(entropy, self.rna_threshold);

        let x_norm = if !self.attn.rmsnorm_weight.is_empty() {
            let res = unsafe {
                crate::compute::kernels::rms_norm(&x, &self.attn.rmsnorm_weight, self.attn.eps)
            };
            if res.iter().any(|v| v.is_nan()) {
                return Err("NaN after attn rms_norm".into());
            }
            res
        } else {
            x.clone()
        };

        // --- STAGE 4: DINAMIC STATE ESTIMATION ---
        let (current_state, modulation) = if let Some(ref topo) = self.topology {
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

        let (q, k, v) = if let Some(ref qkv_gen) = self.fused_qkv {
            let qkv_out = qkv_gen.forward_core(x_norm, modulation, activate_rna)?;
            let q_dim = self.attn.n_head * self.attn.head_dim;
            let kv_dim = self.attn.n_head_kv * self.attn.head_dim;
            let q = qkv_out[0..q_dim].to_vec();
            let k = qkv_out[q_dim..q_dim + kv_dim].to_vec();
            let v = qkv_out[q_dim + kv_dim..q_dim + 2 * kv_dim].to_vec();
            (q, k, v)
        } else {
            GenomicLinear::forward_fused_3(
                &self.q_gen,
                &self.k_gen,
                &self.v_gen,
                &x_norm,
                modulation,
            )?
        };

        let attn_out = self.attn.forward_attention_core(q, k, v, pos)?;
        let projected_attn = self.w_o.forward_core(attn_out, modulation, activate_rna)?;

        let mut x_post = x;
        x_post
            .par_iter_mut()
            .zip(projected_attn.par_iter())
            .for_each(|(xi, &ai)| *xi += ai);

        let x_ffn_n =
            unsafe { crate::compute::kernels::rms_norm(&x_post, &self.ffn_norm, self.eps) };

        let (gate, up) = if let Some(ref gate_up_gen) = self.fused_gate_up {
            let gate_up_out = gate_up_gen.forward_core(x_ffn_n, modulation, activate_rna)?;
            let ffn_dim = gate_up_out.len() / 2;
            let gate = gate_up_out[0..ffn_dim].to_vec();
            let up = gate_up_out[ffn_dim..2 * ffn_dim].to_vec();
            (gate, up)
        } else {
            GenomicLinear::forward_fused_2(&self.gate_gen, &self.up_gen, &x_ffn_n, modulation)?
        };

        if up.iter().any(|v| v.is_nan()) {
            return Err("NaN in up".into());
        }

        let mut ffn_out = vec![0.0f32; gate.len()];
        match self.act_fn.as_str() {
            "swiglu" => {
                crate::compute::kernels::swiglu_balanced(&gate, &up, &mut ffn_out, self.h_scale);
            }
            _ => {
                crate::compute::kernels::swiglu_balanced(&gate, &up, &mut ffn_out, self.h_scale);
            }
        }

        if ffn_out.iter().any(|v| v.is_nan()) {
            return Err("NaN in ffn_out".into());
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

        if projected_ffn.iter().any(|v| v.is_nan()) {
            return Err("NaN in projected_ffn".into());
        }

        let mut final_out = x_post;
        final_out
            .par_iter_mut()
            .zip(projected_ffn.par_iter())
            .for_each(|(fi, &pi)| *fi += pi);

        if final_out.iter().any(|v| v.is_nan()) {
            return Err("NaN after projected_ffn addition".into());
        }

        // --- STAGE 5: TOROIDAL CONFINEMENT (K-WTA Lateral Inhibition) ---
        // Filtramos el ruido de fondo para que solo la señal en resonancia sobreviva.
        // Esto evita la acumulación de entropía (deriva semántica) detectada en Phase 2.
        if self.k_wta_ratio > 0.0 && self.k_wta_ratio < 1.0 {
            let ratio = self.k_wta_ratio;
            let k = ((final_out.len() as f32 * ratio) as usize).max(1);
            crate::compute::kernels::lateral_inhibition_kwta(&mut final_out, k);
        }

        // Inyectar Bias Relacional al final del bloque
        if let Some(ref topo) = self.topology {
            topo.apply_relational_bias(self.idx, current_state, &mut final_out, 0.5);
        }

        Ok(final_out)
    }

    pub fn clear_cache_core(&mut self) {
        self.attn.clear_cache_core();
    }
}
