// =============================================================================
// moe.rs — Mixture of Experts (MoE) Router para Modelos DeepSeek / Switch
// =============================================================================

use crate::nn::linear::GenomicLinear;
use rayon::prelude::*;

#[derive(Clone)]
pub struct MoeExpert {
    pub id: usize,
    pub gate_gen: GenomicLinear,
    pub up_gen: GenomicLinear,
    pub down_gen: GenomicLinear,
}

impl MoeExpert {
    pub fn new(
        id: usize,
        gate_gen: GenomicLinear,
        up_gen: GenomicLinear,
        down_gen: GenomicLinear,
    ) -> Self {
        Self {
            id,
            gate_gen,
            up_gen,
            down_gen,
        }
    }

    pub fn forward(
        &self,
        x: &[f32],
        modulation: Option<[f32; 4]>,
        activate_rna: bool,
        act_fn: &str,
        h_scale: f32,
    ) -> Result<Vec<f32>, String> {
        let (gate, up) =
            GenomicLinear::forward_fused_2(&self.gate_gen, &self.up_gen, x, modulation)?;
        let mut ffn_out = vec![0.0f32; gate.len()];
        if act_fn == "geglu" {
            crate::compute::kernels::geglu(&gate, &up, &mut ffn_out);
        } else if act_fn == "reluglu" || act_fn == "relu_glu" {
            crate::compute::kernels::relu_glu(&gate, &up, &mut ffn_out);
        } else {
            crate::compute::kernels::swiglu_balanced(&gate, &up, &mut ffn_out, h_scale);
        }
        self.down_gen
            .forward_core(ffn_out, modulation, activate_rna)
    }
}

/// Router Top-K de Expertos (MoE) para arquitecturas DeepSeek V2/V3/R1.
#[derive(Clone)]
pub struct MoeRouter {
    pub n_routed_experts: usize,
    pub n_active_experts: usize,
    pub gate_weight: GenomicLinear,
    pub experts: Vec<MoeExpert>,
    pub shared_experts: Option<Vec<MoeExpert>>,
    pub routing_bias: Option<Vec<f32>>,
    pub norm_topk_prob: bool,
}

impl MoeRouter {
    pub fn new(
        n_routed_experts: usize,
        n_active_experts: usize,
        gate_weight: GenomicLinear,
        experts: Vec<MoeExpert>,
        shared_experts: Option<Vec<MoeExpert>>,
        routing_bias: Option<Vec<f32>>,
        norm_topk_prob: bool,
    ) -> Self {
        Self {
            n_routed_experts,
            n_active_experts,
            gate_weight,
            experts,
            shared_experts,
            routing_bias,
            norm_topk_prob,
        }
    }

    pub fn forward_moe(
        &self,
        x: &[f32],
        modulation: Option<[f32; 4]>,
        activate_rna: bool,
        act_fn: &str,
        h_scale: f32,
    ) -> Result<Vec<f32>, String> {
        // 1. Obtener logits de ruteo
        let mut router_logits =
            self.gate_weight
                .forward_core(x.to_vec(), modulation, activate_rna)?;
        if router_logits.is_empty() {
            router_logits = vec![0.0f32; self.n_routed_experts.max(self.experts.len())];
        }
        if let Some(ref bias) = self.routing_bias {
            for (r, &b) in router_logits.iter_mut().zip(bias.iter()) {
                *r += b;
            }
        }

        // 2. Selección Top-K de expertos
        let k = self
            .n_active_experts
            .min(self.experts.len())
            .min(router_logits.len());
        let mut indexed_logits: Vec<(usize, f32)> =
            router_logits.iter().copied().enumerate().collect();

        // Ordenar de mayor a menor probabilidad
        indexed_logits
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let topk = &indexed_logits[0..k];

        // 3. Normalización Softmax sobre los Top-K seleccionados
        let max_logit = topk.iter().map(|&(_, v)| v).fold(-f32::INFINITY, f32::max);
        let mut exp_weights: Vec<(usize, f32)> = topk
            .iter()
            .map(|&(idx, val)| (idx, (val - max_logit).exp()))
            .collect();
        let sum_exp: f32 = exp_weights.iter().map(|&(_, e)| e).sum();
        let inv_sum = 1.0 / (sum_exp + 1e-12);

        if self.norm_topk_prob {
            for entry in exp_weights.iter_mut() {
                entry.1 *= inv_sum;
            }
        }

        // 4. Ejecución de expertos activos y suma ponderada
        let dim = x.len();
        let mut final_out = vec![0.0f32; dim];

        let expert_outputs: Vec<(usize, f32, Result<Vec<f32>, String>)> = exp_weights
            .into_par_iter()
            .map(|(exp_idx, weight)| {
                if exp_idx < self.experts.len() {
                    let out =
                        self.experts[exp_idx].forward(x, modulation, activate_rna, act_fn, h_scale);
                    (exp_idx, weight, out)
                } else {
                    (
                        exp_idx,
                        weight,
                        Err(format!("Expert index {} out of bounds", exp_idx)),
                    )
                }
            })
            .collect();

        for (_idx, weight, res) in expert_outputs {
            let out = res?;
            for i in 0..dim.min(out.len()) {
                final_out[i] += weight * out[i];
            }
        }

        // 5. Agregar salida de expertos compartidos (Shared Experts) si existen
        if let Some(ref shared_list) = self.shared_experts {
            for shared_exp in shared_list {
                let shared_out =
                    shared_exp.forward(x, modulation, activate_rna, act_fn, h_scale)?;
                for i in 0..dim.min(shared_out.len()) {
                    final_out[i] += shared_out[i];
                }
            }
        }

        Ok(final_out)
    }
}
