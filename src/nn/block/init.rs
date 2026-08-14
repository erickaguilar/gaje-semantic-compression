// =============================================================================
// init — Construcción y homeostasis de RustGenomicBlock
// =============================================================================
use crate::nn::attention::GenomicAttention;
use crate::nn::block::RustGenomicBlock;
use crate::nn::linear::GenomicLinear;

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
            k_wta_ratio: 0.0,
            topology: None,
            fused_qkv: None,
            fused_gate_up: None,
        }
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