//! 🧪 Test de Integración y Paridad para MLA (Multi-Head Latent Attention) y MoE (Mixture of Experts)
//!
//! Valida el comportamiento de MlaAttention y MoeRouter según DEEPSEEK_GEMMA_SUPPORT_PLAN.md.

use _impl::nn::attention::MlaAttention;
use _impl::nn::linear::GenomicLinear;
use _impl::nn::moe::{MoeExpert, MoeRouter};
use _impl::nn::block::RustGenomicBlock;
use _impl::nn::attention::GenomicAttention;

#[test]
fn test_mla_attention_latent_cache_and_rope() {
    let n_head = 4;
    let q_lora_rank = 16;
    let kv_lora_rank = 16;
    let qk_nope_head_dim = 16;
    let qk_rope_head_dim = 8;
    let v_head_dim = 16;

    let mut mla = MlaAttention::new(
        n_head,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        vec![1.0; 64],
        1e-5,
        10000.0,
        "split".to_string(),
    );

    // Mock inputs for token 0
    let q_nope = vec![0.1f32; n_head * qk_nope_head_dim];
    let q_rope = vec![0.05f32; n_head * qk_rope_head_dim];
    let kv_latent = vec![0.2f32; kv_lora_rank];
    let k_rope = vec![0.05f32; qk_rope_head_dim];

    let out0 = mla.forward_mla_core(&q_nope, &q_rope, kv_latent, k_rope, 0).expect("MLA forward failed");
    assert_eq!(out0.len(), n_head * v_head_dim);
    assert_eq!(mla.cache_len(), 1);

    for &v in &out0 {
        assert!(!v.is_nan());
        assert!(!v.is_infinite());
    }

    // Forward for token 1 (should use cached token 0 and token 1)
    let kv_latent_1 = vec![0.15f32; kv_lora_rank];
    let k_rope_1 = vec![0.04f32; qk_rope_head_dim];
    let out1 = mla.forward_mla_core(&q_nope, &q_rope, kv_latent_1, k_rope_1, 1).expect("MLA forward 1 failed");
    assert_eq!(out1.len(), n_head * v_head_dim);
    assert_eq!(mla.cache_len(), 2);

    for &v in &out1 {
        assert!(!v.is_nan());
        assert!(!v.is_infinite());
    }
}

#[test]
fn test_moe_router_topk_routing() {
    let dim = 32;
    let n_routed = 4;
    let n_active = 2;

    let gate_weight = GenomicLinear::empty(); // Fallback identidad o ceros
    let mut experts = Vec::new();
    for i in 0..n_routed {
        experts.push(MoeExpert::new(
            i,
            GenomicLinear::empty(),
            GenomicLinear::empty(),
            GenomicLinear::empty(),
        ));
    }

    let router = MoeRouter::new(
        n_routed,
        n_active,
        gate_weight,
        experts,
        None,
        None,
        true,
    );

    let x = vec![0.1f32; dim];
    let out = router.forward_moe(&x, None, false, "swiglu", 1.0).expect("MoE forward failed");
    assert_eq!(out.len(), dim);
    for &v in &out {
        assert!(!v.is_nan());
        assert!(!v.is_infinite());
    }
}

#[test]
fn test_block_with_mla_and_moe() {
    let dim = 32;
    let n_head = 4;
    let head_dim = 8;

    let attn = GenomicAttention::new(
        n_head,
        n_head,
        head_dim,
        vec![1.0; dim],
        1e-5,
        10000.0,
        "split".to_string(),
    );

    let mut block = RustGenomicBlock::new(
        0,
        attn,
        GenomicLinear::empty(),
        GenomicLinear::empty(),
        GenomicLinear::empty(),
        GenomicLinear::empty(),
        GenomicLinear::empty(),
        GenomicLinear::empty(),
        GenomicLinear::empty(),
        vec![1.0; dim],
        1e-5,
        "swiglu".to_string(),
        false,
        1.0,
        0.5,
    );

    // Adjuntar MLA
    block.mla = Some(MlaAttention::new(
        n_head,
        16,
        16,
        head_dim,
        head_dim,
        head_dim,
        vec![1.0; dim],
        1e-5,
        10000.0,
        "split".to_string(),
    ));

    // Adjuntar MoE
    let mut experts = Vec::new();
    for i in 0..4 {
        experts.push(MoeExpert::new(
            i,
            GenomicLinear::empty(),
            GenomicLinear::empty(),
            GenomicLinear::empty(),
        ));
    }
    block.moe = Some(MoeRouter::new(
        4,
        2,
        GenomicLinear::empty(),
        experts,
        None,
        None,
        true,
    ));

    let x = vec![0.5f32; dim];
    let out = block.forward_core(x, 0).expect("Block forward with MLA & MoE failed");
    assert_eq!(out.len(), dim);
    for &v in &out {
        assert!(!v.is_nan());
        assert!(!v.is_infinite());
    }
}
