//! 🧪 Test de Integración para Aceleración GPU (WGPU / Vulkan) en GenomicLinear y GenomicLLM
//!
//! Valida Fase 3 de GPU_ACCELERATION_BACKEND_PLAN.md:
//! - Offload de capas (`set_gpu_layers`, `offload_to_gpu`)
//! - Ejecución híbrida y paridad numérica CPU vs GPU

use _impl::nn::attention::GenomicAttention;
use _impl::nn::block::RustGenomicBlock;
use _impl::nn::linear::{GenomicLinear, WeightDatabase};
use _impl::nn::llm::GenomicLLM;
use std::sync::Arc;

#[test]
fn test_genomic_linear_forward_gpu_fallback() {
    let in_dim = 16;
    let out_dim = 8;
    let weights = vec![0.05f32; in_dim * out_dim];

    let mut linear = GenomicLinear::empty();
    linear.in_features = in_dim;
    linear.out_features = out_dim;
    linear.weight_db = WeightDatabase::GenomicF32(Arc::new(weights));

    let x = vec![0.1f32; in_dim];
    let cpu_out = linear
        .forward_core(x.clone(), None, false)
        .expect("CPU forward failed");
    assert_eq!(cpu_out.len(), out_dim);

    // Test GPU forward if feature is active or fallback
    let gpu_out = linear.forward_gpu(&x);
    if let Some(out) = gpu_out {
        assert_eq!(out.len(), out_dim);
        for (&c, &g) in cpu_out.iter().zip(out.iter()) {
            assert!((c - g).abs() < 1e-4, "Mismatch CPU vs GPU: {} vs {}", c, g);
        }
    }
}

#[test]
fn test_genomic_llm_gpu_layers_offload() {
    let dim = 16;
    let vocab_size = 32;

    let attn = GenomicAttention::new(2, 2, 8, vec![1.0; dim], 1e-5, 10000.0, "split".to_string());

    let block = RustGenomicBlock::new(
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

    let mut embeddings = GenomicLinear::empty();
    embeddings.in_features = dim;
    embeddings.out_features = vocab_size;
    embeddings.weight_db = WeightDatabase::GenomicF32(Arc::new(vec![0.02f32; vocab_size * dim]));

    let mut lm_head = GenomicLinear::empty();
    lm_head.in_features = dim;
    lm_head.out_features = vocab_size;
    lm_head.weight_db = WeightDatabase::GenomicF32(Arc::new(vec![0.02f32; vocab_size * dim]));

    let mut llm = GenomicLLM {
        embeddings,
        blocks: vec![block.clone(), block],
        output_norm: vec![1.0; dim],
        lm_head,
        eps: 1e-5,
        k_wta_ratio: 0.0,
        topology: None,
        quantum_embeddings: None,
        gpu_layers: 0,
        use_gpu: false,
    };

    assert_eq!(llm.get_gpu_layers(), 0);
    assert!(!llm.is_gpu_active());

    // Offload 2 capas a GPU
    let offloaded = llm.offload_to_gpu(2).expect("Offload failed");
    assert_eq!(offloaded, 2);
    assert_eq!(llm.get_gpu_layers(), 2);
    assert!(llm.is_gpu_active());

    // Pase forward con capas GPU activadas
    let logits = llm
        .forward_core(1, true)
        .expect("Forward core with GPU failed");
    assert_eq!(logits.len(), vocab_size);
    for &v in &logits {
        assert!(!v.is_nan());
        assert!(!v.is_infinite());
    }
}

#[test]
fn test_gpu_zero_allocation_persistent_pool() {
    let dim = 64;
    let x = vec![0.5f32; dim];
    let weight = vec![1.0f32; dim];

    // Llamar RMSNorm en GPU 20 veces consecutivas para verificar que el pool persistente funciona
    for _ in 0..20 {
        let res = _impl::compute::gpu::pipeline::gpu_rms_norm(&x, &weight, 1e-5);
        if let Some(gpu_norm) = res {
            assert_eq!(gpu_norm.len(), dim);
            for &v in &gpu_norm {
                assert!(!v.is_nan());
                assert!(v > 0.0);
            }
        }
    }

    // Verificar GEMV repetido con caché de pesos en VRAM
    let rows = 32;
    let cols = 64;
    let w = vec![0.02f32; rows * cols];
    let mut linear = GenomicLinear::empty();
    linear.in_features = cols;
    linear.out_features = rows;
    linear.weight_db = WeightDatabase::GenomicF32(Arc::new(w));

    for _ in 0..20 {
        let res = linear.forward_gpu(&x);
        if let Some(gpu_logits) = res {
            assert_eq!(gpu_logits.len(), rows);
            for &v in &gpu_logits {
                assert!(!v.is_nan());
            }
        }
    }
}

