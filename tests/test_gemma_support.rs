//! 🧪 Test de Soporte Arquitectural para Google Gemma y DeepSeek-R1 (Fase 2)

use _impl::compute::kernels::{geglu, rms_norm_offset};
use _impl::io::arch::{ArchitectureDescriptor, ModelFamily};
use _impl::io::downloader::resolve_model_url;

#[test]
fn test_gemma_and_deepseek_detection() {
    let gemma_desc = ArchitectureDescriptor::infer_from_name_core("gemma-2-2b-it.gguf", 2048, 8, 4, 18);
    assert_eq!(gemma_desc.family, ModelFamily::Gemma);
    assert_eq!(gemma_desc.ffn_act, "geglu");
    assert_eq!(gemma_desc.rope_style, "interleaved");

    let deepseek_desc = ArchitectureDescriptor::infer_from_name_core("DeepSeek-R1-Distill-Qwen-1.5B.gguf", 1536, 12, 2, 28);
    assert_eq!(deepseek_desc.family, ModelFamily::Qwen2_5);
    assert_eq!(deepseek_desc.ffn_act, "swiglu");
    assert_eq!(deepseek_desc.rope_style, "split");
}

#[test]
fn test_geglu_activation_numerical_stability() {
    let gate = vec![1.0, -2.0, 0.5, 3.0];
    let up = vec![2.0, 1.5, -1.0, 0.5];
    let mut out = vec![0.0; 4];

    geglu(&gate, &up, &mut out);

    // GeGLU no debe producir NaNs ni Infs
    for &v in &out {
        assert!(!v.is_nan());
        assert!(!v.is_infinite());
    }
}

#[test]
fn test_rms_norm_offset_gemma() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![0.1, -0.1, 0.05, -0.05]; // En Gemma los pesos son offsets respecto a 1.0
    let eps = 1e-6;

    let res_gemma = unsafe { rms_norm_offset(&x, &weight, eps, 1.0) };
    assert_eq!(res_gemma.len(), 4);
    for &v in &res_gemma {
        assert!(!v.is_nan());
        assert!(!v.is_infinite());
    }
}

#[test]
fn test_downloader_aliases() {
    let (url_r1, fname_r1) = resolve_model_url("r1");
    assert!(url_r1.contains("deepseek_r1_distill_qwen_1.5b.flat"));
    assert_eq!(fname_r1, "deepseek_r1_distill_qwen_1.5b.flat");

    let (url_gemma, fname_gemma) = resolve_model_url("gemma");
    assert!(url_gemma.contains("gaje_gemma_2b.flat"));
    assert_eq!(fname_gemma, "gaje_gemma_2b.flat");
}
