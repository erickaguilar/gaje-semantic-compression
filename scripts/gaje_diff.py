import os
import sys
import gc
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM  # noqa: E402
from gaje.nn.configs import ARCHITECTURES  # noqa: E402


def build_smollm2_fp32():
    """Genera un organismo SmolLM2-135M con todas sus capas en FP32 puro (sin compresión)."""
    gguf_path = os.path.join(
        PROJECT_ROOT, "data", "models", "smollm2-135m-instruct-fp16.gguf"
    )
    fp32_output_path = os.path.join(PROJECT_ROOT, "models", "smollm2-135m-fp32.gaje")

    print(
        f"🧬 [gaje_diff] Generando baseline SmolLM2 FP32 nativo en: {fp32_output_path}..."
    )
    cfg = ARCHITECTURES["llama"]
    old_attn = cfg.attn_bit_depth
    old_ffn = cfg.ffn_bit_depth
    cfg.attn_bit_depth = 32
    cfg.ffn_bit_depth = 32

    try:
        llm = GenomicLLM(gguf_path)
        gc.collect()
        llm.save(fp32_output_path)
        print(f"✅ Baseline FP32 creado exitosamente en: {fp32_output_path}")
    finally:
        cfg.attn_bit_depth = old_attn
        cfg.ffn_bit_depth = old_ffn

    return fp32_output_path


def build_smollm2_mixed():
    """Genera un organismo SmolLM2-135M Mixed-Bit (4-bit atención, 2-bit FFN)."""
    gguf_path = os.path.join(
        PROJECT_ROOT, "data", "models", "smollm2-135m-instruct-fp16.gguf"
    )
    mixed_output_path = os.path.join(PROJECT_ROOT, "models", "smollm2-135m-mixed.gaje")

    print(
        f"🧬 [gaje_diff] Generando SmolLM2 Mixed-Bit (4-bit Attn, 2-bit FFN) en: {mixed_output_path}..."
    )
    cfg = ARCHITECTURES["llama"]
    old_attn = cfg.attn_bit_depth
    old_ffn = cfg.ffn_bit_depth
    cfg.attn_bit_depth = 4
    cfg.ffn_bit_depth = 2

    try:
        llm = GenomicLLM(gguf_path)
        gc.collect()
        llm.save(mixed_output_path)
        print(f"✅ Organismo Mixed-Bit creado exitosamente en: {mixed_output_path}")
    finally:
        cfg.attn_bit_depth = old_attn
        cfg.ffn_bit_depth = old_ffn

    return mixed_output_path


def run_gaje_diff():
    print("\n=======================================================")
    print("🔬 GAJE DIFF: Certificación Capa por Capa (SmolLM2 FP32 y Mixed-Bit)")
    print("=======================================================")

    fp32_path = build_smollm2_fp32()
    mixed_path = build_smollm2_mixed()

    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    print(f"[*] Cargando PyTorch FP32 de referencia ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    hf_model.eval()

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    inputs = torch.tensor([input_ids])
    with torch.no_grad():
        hf_outputs = hf_model(inputs)
        hf_logits = hf_outputs.logits[0, -1, :].numpy()

    # --- EVALUACIÓN FP32 ---
    print(f"[*] Evaluando GAJE Rust FP32 Organism desde '{fp32_path}'...")
    gaje_llm_fp32 = GenomicLLM.load_genomic(fp32_path)
    gaje_llm_fp32.rust_llm.set_k_wta_ratio(0.0)

    gaje_logits_fp32 = None
    for p_idx, tok_id in enumerate(input_ids):
        clear_cache = p_idx == 0
        gaje_logits_fp32 = np.array(gaje_llm_fp32.rust_llm.forward(tok_id, clear_cache))

    cos_fp32 = np.dot(hf_logits, gaje_logits_fp32) / (
        np.linalg.norm(hf_logits) * np.linalg.norm(gaje_logits_fp32) + 1e-9
    )

    # --- EVALUACIÓN MIXED-BIT ---
    print(
        f"[*] Evaluando GAJE Rust Mixed-Bit (4b Attn, 2b FFN) desde '{mixed_path}'..."
    )
    gaje_llm_mixed = GenomicLLM.load_genomic(mixed_path)
    gaje_llm_mixed.rust_llm.set_k_wta_ratio(0.0)

    gaje_logits_mixed = None
    for p_idx, tok_id in enumerate(input_ids):
        clear_cache = p_idx == 0
        gaje_logits_mixed = np.array(
            gaje_llm_mixed.rust_llm.forward(tok_id, clear_cache)
        )

    cos_mixed = np.dot(hf_logits, gaje_logits_mixed) / (
        np.linalg.norm(hf_logits) * np.linalg.norm(gaje_logits_mixed) + 1e-9
    )

    hf_top1 = int(np.argmax(hf_logits))
    fp32_top1 = int(np.argmax(gaje_logits_fp32))
    mixed_top1 = int(np.argmax(gaje_logits_mixed))

    print("\n======================================================")
    print("GAJE Validation Report (SmolLM2 FP32 vs Mixed-Bit)")
    print("======================================================")
    print("FP32 Baseline:")
    print(f"  - Cosine Similarity: {cos_fp32:.6f}")
    print(f"  - Top-1 Prediction:  '{tokenizer.decode([fp32_top1])}' ({fp32_top1})")
    print("\nMixed-Bit (4-bit Attn, 2-bit FFN):")
    print(f"  - Cosine Similarity: {cos_mixed:.6f}")
    print(f"  - Top-1 Prediction:  '{tokenizer.decode([mixed_top1])}' ({mixed_top1})")
    print(f"  - Top-1 Match vs HF: {'✅ SÍ' if mixed_top1 == hf_top1 else '❌ NO'}")


if __name__ == "__main__":
    run_gaje_diff()
