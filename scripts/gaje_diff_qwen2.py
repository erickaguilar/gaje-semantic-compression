import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM  # noqa: E402

def run_qwen2_diff():
    print("\n=======================================================")
    print("🔬 GAJE DIFF QWEN2: Certificación Numérica (HuggingFace vs GAJE v0.9.7 Fused 4-bit)")
    print("=======================================================")

    gaje_path = os.path.join(PROJECT_ROOT, "models", "production", "qwen2_0_5b_4bit.gaje")
    model_id = "Qwen/Qwen2-0.5B-Instruct"

    print(f"[*] Cargando PyTorch FP32 de referencia ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    hf_model.eval()

    prompt = "A cuál país pertenece la capital París?"
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    inputs = torch.tensor([input_ids])
    with torch.no_grad():
        hf_outputs = hf_model(inputs)
        hf_logits = hf_outputs.logits[0, -1, :].numpy()

    # --- EVALUACIÓN GAJE v0.9.7 FUSED 4-BIT ---
    print(f"[*] Evaluando GAJE Rust v0.9.7 Fused 4-bit desde '{gaje_path}'...")
    gaje_llm = GenomicLLM.load_genomic(gaje_path)
    gaje_llm.rust_llm.set_k_wta_ratio(0.0)

    gaje_logits = None
    for p_idx, tok_id in enumerate(input_ids):
        clear_cache = p_idx == 0
        gaje_logits = np.array(gaje_llm.rust_llm.forward(tok_id, clear_cache))

    cos_sim = np.dot(hf_logits, gaje_logits) / (
        np.linalg.norm(hf_logits) * np.linalg.norm(gaje_logits) + 1e-9
    )

    hf_top1 = int(np.argmax(hf_logits))
    gaje_top1 = int(np.argmax(gaje_logits))

    hf_top1_word = tokenizer.decode([hf_top1])
    gaje_top1_word = tokenizer.decode([gaje_top1])

    print("\n======================================================")
    print("GAJE Validation Report (Qwen2-0.5B Fused 4-bit v0.9.7)")
    print("======================================================")
    print(f"  - Cosine Similarity: {cos_sim:.6f}")
    print(f"  - HuggingFace Top-1: '{hf_top1_word}' ({hf_top1})")
    print(f"  - GAJE v0.9.7 Top-1:  '{gaje_top1_word}' ({gaje_top1})")
    print(f"  - Top-1 Match vs HF: {'✅ SÍ' if gaje_top1 == hf_top1 else '❌ NO'}")

if __name__ == "__main__":
    run_qwen2_diff()
