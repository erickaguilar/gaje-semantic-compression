import os
import sys
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM  # noqa: E402


def run_diff(gaje_path, model_id, prompt):
    print(f"[*] Cargando PyTorch FP32 de referencia ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    hf_model.eval()

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    inputs = torch.tensor([input_ids])
    with torch.no_grad():
        hf_logits = hf_model(inputs).logits[0, -1, :].numpy()

    print(f"[*] Evaluando GAJE Rust desde '{gaje_path}'...")
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

    print(f"\n  - Cosine Similarity: {cos_sim:.6f}")
    print(f"  - HuggingFace Top-1: '{tokenizer.decode([hf_top1])}' ({hf_top1})")
    print(f"  - GAJE Top-1:        '{tokenizer.decode([gaje_top1])}' ({gaje_top1})")
    print(f"  - Top-1 Match: {'✅ SÍ' if gaje_top1 == hf_top1 else '❌ NO'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaje-path", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--prompt", default="¿Cuál es la capital de Francia?")
    args = ap.parse_args()
    run_diff(args.gaje_path, args.model_id, args.prompt)
