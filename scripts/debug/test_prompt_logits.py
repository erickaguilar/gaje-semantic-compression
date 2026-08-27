import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.core import _impl as dna_semantic_compression  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def test():
    flat_path = os.path.join(
        PROJECT_ROOT, "models", "production", "qwen2_0_5b_4bit.gaje.flat"
    )
    llm = dna_semantic_compression.load_genomic_auto(flat_path)
    llm.set_k_wta_ratio(0.0)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    prompt = "A cuál país pertenece la capital París?"
    tokens = tokenizer.encode(prompt, add_special_tokens=False)

    print(f"[*] Prompt: {prompt!r}")
    print(f"[*] Tokens: {tokens}")

    # Forward prompt token by token
    llm.clear_cache_py()
    logits = None
    for tid in tokens:
        logits = llm.forward(tid, False)

    import torch
    from transformers import AutoModelForCausalLM

    print("[*] Cargando HuggingFace PyTorch FP32 de referencia...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct", torch_dtype=torch.float32
    )
    hf_model.eval()

    inputs = torch.tensor([tokens])
    with torch.no_grad():
        hf_logits = hf_model(inputs).logits[0, -1, :].numpy()

    hf_top5 = np.argsort(hf_logits)[-5:][::-1]
    print("\n--- HUGGINGFACE PYTORCH FP32 TOP 5 ---")
    for rank, tid in enumerate(hf_top5, 1):
        print(
            f"  {rank}. ID={tid} ({tokenizer.decode([tid])!r}): Logit={hf_logits[tid]:.4f}"
        )

    top5_ids = np.argsort(logits)[-5:][::-1]
    print("\n--- GAJE v0.9.7 FLAT MMAP TOP 5 ---")
    for rank, tid in enumerate(top5_ids, 1):
        token_str = tokenizer.decode([tid])
        logit_val = logits[tid]
        print(f"  {rank}. ID={tid} ({token_str!r}): Logit={logit_val:.4f}")

    cos_sim = np.dot(hf_logits, logits) / (
        np.linalg.norm(hf_logits) * np.linalg.norm(logits)
    )
    print(f"\n🔬 Logits Cosine Similarity vs HF FP32: {cos_sim:.6f}")


if __name__ == "__main__":
    test()
