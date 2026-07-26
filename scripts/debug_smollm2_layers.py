import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM  # noqa: E402


def debug_layers():
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    hf_model.eval()

    fp32_path = os.path.join(PROJECT_ROOT, "models", "smollm2-135m-fp32.gaje")
    gaje_llm = GenomicLLM.load_genomic(fp32_path)

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    print(f"\n🔍 [Layer Debugger] Auditando Forward Pass para Prompt: '{prompt}'")
    print(f"Token IDs: {input_ids}")

    # 1. Embeddings Test
    emb_hf = hf_model.model.embed_tokens(torch.tensor([input_ids])).detach().numpy()[0]
    print("\n1. Embeddings:")
    print(
        f"   - HF Embedding Shape: {emb_hf.shape}, Norm token -1: {np.linalg.norm(emb_hf[-1]):.4f}"
    )

    # 2. Test Layer by Layer in PyTorch
    inputs_hf = torch.tensor([input_ids])
    with torch.no_grad():
        hf_outputs = hf_model(inputs_hf, output_hidden_states=True)
        for i, h_state in enumerate(
            hf_outputs.hidden_states[1:]
        ):  # Skip embedding layer
            norm_val = torch.norm(h_state[0, -1, :]).item()
            print(f"   - HF Block {i:02d} Hidden Norm: {norm_val:.4f}")

        logits_hf = hf_outputs.logits[0, -1, :].numpy()
        print(
            f"   - HF Final Logits Norm: {np.linalg.norm(logits_hf):.4f}, Max: {np.max(logits_hf):.4f}, Min: {np.min(logits_hf):.4f}"
        )

    # 3. Test Rust LLM Layer by Layer
    print("\n2. Rust Motor Layer-by-Layer State Trace:")
    # We trace block by block in Rust
    tok_id = input_ids[0]
    gaje_llm.rust_llm.clear_cache_py()
    h = np.array(gaje_llm.embeddings.get_row_core(tok_id))
    print(f"   - Rust Initial Embedding Norm: {np.linalg.norm(h):.4f}")

    for idx, block in enumerate(gaje_llm.blocks):
        # We forward through block
        h = block.forward(h)
        h_norm_val = np.linalg.norm(h)
        print(f"   - Rust Block {idx:02d} Hidden Norm: {h_norm_val:.4f}")
        if np.isnan(h_norm_val) or h_norm_val > 100000:
            print(f"❌ EXPLOSIÓN DETECTADA EN BLOQUE {idx}!")
            break


if __name__ == "__main__":
    debug_layers()
