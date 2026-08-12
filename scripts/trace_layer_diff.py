import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM  # noqa: E402


def trace_layer_diff():
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    hf_model.eval()

    fp32_path = os.path.join(PROJECT_ROOT, "models", "smollm2-135m-fp32.gaje")
    gaje_llm = GenomicLLM.load_genomic(fp32_path)

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    print("\n=======================================================")
    print("🔬 DETECTOR DE DIVERGENCIA CAPA POR CAPA (SmolLM2 FP32)")
    print("=======================================================")
    print(f"Prompt: '{prompt}' | Token IDs: {input_ids}")

    # 1. Obtener hidden states de PyTorch para cada bloque
    inputs_hf = torch.tensor([input_ids])
    with torch.no_grad():
        hf_outputs = hf_model(inputs_hf, output_hidden_states=True)

    # 2. Rastreo en el motor Rust paso a paso
    # Para el último token del prompt (index -1, es decir pos 4)
    _tok_id = input_ids[0]
    gaje_llm.rust_llm.clear_cache_py()

    # Pre-fill secuencial para llenar KV cache hasta pos = len(input_ids)-1
    print("\n--- PRE-FILL KV CACHE EN MOTOR RUST ---")
    for pos, t_id in enumerate(input_ids[:-1]):
        _ = gaje_llm.rust_llm.forward(t_id, False)

    # Ahora procesamos el token final (pos = 4)
    target_token_id = input_ids[-1]
    target_pos = len(input_ids) - 1

    # Obtenemos el embedding inicial en Rust
    h_rust = gaje_llm.embeddings.linear.get_row(target_token_id)
    h_rust = np.array(h_rust, dtype=np.float32)

    # PyTorch embedding para el último token
    h_hf_emb = hf_outputs.hidden_states[0][0, -1, :].numpy()

    cos_emb = np.dot(h_hf_emb, h_rust) / (
        np.linalg.norm(h_hf_emb) * np.linalg.norm(h_rust) + 1e-9
    )
    print(
        f"\n[Capa 0 - Embeddings] CosSim: {cos_emb:.6f} | HF Norm: {np.linalg.norm(h_hf_emb):.4f} | Rust Norm: {np.linalg.norm(h_rust):.4f}"
    )

    # Paso bloque a bloque
    first_divergent_block = None

    for i in range(len(hf_model.model.layers)):
        h_hf_block = hf_outputs.hidden_states[i + 1][0, -1, :].numpy()

        # Ejecutamos bloque i en Rust
        rust_block = gaje_llm.blocks[i].rust_block
        # forward_core toma (x: Vec<f32>, pos: usize)
        h_rust = np.array(rust_block.forward(h_rust.tolist(), target_pos))

        cos_block = np.dot(h_hf_block, h_rust) / (
            np.linalg.norm(h_hf_block) * np.linalg.norm(h_rust) + 1e-9
        )
        mae_block = np.mean(np.abs(h_hf_block - h_rust))
        hf_norm = np.linalg.norm(h_hf_block)
        rust_norm = np.linalg.norm(h_rust)

        ratio = rust_norm / (hf_norm + 1e-9)
        status = "✅" if cos_block > 0.99 and 0.9 < ratio < 1.1 else "❌ DIVERGENCIA"

        print(
            f"Bloque {i:02d} | CosSim: {cos_block:.6f} | MAE: {mae_block:.6f} | HF Norm: {hf_norm:8.2f} | Rust Norm: {rust_norm:8.2f} | Ratio: {ratio:6.2f}x | {status}"
        )

        if status == "❌ DIVERGENCIA" and first_divergent_block is None:
            first_divergent_block = i

    print("\n=======================================================")
    if first_divergent_block is not None:
        print(
            f"🚨 PRIMER BLOQUE DIVERGENTE DETECTADO: BLOQUE {first_divergent_block:02d}"
        )
    else:
        print("🎉 TODOS LOS BLOQUES ESTÁN EN PARIDAD PERFECTA CON PYTORCH!")
    print("=======================================================")


if __name__ == "__main__":
    trace_layer_diff()
