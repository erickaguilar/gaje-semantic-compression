import os
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM  # noqa: E402


def run_gaje_diff():
    print("=" * 70)
    print("🔬 GAJE DIFF: COMPARADOR TENSOR A TENSOR (PyTorch HF vs GAJE Rust)")
    print("=" * 70)

    model_id = "Qwen/Qwen2-0.5B-Instruct"
    print(f"[*] Cargando modelo de referencia PyTorch/HuggingFace: {model_id}...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    prompt = "The"
    print(f"\n[*] Prompt: '{prompt}'")
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    # 1. Obtener hidden states de PyTorch/HuggingFace
    with torch.no_grad():
        hf_outputs = hf_model(input_tensor, output_hidden_states=True)

    # hf_outputs.hidden_states es una tupla de 25 elementos (capa 0 = embedding, capas 1..24 = bloques)
    # Tomamos la representación del último token del prompt (posición -1)
    hf_hidden_states = [h[0, -1].numpy() for h in hf_outputs.hidden_states]

    # 2. Cargar modelo GAJE FP32 Puro en Rust
    gguf_path = os.path.join(
        PROJECT_ROOT, "data", "models", "qwen2-0_5b-instruct-fp16.gguf"
    )
    print("\n[*] Cargando modelo GAJE en motor Rust...")
    gaje_llm = GenomicLLM(gguf_path)
    gaje_llm.clear_cache()

    print("\n" + "=" * 70)
    print("📊 INFORME DE PARIDAD CAPA POR CAPA (Similitud Coseno & Norma L2)")
    print("=" * 70)
    print(
        f"{'Capa':<12} | {'Nombre Capa':<20} | {'CosSim':<10} | {'L2 Norm (HF)':<14} | {'L2 Norm (GAJE)':<14} | {'Estado':<8}"
    )
    print("-" * 70)

    # Evaluamos hidden state a la entrada de las capas procesando token a token
    for p_idx, tid in enumerate(input_ids):
        h_curr = gaje_llm.embeddings.linear.get_row(tid)

        if p_idx == len(input_ids) - 1:
            # En el último token, capturamos la salida de cada bloque
            for l_idx, block in enumerate(gaje_llm.blocks):
                # Forward pass de este bloque en Rust
                h_next = block.rust_block.forward(h_curr, p_idx)
                rust_h = np.array(h_next)

                if l_idx == 23:
                    # hf_hidden_states[24] en Hugging Face tiene model.norm aplicado.
                    # Aplicamos output_norm a rust_h para comparar con la misma norma final de HF
                    out_norm_w = np.array(gaje_llm.output_norm)
                    h_rms = np.sqrt(np.mean(rust_h**2) + gaje_llm.eps)
                    rust_h_norm = (rust_h / h_rms) * out_norm_w
                    hf_h = hf_hidden_states[l_idx + 1]
                    cos_sim = np.dot(hf_h, rust_h_norm) / (
                        np.linalg.norm(hf_h) * np.linalg.norm(rust_h_norm) + 1e-9
                    )
                    hf_norm = np.linalg.norm(hf_h)
                    rust_norm = np.linalg.norm(rust_h_norm)
                else:
                    hf_h = hf_hidden_states[l_idx + 1]
                    cos_sim = np.dot(hf_h, rust_h) / (
                        np.linalg.norm(hf_h) * np.linalg.norm(rust_h) + 1e-9
                    )
                    hf_norm = np.linalg.norm(hf_h)
                    rust_norm = np.linalg.norm(rust_h)

                status = (
                    "✔ PERFECT"
                    if cos_sim > 0.99
                    else ("⚠️ DERIVA" if cos_sim > 0.5 else "❌ FALTAN BIAS")
                )
                print(
                    f"Bloque {l_idx:02d}    | {f'TransformerBlock.{l_idx}':<20} | {cos_sim:.6f}   | {hf_norm:<14.4f} | {rust_norm:<14.4f} | {status}"
                )

                h_curr = h_next
        else:
            # Para tokens previos, poblamos la memoria KV cache de Rust
            for block in gaje_llm.blocks:
                h_curr = block.rust_block.forward(h_curr, p_idx)

    # Capa final Logits
    hf_logits = hf_outputs.logits[0, -1].numpy()
    out_norm_w = np.array(gaje_llm.output_norm)
    h_curr_arr = np.array(h_curr)
    h_rms = np.sqrt(np.mean(h_curr_arr**2) + gaje_llm.eps)
    h_norm = (h_curr_arr / h_rms) * out_norm_w
    rust_logits = np.array(gaje_llm.lm_head.linear.forward(h_norm.tolist(), False))

    logit_cossim = np.dot(hf_logits, rust_logits) / (
        np.linalg.norm(hf_logits) * np.linalg.norm(rust_logits) + 1e-9
    )
    print("-" * 70)
    print(
        f"LM Head      | Output Vocabulary Logits | {logit_cossim:.6f}   | {np.linalg.norm(hf_logits):<14.4f} | {np.linalg.norm(rust_logits):<14.4f} | {'✔ PERFECT' if logit_cossim > 0.99 else '❌ FALTAN BIAS'}"
    )
    print("=" * 70)


if __name__ == "__main__":
    run_gaje_diff()
