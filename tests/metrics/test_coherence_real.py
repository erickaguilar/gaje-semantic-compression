import numpy as np
import time
import os
from gaje.nn.stabilized import GenomicLLM


def test_coherence(model_arg=None):
    # Intentar cargar un modelo plano de producción primero para evitar OOM y optimizar velocidad
    flat_path = model_arg or "models/production/smollm2_4bit.gaje.flat"
    if os.path.exists(flat_path) and flat_path.endswith(".flat"):
        print(f"🧬 Loading pre-compiled GAJE Flat Model: {flat_path}")
        llm = GenomicLLM.load_genomic(flat_path)
    else:
        # Fallback a GGUF
        model_path = model_arg or "models/gguf/smollm2-135m-q8_0.gguf"
        if not os.path.exists(model_path):
            # Intentar rutas alternativas
            possible = [
                "/data/data/com.termux/files/home/models/gguf/smollm2-135m-f16.gguf",
                "data/models/qwen2-0_5b-instruct-fp16.gguf",
                "data/models/smollm2-135m-f16.gguf",
            ]
            for p in possible:
                if os.path.exists(p):
                    model_path = p
                    break
            else:
                print("❌ Model not found. Skipping test.")
                return

        print(f"🧬 Starting Real Coherence Test from GGUF (Model: {model_path})")
        print("-" * 50)
        # Limit blocks to 4 to reduce memory usage during on-the-fly python genomization
        llm = GenomicLLM(model_path, num_blocks=4)

    prompt = "The capital of France is"
    # Target: " Paris"

    print(f"[*] Prompt: '{prompt}'")
    tokens = llm.tokenizer.encode(prompt, add_special_tokens=False)
    print(f"[*] Tokens: {tokens} ({llm.tokenizer.convert_ids_to_tokens(tokens)})")

    start_time = time.time()
    all_logits = llm.forward(tokens)
    logits = all_logits[-1]  # Get logits for the last token
    end_time = time.time()

    next_token_id = np.argmax(logits)
    next_token = llm.tokenizer.decode([next_token_id])

    print(f"[*] Predicted next token: '{next_token}' (ID: {next_token_id})")
    print(f"[*] Inference time: {(end_time - start_time) * 1000:.2f} ms")

    # Top 5 tokens
    top_5_ids = np.argsort(logits)[-5:][::-1]
    print("[*] Top 5 candidates:")
    for i, tid in enumerate(top_5_ids):
        # Evitar overflow/underflow en softmax manual
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        prob = exp_logits[tid] / np.sum(exp_logits)
        print(
            f"  {i + 1}. '{llm.tokenizer.decode([tid])}' (ID: {tid}) - Prob: {prob:.4f}"
        )

    if "Paris" in next_token:
        print("✅ SUCCESS: Coherence verified! RoPE alignment is working.")
    else:
        print("⚠️ WARNING: Coherence still low. Further calibration needed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    test_coherence(args.model)
