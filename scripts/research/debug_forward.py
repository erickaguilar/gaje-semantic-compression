import os
import sys
import numpy as np

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM


def debug_forward(model_path):
    print("🔍 [DEBUG] Inferencia Simple")
    model = GenomicLLM.load_genomic(model_path)
    tokenizer = model.tokenizer

    prompt = "Hola"
    # BOS = 1
    tokens = [1] + tokenizer.encode(prompt, add_special_tokens=False)
    if hasattr(tokens[1], "ids"):
        tokens = [1] + tokens[1].ids

    print(f"Input tokens: {tokens}")

    logits = model.forward(tokens, clear_cache=True)
    last_logits = logits[-1]

    print(f"Logits range: [{np.min(last_logits):.4f}, {np.max(last_logits):.4f}]")
    print(f"Logits mean: {np.mean(last_logits):.4f} | Std: {np.std(last_logits):.4f}")

    # Top 5
    top_indices = np.argsort(last_logits)[-5:][::-1]
    print("\nTop 5 predictions:")
    for idx in top_indices:
        token_str = tokenizer.decode([int(idx)])
        print(
            f"  ID: {idx:<6} | Logit: {last_logits[idx]:>8.4f} | Token: '{token_str}'"
        )


if __name__ == "__main__":
    debug_forward("models/production/smollm2_mixed_v1.gaje")
