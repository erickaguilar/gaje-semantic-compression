import os
import sys
import numpy as np

# Asegurar uso de código local
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
)

from gaje.core._impl import NativeLoader
from tokenizers import Tokenizer


def evaluate_k_wta_sweep(model_path, tokenizer_path):
    print("🔬 Evaluation: Dynamic K-WTA Lateral Inhibition Sweep 🔬")
    print(f"[*] Target Model: {model_path}")
    print("-" * 60)

    loader = NativeLoader(model_path)
    rust_llm = loader.py_load_llm()
    tokenizer = Tokenizer.from_file(tokenizer_path)

    sample_text = (
        "El protocolo GAJE utiliza cuantización de 2 bits para lograr densidades extremas "
        "en redes neuronales profundas y sistemas de inferencia soberana."
    )

    tokens = tokenizer.encode(sample_text, add_special_tokens=False)
    if hasattr(tokens, "ids"):
        tokens = tokens.ids

    ratios = [0.10, 0.25, 0.50, 0.75, 0.95, 1.00]
    results = []

    for ratio in ratios:
        rust_llm.set_k_wta_ratio(ratio)
        rust_llm.clear_cache_py()

        total_log_prob = 0.0
        n_eval = 0

        for i in range(len(tokens) - 1):
            tid = tokens[i]
            target_tid = tokens[i + 1] % rust_llm.vocab_size

            logits = np.array(rust_llm.forward(tid, False), dtype=np.float64)

            # Mask out silenced tokens (-1e9) for numerical stability in softmax
            max_l = np.max(logits)
            exp_logits = np.exp(logits - max_l)
            probs = exp_logits / np.sum(exp_logits)

            target_prob = max(probs[target_tid], 1e-12)
            total_log_prob += np.log(target_prob)
            n_eval += 1

        ppl = np.exp(-total_log_prob / max(n_eval, 1))
        print(f"[*] K-WTA Ratio: {ratio*100:5.1f}% | Perplexity (PPL): {ppl:8.2f}")
        results.append((ratio, ppl))

    print("-" * 60)
    best_ratio, best_ppl = min(results, key=lambda x: x[1])
    print(f"🏆 Best Configuration: K-WTA Ratio = {best_ratio*100:.1f}% with PPL = {best_ppl:.2f}")


if __name__ == "__main__":
    evaluate_k_wta_sweep("models/silver_adult.gaje", "models/core/tokenizer.json")
