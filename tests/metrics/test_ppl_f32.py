import os
import numpy as np


from gaje.nn.stabilized import GenomicLLM


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def calculate_ppl(model, text, tokenizer, max_length=128):
    """Calcula la perplejidad de un texto usando el modelo GAJE."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return None

    tokens = tokens[:max_length]
    logits_seq = model.forward(tokens, clear_cache=True)

    logits_seq = logits_seq[:-1]
    target_tokens = tokens[1:]

    log_probs = []
    for i, target_id in enumerate(target_tokens):
        logits = logits_seq[i]
        probs = softmax(logits)
        p = np.clip(probs[target_id], 1e-10, 1.0)
        log_probs.append(np.log(p))

    if not log_probs:
        model.clear_cache()
        return None

    avg_log_prob = np.mean(log_probs)
    model.clear_cache()
    return np.exp(-avg_log_prob)


def main():
    print("🧪 GAJE F32/Q8_0 Coherence Diagnostic")

    model_path = "models/SmolLM2-135M-Instruct-Q8_0.gguf"
    control_data_path = "data/datasets/master/coherence_es.txt"

    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return

    print(f"[~] Loading GGUF model in F32 mode from {model_path}...")
    model = GenomicLLM(model_path)
    tokenizer = model.tokenizer

    print("[~] Measuring Perplexity on control text...")
    with open(control_data_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if len(line.strip()) > 20][:3]

    ppls = []
    for line in lines:
        val = calculate_ppl(model, line, tokenizer)
        if val:
            ppls.append(val)
            print(f"  - Line: '{line[:50]}...' -> PPL: {val:.4f}")

    if ppls:
        avg_ppl = np.mean(ppls)
        print(f"🧬 Average F32 PPL: {avg_ppl:.4f}")
    else:
        print("❌ Could not compute PPL")
        avg_ppl = float("nan")

    print("\n[~] Testing top-5 next token predictions...")
    prompt = "The capital of France is Paris. The capital of Germany is"
    print(f'Prompt: "{prompt}"')
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    logits = model.forward(tokens, clear_cache=True)[-1]
    probs = softmax(logits)
    top_indices = np.argsort(probs)[::-1][:5]

    print("Top-5 predictions:")
    for idx in top_indices:
        token_str = tokenizer.decode([int(idx)])
        print(f"  - Token: {int(idx)} ({repr(token_str)}) -> Prob: {probs[idx]:.6f}")


if __name__ == "__main__":
    main()
