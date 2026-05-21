import os
import sys
import numpy as np

sys.path.append(os.path.abspath("python"))
from stabilized_genomic_llm import GenomicLLM

def diagnose():
    model_path = "/data/data/com.termux/files/home/models/smollm2-135m-q8_0.gguf"
    model = GenomicLLM(model_path, num_blocks=4)
    
    text = "The capital of France is"
    tokens = model.tokenizer.encode(text, add_special_tokens=False)
    print(f"[*] Tokens: {tokens} ({[model.tokenizer.decode([t]) for t in tokens]})")
    
    logits = model.forward(tokens)
    
    # Softmax
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    
    top_5_ids = np.argsort(logits)[::-1][:5]
    print(f"[*] Top 5 predictions after '{text}':")
    for tid in top_5_ids:
        print(f"    - '{model.tokenizer.decode([tid])}' (Prob: {probs[tid]:.4f})")

if __name__ == "__main__":
    diagnose()
