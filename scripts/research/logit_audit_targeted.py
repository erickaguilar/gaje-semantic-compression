import numpy as np
import os
import sys

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM

def logit_audit():
    model_path = 'models/production/silver_adult_clean_v1.gaje'
    llm = GenomicLLM.load_genomic(model_path)
    tokenizer = llm.tokenizer
    
    prompt = "Hola, ¿cómo"
    print(f"\nPrompt: '{prompt}'")
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if hasattr(tokens, "ids"): tokens = tokens.ids
    
    llm.rust_llm.clear_cache_py()
    logits = None
    for t in tokens:
        logits = np.array(llm.rust_llm.forward(t, False))
        
    top_indices = np.argsort(logits)[-5:][::-1]
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. [{tokenizer.decode([int(idx)])}] Logit: {logits[idx]:.4f}")

if __name__ == "__main__":
    logit_audit()
