import numpy as np
import os
import sys

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM

def logit_audit():
    model_path = 'models/production/silver_adult_clean_v1.gaje'
    print(f"Loading {model_path} for LOGIT audit...")
    llm = GenomicLLM.load_genomic(model_path)
    tokenizer = llm.tokenizer
    
    prompts = [
        "Hola",
        "Once upon a time",
        "The capital of France is"
    ]
    
    print("\n" + "="*50)
    print("🔍 AUDITORÍA DE LOGITS (Top 10)")
    print("="*50)
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if hasattr(tokens, "ids"): tokens = tokens.ids
        
        llm.rust_llm.clear_cache_py()
        logits = None
        for t in tokens:
            logits = np.array(llm.rust_llm.forward(t, False))
            
        # Top 10
        top_indices = np.argsort(logits)[-10:][::-1]
        top_values = logits[top_indices]
        
        for i, (idx, val) in enumerate(zip(top_indices, top_values)):
            token_str = tokenizer.decode([int(idx)])
            print(f"  {i+1}. [{token_str:15}] ID: {idx:6} | Logit: {val:.4f}")

if __name__ == "__main__":
    logit_audit()
