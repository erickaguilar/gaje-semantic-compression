import numpy as np
import os
import sys

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM

def greedy_diagnostic():
    model_path = 'models/production/silver_adult_clean_v1.gaje'
    print(f"Loading {model_path} for GREEDY diagnostic...")
    llm = GenomicLLM.load_genomic(model_path)
    tokenizer = llm.tokenizer
    
    prompts = [
        "Hola, ¿cómo",
        "GAJE",
        "El",
        "123"
    ]
    
    print("\n--- Resultados Greedy (Temp=0) ---")
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if hasattr(tokens, "ids"): tokens = tokens.ids
        
        llm.rust_llm.clear_cache_py()
        next_logits = None
        for t in tokens:
            next_logits = llm.rust_llm.forward(t, False)
            
        print(f"Prompt: '{prompt}' -> Tokens: ", end="", flush=True)
        for _ in range(5):
            next_id = int(np.argmax(next_logits))
            word = tokenizer.decode([next_id])
            print(f"[{word}]", end="", flush=True)
            next_logits = llm.rust_llm.forward(next_id, False)
        print()

if __name__ == "__main__":
    greedy_diagnostic()
