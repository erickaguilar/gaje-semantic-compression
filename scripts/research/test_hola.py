import numpy as np
import os
import sys

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM

def test_hola():
    model_path = 'models/production/silver_adult_clean_v1.gaje'
    llm = GenomicLLM.load_genomic(model_path)
    tokenizer = llm.tokenizer
    
    prompt = "Hola"
    print(f"\nPrompt: '{prompt}' (with BOS)")
    tokens = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)
    if hasattr(tokens, "ids"): tokens = tokens.ids
    
    llm.rust_llm.clear_cache_py()
    next_logits = None
    for t in tokens:
        next_logits = llm.rust_llm.forward(t, False)
        
    next_id = int(np.argmax(next_logits))
    print(f"Top Token: '{tokenizer.decode([next_id])}' (ID: {next_id})")

if __name__ == "__main__":
    test_hola()
