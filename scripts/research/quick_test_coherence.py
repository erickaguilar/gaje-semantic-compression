from gaje.nn.stabilized import GenomicLLM
import numpy as np
import sys

def test():
    model_path = 'models/production/silver_adult_clean_v1.gaje'
    print(f"Loading {model_path}...")
    llm = GenomicLLM.load_genomic(model_path)
    tokenizer = llm.tokenizer
    
    prompts = [
        "Hola, ¿cómo estás?",
        "GAJE es un protocolo de",
        "Once upon a time,"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if hasattr(tokens, "ids"): tokens = tokens.ids
        
        llm.rust_llm.clear_cache_py()
        next_logits = None
        for t in tokens:
            next_logits = llm.rust_llm.forward(t, False)
            
        print("Response: ", end="", flush=True)
        for _ in range(30):
            # Using simple argmax for basic test
            next_id = int(np.argmax(next_logits))
            if next_id == tokenizer.eos_token_id:
                break
            word = tokenizer.decode([next_id])
            print(word, end="", flush=True)
            next_logits = llm.rust_llm.forward(next_id, False)
        print()

if __name__ == "__main__":
    test()
