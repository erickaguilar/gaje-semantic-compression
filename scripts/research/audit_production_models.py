from gaje.nn.stabilized import GenomicLLM
import numpy as np
import os


def test_models():
    model_dir = "models/production"
    models = [f for f in os.listdir(model_dir) if f.endswith(".gaje")]

    prompt = "GAJE es un protocolo de"

    for model_name in models:
        model_path = os.path.join(model_dir, model_name)
        print(f"\n--- Testing Model: {model_name} ---")
        try:
            llm = GenomicLLM.load_genomic(model_path)
            tokenizer = llm.tokenizer
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            if hasattr(tokens, "ids"):
                tokens = tokens.ids

            llm.rust_llm.clear_cache_py()
            next_logits = None
            for t in tokens:
                next_logits = llm.rust_llm.forward(t, False)

            print("Response: ", end="", flush=True)
            for _ in range(20):
                next_id = int(np.argmax(next_logits))
                word = tokenizer.decode([next_id])
                print(word, end="", flush=True)
                next_logits = llm.rust_llm.forward(next_id, False)
            print()
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    test_models()
