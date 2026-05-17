import numpy as np
import time
import os
<<<<<<< HEAD
from gaje.nn.genomize import GenomicLLM
=======
import sys
from gaje.nn.stabilized import GenomicLLM
>>>>>>> origin/develop


def test_coherence():
<<<<<<< HEAD
    model_path = "./data/models/qwen2-0_5b-instruct-fp16.gguf"
=======
    model_path = "/data/data/com.termux/files/home/models/smollm2-135m-f16.gguf"
>>>>>>> origin/develop
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    print("🧬 Starting Real Coherence Test (Native DGI Flow)")
    print("-" * 50)

    # Using all blocks for full verification
    llm = GenomicLLM(model_path, num_blocks=None)

    prompt = "The capital of France is"
    # Target: " Paris"

    print(f"[*] Prompt: '{prompt}'")
    tokens = llm.tokenizer.encode(prompt, add_special_tokens=False)
    print(f"[*] Tokens: {tokens} ({llm.tokenizer.convert_ids_to_tokens(tokens)})")

    start_time = time.time()
    all_logits = llm.forward(tokens)
    logits = all_logits[-1] # Get logits for the last token
    end_time = time.time()

    next_token_id = np.argmax(logits)
    next_token = llm.tokenizer.decode([next_token_id])

    print(f"[*] Predicted next token: '{next_token}' (ID: {next_token_id})")
    print(f"[*] Inference time: {(end_time - start_time)*1000:.2f} ms")

    # Top 5 tokens
    top_5_ids = np.argsort(logits)[-5:][::-1]
    print("[*] Top 5 candidates:")
    for i, tid in enumerate(top_5_ids):
        prob = np.exp(logits[tid] - np.max(logits))
        prob /= np.sum(np.exp(logits - np.max(logits)))
        print(
            f"  {i+1}. '{llm.tokenizer.decode([tid])}' (ID: {tid}) - Prob: {prob:.4f}"
        )

    if "Paris" in next_token:
        print("✅ SUCCESS: Coherence verified! RoPE alignment is working.")
    else:
        print("⚠️ WARNING: Coherence still low. Further calibration needed.")


if __name__ == "__main__":
    test_coherence()
   print("⚠️ WARNING: Coherence still low. Further calibration needed.")


if __name__ == "__main__":
    test_coherence()
me__ == "__main__":
    test_coherence()
