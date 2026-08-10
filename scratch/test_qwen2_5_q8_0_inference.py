import time
from gaje.nn.stabilized import GenomicLLM

def run_1_5b_inference():
    model_path = "models/production/qwen2_5_1_5b_q4_0_q8_0_embd.gaje.flat"
    print(f"Loading Qwen2.5 1.5B hybrid model (Q4_0 + Q8_0 Embeddings) from {model_path}...")
    
    start_load = time.time()
    model = GenomicLLM.load_genomic(model_path)
    load_time = (time.time() - start_load) * 1000
    print(f"Model loaded in {load_time:.2f} ms")
    
    prompt = "<|im_start|>user\nUn padre tiene el triple de la edad de su hijo. Dentro de 12 años, tendrá el doble. ¿Qué edad tienen ambos actualmente? Explica el procedimiento paso a paso.<|im_end|>\n<|im_start|>assistant\n"
    print(f"\nPrompt: {prompt.strip()}")
    
    temp = 0.3
    rep_penalty = 1.1
    max_tokens = 450
    
    print("\nGenerating response (streaming):")
    start_gen = time.time()
    response_list = []
    for tok in model.generate(prompt, max_tokens, temp, rep_penalty):
        print(tok, end="", flush=True)
        response_list.append(tok)
    print()
    gen_time = time.time() - start_gen
    response = "".join(response_list)
    
    tokens = len(response_list)
    tok_s = tokens / gen_time
    print(f"\nThroughput: ~{tok_s:.2f} tok/s (generated {tokens} tokens in {gen_time:.2f} s)")

if __name__ == "__main__":
    run_1_5b_inference()
