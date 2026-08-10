import os
import sys
import time
import json

PROJECT_ROOT = "/home/erickaguilar/Documentos/gaje-semantic-compression"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM

def benchmark_model(model_name, model_path, prompts):
    print(f"\n🧬 ========================================================")
    print(f"🧬 Benchmarking Model: {model_name}")
    print(f"🧬 Path: {model_path}")
    print(f"🧬 ========================================================")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return None
        
    start_load = time.time()
    llm = GenomicLLM.load_genomic(model_path)
    load_time = time.time() - start_load
    print(f"[*] Loaded in {load_time:.4f} seconds (Cold Start mmap)")
    
    # Disable k_wta lateral inhibition to run base model capability
    llm.rust_llm.set_k_wta_ratio(0.0)
    
    eos_ids = [151643, 151645]
    if hasattr(llm.tokenizer, "eos_token_id") and llm.tokenizer.eos_token_id is not None:
        eos_ids.append(llm.tokenizer.eos_token_id)
        
    results = []
    
    for idx, prompt in enumerate(prompts):
        formatted = f"<|im_start|>system\nYou are a helpful and precise assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        tokens = llm.tokenizer.encode(formatted, add_special_tokens=False)
        if hasattr(tokens, "ids"):
            tokens = tokens.ids
            
        llm.rust_llm.clear_cache_py()
        
        start_gen = time.time()
        # greedy search (temp=0.0)
        gen_ids = llm.rust_llm.generate_native_py(tokens, 64, 0.0, 1.0, eos_ids)
        gen_duration = time.time() - start_gen
        
        response = llm.tokenizer.decode(gen_ids)
        cleaned = response.split("<|im_start|>")[0].split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
        
        n_tokens = len(gen_ids)
        tok_per_sec = n_tokens / gen_duration if gen_duration > 0 else 0.0
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Response: {cleaned!r}")
        print(f"Generated {n_tokens} tokens in {gen_duration:.4f}s ({tok_per_sec:.2f} tok/s)")
        
        results.append({
            "prompt": prompt,
            "response": cleaned,
            "gen_time_sec": gen_duration,
            "tokens_generated": n_tokens,
            "tokens_per_sec": tok_per_sec
        })
        
    # Free memory
    del llm
    
    return {
        "model_name": model_name,
        "load_time_sec": load_time,
        "results": results
    }

def main():
    prompts = [
        "¿Cuál es la capital de Francia?",
        "太阳系中最大的行星是哪一颗？",
        "Count from 1 to 5"
    ]
    
    models_to_test = [
        ("Qwen2.5-1.5B (Q4_0 weights + FP32 embeddings)", os.path.join(PROJECT_ROOT, "models", "production", "qwen2_5_1_5b_q4_0.gaje.flat")),
        ("Qwen2.5-1.5B Híbrido (Q4_0 weights + Q8_0 embeddings)", os.path.join(PROJECT_ROOT, "models", "production", "qwen2_5_1_5b_q4_0_q8_0_embd.gaje.flat")),
        ("Qwen2.5-3B Híbrido (Q4_0 weights + Q8_0 embeddings)", os.path.join(PROJECT_ROOT, "models", "production", "qwen2_5_3b_q4_0_q8_0_embd.gaje.flat"))
    ]
    
    all_benchmarks = {}
    for name, path in models_to_test:
        res = benchmark_model(name, path, prompts)
        if res:
            all_benchmarks[name] = res
            
    # Save results to a json file
    output_json = os.path.join(PROJECT_ROOT, "docs", "reports", "factual_audit_phase_3.3_results.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_benchmarks, f, indent=4, ensure_ascii=False)
        
    print(f"\n[+] Benchmarks written to: {output_json}")

if __name__ == "__main__":
    main()
