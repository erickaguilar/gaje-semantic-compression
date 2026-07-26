import os
import sys
import time
import psutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM

model_path = os.path.join(PROJECT_ROOT, "models", "production", "qwen2_0_5b_4bit.gaje")
print(f"🔬 FASE 1A: PROFILING EMPÍRICO (Qwen2-0.5B 4-bit Nativo Rust)")
print(f"  - Cargar modelo: {model_path}...")

t0 = time.time()
llm = GenomicLLM.load_genomic(model_path)
llm.rust_llm.set_k_wta_ratio(0.0)
t_load = (time.time() - t0) * 1000

process = psutil.Process(os.getpid())
mem_mb = process.memory_info().rss / (1024 * 1024)
print(f"✅ Reconstrucción completada en {t_load:.2f} ms | RAM Usada: {mem_mb:.2f} MB")

prompt = "¿Cuál es la capital de Francia?"
formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

print(f"\n--- PROFILING GENERACIÓN PARA: {repr(prompt)} ---")

# 1. Tokenización
t_tok_0 = time.time()
tokens = llm.tokenizer.encode(formatted_prompt, add_special_tokens=False)
if hasattr(tokens, "ids"):
    tokens = tokens.ids
t_tok_ms = (time.time() - t_tok_0) * 1000
print(f"1. Tokenización Prompt ({len(tokens)} tokens): {t_tok_ms:.2f} ms")

# 2. Prefill + Generación token a token
t_gen_start = time.time()
token_times = []
generated_tokens = []

# Clear cache before generation
llm.rust_llm.clear_cache_py()

t_step_prev = time.time()
first_token_time = 0

for i, tok_text in enumerate(llm.generate(formatted_prompt, max_new_tokens=15, temperature=0.0)):
    t_now = time.time()
    step_ms = (t_now - t_step_prev) * 1000
    t_step_prev = t_now
    token_times.append(step_ms)
    generated_tokens.append(tok_text)
    if i == 0:
        first_token_time = step_ms
    if "<|im_end|>" in tok_text:
        break

t_total_gen_ms = (time.time() - t_gen_start) * 1000
num_tokens = len(generated_tokens)
avg_step_ms = sum(token_times[1:]) / max(len(token_times) - 1, 1) if len(token_times) > 1 else token_times[0]
tok_per_sec = (num_tokens / (t_total_gen_ms / 1000)) if t_total_gen_ms > 0 else 0

print(f"\n📊 DESGLOSE DE TIEMPOS DE INFERENCIA:")
print(f"  - Tiempo al Primer Token (TTFT / Prefill): {first_token_time:.2f} ms")
print(f"  - Promedio por Token Autorregresivo (Decode): {avg_step_ms:.2f} ms/tok")
print(f"  - Tiempo Total de Generación: {t_total_gen_ms:.2f} ms ({num_tokens} tokens)")
print(f"  - Velocidad Final: {tok_per_sec:.2f} tok/s")
print(f"  - Texto Generado: '{''.join(generated_tokens).strip()}'")

mem_final_mb = process.memory_info().rss / (1024 * 1024)
print(f"  - RAM Final del Proceso: {mem_final_mb:.2f} MB (Delta: {mem_final_mb - mem_mb:+.2f} MB)")
