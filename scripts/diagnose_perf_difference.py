import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.core import _impl as dna_semantic_compression  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

flat_path = os.path.join(
    PROJECT_ROOT, "models", "production", "qwen2_0_5b_4bit.gaje.flat"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
gaje_llm = dna_semantic_compression.load_genomic_auto(flat_path)
gaje_llm.set_k_wta_ratio(0.0)

prompt = "A cuál país pertenece la capital París?"
input_ids = tokenizer.encode(prompt, add_special_tokens=False)

print(f"[*] Prompt: {prompt!r} | Tokens ({len(input_ids)}): {input_ids}")

# Step 1: Forward prompt tokens
gaje_llm.clear_cache_py()
t0_prefill = time.perf_counter()
for tid in input_ids:
    _ = gaje_llm.forward(tid, False)
prefill_ms = (time.perf_counter() - t0_prefill) * 1000.0
print(
    f"[*] Prefill forward loop ({len(input_ids)} tokens): {prefill_ms:.2f} ms ({prefill_ms / len(input_ids):.2f} ms/tok)"
)

# Step 2: Native generate
gaje_llm.clear_cache_py()
t0_gen = time.perf_counter()
gen_tokens = gaje_llm.generate_native_py(input_ids, 30, 0.3, 1.1, [2, 151643, 151645])
gen_ms = (time.perf_counter() - t0_gen) * 1000.0
n_gen = len(gen_tokens)
print(
    f"[*] generate_native_py: {gen_ms:.2f} ms for {n_gen} tokens ({gen_ms / n_gen:.2f} ms/tok)"
)
print(f"[*] Output: {tokenizer.decode(gen_tokens)!r}")
