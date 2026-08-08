import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

print(f"[*] PID: {os.getpid()}")
print(f"[*] OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
print(f"[*] RAYON_NUM_THREADS: {os.environ.get('RAYON_NUM_THREADS')}")

# Test 1: Standard launch without torch
from gaje.core import _impl as dna_semantic_compression  # noqa: E402

flat_path = os.path.join(
    PROJECT_ROOT, "models", "production", "qwen2_0_5b_4bit.gaje.flat"
)
llm = dna_semantic_compression.load_genomic_auto(flat_path)
llm.set_k_wta_ratio(0.0)

prompt_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

t0 = time.perf_counter()
tokens = llm.generate_native_py(prompt_tokens, 20, 0.3, 1.1, [2, 151643, 151645])
dt1 = (time.perf_counter() - t0) * 1000.0
print(
    f"Test 1 (Without torch import): {dt1:.2f} ms total for {len(tokens)} tokens ({dt1/len(tokens):.2f} ms/tok)"
)

# Test 2: Import torch (which configures CPU thread pool/affinity)
import torch  # noqa: E402

print(f"[*] Torch num_threads: {torch.get_num_threads()}")

llm.clear_cache_py()
t0 = time.perf_counter()
tokens = llm.generate_native_py(prompt_tokens, 20, 0.3, 1.1, [2, 151643, 151645])
dt2 = (time.perf_counter() - t0) * 1000.0
print(
    f"Test 2 (With torch imported):   {dt2:.2f} ms total for {len(tokens)} tokens ({dt2/len(tokens):.2f} ms/tok)"
)
