import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.core import _impl as dna_semantic_compression  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
import torch  # noqa: E402

flat_path = os.path.join(
    PROJECT_ROOT, "models", "production", "qwen2_0_5b_4bit.gaje.flat"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# Load HF model as in benchmark_scientific.py
print("[*] Loading HF FP32 Model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct", torch_dtype=torch.float32
)
hf_model.eval()

# Run one HF forward pass
prompt = "A cuál país pertenece la capital París?"
input_ids = tokenizer.encode(prompt, add_special_tokens=False)
inputs_tensor = torch.tensor([input_ids])
with torch.no_grad():
    _ = hf_model(inputs_tensor)

print("[*] Loading GAJE Model...")
gaje_llm = dna_semantic_compression.load_genomic_auto(flat_path)
gaje_llm.set_k_wta_ratio(0.0)

t0 = time.perf_counter()
gen_tokens = gaje_llm.generate_native_py(input_ids, 30, 0.3, 1.1, [2, 151643, 151645])
dt = (time.perf_counter() - t0) * 1000.0

print(
    f"[*] With HF FP32 warm-up: {dt:.2f} ms for {len(gen_tokens)} tokens ({dt/len(gen_tokens):.2f} ms/tok)"
)
