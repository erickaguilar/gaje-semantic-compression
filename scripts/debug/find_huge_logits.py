import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
hf_model.eval()

fp32_path = os.path.join(PROJECT_ROOT, "models", "smollm2-135m-fp32.gaje")
gaje_llm = GenomicLLM.load_genomic(fp32_path)

prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, add_special_tokens=False)

for p_idx, tok_id in enumerate(input_ids):
    clear_cache = p_idx == 0
    gaje_logits = np.array(gaje_llm.rust_llm.forward(tok_id, clear_cache))

print("Finding indices with huge values in gaje_logits...")
huge_indices = np.where(np.abs(gaje_logits) > 1e4)[0]
print(f"Number of huge logit indices: {len(huge_indices)}")
if len(huge_indices) > 0:
    print(f"First 10 huge indices: {huge_indices[:10]}")
    print(f"Values at these indices: {gaje_logits[huge_indices[:10]]}")
