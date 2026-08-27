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
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"][0].tolist()

with torch.no_grad():
    hf_out = hf_model(**inputs, output_hidden_states=True)

# Trace token 0 across all 30 blocks
x_rust = gaje_llm.embeddings.get_row_core(input_ids[0])

print(
    f"\n--- TRACING TOKEN 0 ('{tokenizer.decode([input_ids[0]])}') ACROSS ALL 30 BLOCKS ---"
)
for b in range(30):
    x_rust = gaje_llm.blocks[b].rust_block.forward(x_rust, 0)
    hf_hidden_b = hf_out.hidden_states[b + 1][0, 0, :].detach().numpy()
    cossim = np.dot(hf_hidden_b, x_rust) / (
        np.linalg.norm(hf_hidden_b) * np.linalg.norm(x_rust)
    )
    print(
        f"Block {b:02d}: CosSim = {cossim:.8f} | HF Norm = {np.linalg.norm(hf_hidden_b):.4f} | Rust Norm = {np.linalg.norm(x_rust):.4f}"
    )
