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

inputs = torch.tensor([input_ids])
with torch.no_grad():
    hf_out = hf_model(inputs, output_hidden_states=True)

# Test single token by single token in Rust:
# Let's inspect block 0 output for token 0 (pos 0):
h_rust = gaje_llm.embeddings.get_row_core(input_ids[0])
h_rust = gaje_llm.blocks[0].rust_block.forward(h_rust, 0)
h_hf_b0 = hf_out.hidden_states[1][0, 0, :].detach().numpy()

cos_b0_t0 = np.dot(h_hf_b0, h_rust) / (
    np.linalg.norm(h_hf_b0) * np.linalg.norm(h_rust) + 1e-9
)
print(f"🔬 Block 0 Token 0 (Pos 0) Hidden State CosSim: {cos_b0_t0:.8f}")
print(f"  - HF Norm:   {np.linalg.norm(h_hf_b0):.6f}")
print(f"  - Rust Norm: {np.linalg.norm(h_rust):.6f}")
