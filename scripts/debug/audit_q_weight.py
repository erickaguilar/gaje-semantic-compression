import os
import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

fp32_path = os.path.join(PROJECT_ROOT, "models", "smollm2-135m-fp32.gaje")
gaje_llm = GenomicLLM.load_genomic(fp32_path)

w_q_hf = hf_model.model.layers[0].self_attn.q_proj.weight.detach().numpy()

# Compare with q_gen weight inside gaje_llm block 0
# Retrieve row 0 of q_gen weight in Rust
w_q_rust_rows = []
for i in range(576):
    row_i = gaje_llm.blocks[0].attn_layer.q_gen.linear.get_row(i)
    w_q_rust_rows.append(row_i)

w_q_rust = np.array(w_q_rust_rows)

cos_sim_q = np.dot(w_q_hf.flatten(), w_q_rust.flatten()) / (
    np.linalg.norm(w_q_hf) * np.linalg.norm(w_q_rust) + 1e-9
)
mae_q = np.mean(np.abs(w_q_hf - w_q_rust))

print(f"🔬 Layer 0 Q Projection Weight CosSim (HF vs GAJE FP32 .gaje): {cos_sim_q:.8f}")
print(f"  - MAE: {mae_q:.8f}")
print(f"  - HF W_q Norm:   {np.linalg.norm(w_q_hf):.6f}")
print(f"  - Rust W_q Norm: {np.linalg.norm(w_q_rust):.6f}")
