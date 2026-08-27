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

fp32_path = os.path.join(PROJECT_ROOT, "models", "smollm2-135m-fp32.gaje")
gaje_llm = GenomicLLM.load_genomic(fp32_path)

# Check token 791 ("The")
hf_emb_791 = hf_model.model.embed_tokens(torch.tensor([791])).detach().numpy()[0, :]
gaje_emb_791 = np.array(gaje_llm.embeddings.get_row_core(791))

cos_sim = np.dot(hf_emb_791, gaje_emb_791) / (
    np.linalg.norm(hf_emb_791) * np.linalg.norm(gaje_emb_791) + 1e-9
)
print(f"🔬 Token Embedding 791 CosSim: {cos_sim:.8f}")
print(f"  - HF Emb Norm:   {np.linalg.norm(hf_emb_791):.6f}")
print(f"  - GAJE Emb Norm: {np.linalg.norm(gaje_emb_791):.6f}")
