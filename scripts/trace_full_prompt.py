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
    hf_outputs = hf_model(inputs, output_hidden_states=True)

print(f"\n--- TRACING ALL TOKENS IN PROMPT: {prompt} ---")

for p_idx, tok_id in enumerate(input_ids):
    clear_cache = p_idx == 0
    gaje_logits = np.array(gaje_llm.rust_llm.forward(tok_id, clear_cache))
    hf_logits_pos = hf_outputs.logits[0, p_idx, :].detach().numpy()

    cos_sim = np.dot(hf_logits_pos, gaje_logits) / (
        np.linalg.norm(hf_logits_pos) * np.linalg.norm(gaje_logits)
    )
    print(
        f"Token {p_idx} ('{tokenizer.decode([tok_id])}'): Logits CosSim = {cos_sim:.8f}"
    )
    print(
        f"  - HF Top-1:   '{tokenizer.decode([int(np.argmax(hf_logits_pos))])}' ({np.argmax(hf_logits_pos)})"
    )
    print(
        f"  - GAJE Top-1: '{tokenizer.decode([int(np.argmax(gaje_logits))])}' ({np.argmax(gaje_logits)})"
    )
