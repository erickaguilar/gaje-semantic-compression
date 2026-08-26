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

# Process 5 tokens in GAJE
for p_idx, tok_id in enumerate(input_ids):
    clear_cache = p_idx == 0
    gaje_logits = np.array(gaje_llm.rust_llm.forward(tok_id, clear_cache))

# Get HF last hidden state after block 29
hf_b29_hidden = hf_outputs.hidden_states[30][0, -1, :].detach().numpy()  # [0, -1, :]

# HF output norm
hf_norm_hidden = (
    hf_model.model.norm(torch.tensor([hf_b28_or_b29 := hf_b29_hidden]))[0, :]
    .detach()
    .numpy()
)

# HF logits
hf_logits = hf_outputs.logits[0, -1, :].detach().numpy()

# Compare logits
print(f"HF Logits Shape:   {hf_logits.shape}")
print(f"GAJE Logits Shape: {gaje_logits.shape}")

# Slice GAJE logits to vocab size (49152)
vocab_size = hf_logits.shape[0]
gaje_logits_vocab = gaje_logits[:vocab_size]

print(f"HF Logits Norm:         {np.linalg.norm(hf_logits):.4f}")
print(f"GAJE Logits Vocab Norm: {np.linalg.norm(gaje_logits_vocab):.4f}")
cos_vocab = np.dot(hf_logits, gaje_logits_vocab) / (
    np.linalg.norm(hf_logits) * np.linalg.norm(gaje_logits_vocab)
)
print(f"Logits Vocab CosSim:    {cos_vocab:.8f}")

# Check top 10 tokens
hf_top10 = np.argsort(hf_logits)[-10:][::-1]
gaje_top10 = np.argsort(gaje_logits)[-10:][::-1]

print("\n--- TOP 10 PREDICTIONS ---")
for i in range(10):
    hf_tok = tokenizer.decode([int(hf_top10[i])])
    gaje_tok = tokenizer.decode([int(gaje_top10[i])])
    print(
        f"Rank {i + 1}: HF = '{hf_tok}' ({hf_top10[i]}) score={hf_logits[hf_top10[i]]:.2f} | GAJE = '{gaje_tok}' ({gaje_top10[i]}) score={gaje_logits[gaje_top10[i]]:.2f}"
    )
