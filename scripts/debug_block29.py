import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM
from gaje.core import _impl as dna_semantic_compression

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

# Run Rust through Block 28
x_rust = gaje_llm.embeddings.get_row_core(input_ids[0])
for b in range(29):
    x_rust = gaje_llm.blocks[b].rust_block.forward(x_rust, 0)

# Input to Block 29
x_hf_b28 = hf_out.hidden_states[29][0, 0, :].detach().numpy()
print(
    f"Block 28 Output CosSim: {np.dot(x_hf_b28, x_rust) / (np.linalg.norm(x_hf_b28) * np.linalg.norm(x_rust)):.8f}"
)

# Block 29 subops trace
blk29_rust = gaje_llm.blocks[29].rust_block
blk29_mock = gaje_llm.blocks[29]
blk29_hf = hf_model.model.layers[29]

x_norm_rust = (
    np.array(x_rust) / np.sqrt(np.mean(np.array(x_rust) ** 2) + 1e-5)
) * np.array(blk29_mock.attn_norm)
x_norm_hf = (
    blk29_hf.input_layernorm(torch.tensor([x_hf_b28], dtype=torch.float32))[0, :]
    .detach()
    .numpy()
)
print(
    f"Block 29 Attn Norm CosSim: {np.dot(x_norm_hf, x_norm_rust) / (np.linalg.norm(x_norm_hf) * np.linalg.norm(x_norm_rust)):.8f}"
)

q_rust = np.array(blk29_mock.attn_layer.q_gen.linear.forward(x_norm_rust, False))
k_rust = np.array(blk29_mock.attn_layer.k_gen.linear.forward(x_norm_rust, False))
v_rust = np.array(blk29_mock.attn_layer.v_gen.linear.forward(x_norm_rust, False))

q_hf = blk29_hf.self_attn.q_proj(torch.tensor([x_norm_hf]))[0, :].detach().numpy()
print(
    f"Block 29 Q Proj CosSim: {np.dot(q_hf, q_rust) / (np.linalg.norm(q_hf) * np.linalg.norm(q_rust)):.8f}"
)

attn_obj = dna_semantic_compression.GenomicAttention(
    9, 3, 64, blk29_mock.attn_norm, 1e-5, 100000.0, "split"
)
attn_out_rust = attn_obj.forward_attention(
    q_rust.tolist(), k_rust.tolist(), v_rust.tolist(), 0
)
proj_attn_rust = np.array(
    blk29_mock.attn_layer.w_o.linear.forward(attn_out_rust, False)
)

res1_rust = np.array(x_rust) + proj_attn_rust
res1_hf = (
    x_hf_b28
    + blk29_hf.self_attn(
        torch.tensor([[x_norm_hf]]),
        position_embeddings=hf_model.model.rotary_emb(
            torch.tensor([[x_norm_hf]]), torch.tensor([[0]])
        ),
    )[0][0, 0, :]
    .detach()
    .numpy()
)
print(
    f"Block 29 Residual 1 CosSim: {np.dot(res1_hf, res1_rust) / (np.linalg.norm(res1_hf) * np.linalg.norm(res1_rust)):.8f}"
)

# FFN norm in block 29
ffn_norm_rust = (res1_rust / np.sqrt(np.mean(res1_rust**2) + 1e-5)) * np.array(
    blk29_mock.ffn_norm
)
ffn_norm_hf = (
    blk29_hf.post_attention_layernorm(torch.tensor([res1_hf], dtype=torch.float32))[
        0, :
    ]
    .detach()
    .numpy()
)
print(
    f"Block 29 FFN Norm CosSim: {np.dot(ffn_norm_hf, ffn_norm_rust) / (np.linalg.norm(ffn_norm_hf) * np.linalg.norm(ffn_norm_rust)):.8f}"
)
print(
    f"   - HF FFN Norm weight norm:   {np.linalg.norm(blk29_hf.post_attention_layernorm.weight.detach().numpy()):.4f}"
)
print(f"   - Rust FFN Norm weight norm: {np.linalg.norm(blk29_mock.ffn_norm):.4f}")

# SwiGLU & down
gate_rust = np.array(blk29_mock.gate_gen.linear.forward(ffn_norm_rust.tolist(), False))
up_rust = np.array(blk29_mock.up_gen.linear.forward(ffn_norm_rust.tolist(), False))


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


swiglu_rust = silu(gate_rust) * up_rust
down_rust = np.array(blk29_mock.w_down.linear.forward(swiglu_rust.tolist(), False))

out_rust = res1_rust + down_rust
pos_emb_hf = hf_model.model.rotary_emb(torch.tensor([[x_hf_b28]]), torch.tensor([[0]]))
out_hf = (
    blk29_hf(torch.tensor([[x_hf_b28]]), position_embeddings=pos_emb_hf)[0]
    .flatten()
    .detach()
    .numpy()
)

print(
    f"\n🎯 BLOCK 29 TOTAL OUTPUT CosSim: {np.dot(out_hf, out_rust) / (np.linalg.norm(out_hf) * np.linalg.norm(out_rust)):.8f}"
)
print(f"  - HF Block 29 Norm:   {np.linalg.norm(out_hf):.6f}")
print(f"  - Rust Block 29 Norm: {np.linalg.norm(out_rust):.6f}")
