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

tok_id = 791  # "The"
blk0_mock = gaje_llm.blocks[0]
blk0_rust = blk0_mock.rust_block
blk0_hf = hf_model.model.layers[0]

# Input Embedding
x_emb = gaje_llm.embeddings.get_row_core(tok_id)

x_emb_t = torch.tensor(x_emb).view(1, 1, -1)
pos_emb_hf = hf_model.model.rotary_emb(x_emb_t, torch.tensor([[0]]))
res_hf_tuple = blk0_hf(x_emb_t, position_embeddings=pos_emb_hf)
out_hf = res_hf_tuple[0].flatten().detach().numpy()

# Step-by-step in Rust:
attn_norm_w = blk0_hf.input_layernorm.weight.detach().numpy()
x_norm_rust = (
    np.array(x_emb) / np.sqrt(np.mean(np.array(x_emb) ** 2) + 1e-5)
) * attn_norm_w
q_rust = np.array(blk0_mock.attn_layer.q_gen.linear.forward(x_norm_rust, False))
k_rust = np.array(blk0_mock.attn_layer.k_gen.linear.forward(x_norm_rust, False))
v_rust = np.array(blk0_mock.attn_layer.v_gen.linear.forward(x_norm_rust, False))

attn_obj = dna_semantic_compression.GenomicAttention(
    9,
    3,
    64,
    blk0_hf.input_layernorm.weight.detach().numpy().tolist(),
    1e-5,
    100000.0,
    "split",
)
attn_out_rust = np.array(
    attn_obj.forward_attention(q_rust.tolist(), k_rust.tolist(), v_rust.tolist(), 0)
)
proj_attn_rust = np.array(
    blk0_mock.attn_layer.w_o.linear.forward(attn_out_rust.tolist(), False)
)

# Compare with HF:
q_hf = (
    blk0_hf.self_attn.q_proj(blk0_hf.input_layernorm(x_emb_t))[0, 0, :].detach().numpy()
)
k_hf = (
    blk0_hf.self_attn.k_proj(blk0_hf.input_layernorm(x_emb_t))[0, 0, :].detach().numpy()
)
v_hf = (
    blk0_hf.self_attn.v_proj(blk0_hf.input_layernorm(x_emb_t))[0, 0, :].detach().numpy()
)
attn_out_hf = (
    blk0_hf.self_attn(blk0_hf.input_layernorm(x_emb_t), position_embeddings=pos_emb_hf)[
        0
    ][0, 0, :]
    .detach()
    .numpy()
)

print(
    f"1. Norm x_emb:     HF = {np.linalg.norm(x_emb):.4f} | Rust = {np.linalg.norm(x_emb):.4f}"
)
print(
    f"2. Norm Q:         HF = {np.linalg.norm(q_hf):.4f} | Rust = {np.linalg.norm(q_rust):.4f}"
)
print(
    f"3. Norm K:         HF = {np.linalg.norm(k_hf):.4f} | Rust = {np.linalg.norm(k_rust):.4f}"
)
print(
    f"4. Norm V:         HF = {np.linalg.norm(v_hf):.4f} | Rust = {np.linalg.norm(v_rust):.4f}"
)
print(f"5. Norm Attn Out:  Rust = {np.linalg.norm(attn_out_rust):.4f}")
print(
    f"6. Norm Proj Attn: HF = {np.linalg.norm(attn_out_hf):.4f} | Rust = {np.linalg.norm(proj_attn_rust):.4f}"
)
print(
    f"   CosSim(Proj Attn): {np.dot(attn_out_hf, proj_attn_rust) / (np.linalg.norm(attn_out_hf) * np.linalg.norm(proj_attn_rust)):.8f}"
)

# 7. Residual 1
res1_rust = np.array(x_emb) + proj_attn_rust
res1_hf = np.array(x_emb) + attn_out_hf
print(
    f"7. Residual 1: CosSim = {np.dot(res1_hf, res1_rust) / (np.linalg.norm(res1_hf) * np.linalg.norm(res1_rust)):.8f}"
)

# 8. FFN Norm
eps = 1e-5
ffn_w = np.array(blk0_hf.post_attention_layernorm.weight.detach().numpy())
ffn_norm_rust = (res1_rust / np.sqrt(np.mean(res1_rust**2) + eps)) * ffn_w
ffn_norm_hf = (
    blk0_hf.post_attention_layernorm(torch.tensor([res1_hf]))[0, :].detach().numpy()
)
print(
    f"8. FFN Norm: CosSim = {np.dot(ffn_norm_hf, ffn_norm_rust) / (np.linalg.norm(ffn_norm_hf) * np.linalg.norm(ffn_norm_rust)):.8f}"
)

# 9. Gate & Up
gate_rust = np.array(blk0_mock.gate_gen.linear.forward(ffn_norm_rust.tolist(), False))
up_rust = np.array(blk0_mock.up_gen.linear.forward(ffn_norm_rust.tolist(), False))

gate_hf = (
    blk0_hf.mlp.gate_proj(torch.tensor([ffn_norm_hf], dtype=torch.float32))[0, :]
    .detach()
    .numpy()
)
up_hf = (
    blk0_hf.mlp.up_proj(torch.tensor([ffn_norm_hf], dtype=torch.float32))[0, :]
    .detach()
    .numpy()
)

print(
    f"9. Gate Proj: CosSim = {np.dot(gate_hf, gate_rust) / (np.linalg.norm(gate_hf) * np.linalg.norm(gate_rust)):.8f}"
)
print(
    f"10. Up Proj:  CosSim = {np.dot(up_hf, up_rust) / (np.linalg.norm(up_hf) * np.linalg.norm(up_rust)):.8f}"
)


# SwiGLU
def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


swiglu_hf = silu(gate_hf) * up_hf
swiglu_rust = silu(gate_rust) * up_rust
print(
    f"11. SwiGLU:   CosSim = {np.dot(swiglu_hf, swiglu_rust) / (np.linalg.norm(swiglu_hf) * np.linalg.norm(swiglu_rust)):.8f}"
)

# ffn_down
down_rust = np.array(blk0_mock.w_down.linear.forward(swiglu_rust.tolist(), False))
down_hf = (
    blk0_hf.mlp.down_proj(torch.tensor([swiglu_hf], dtype=torch.float32))[0, :]
    .detach()
    .numpy()
)
print(
    f"12. Down Proj:CosSim = {np.dot(down_hf, down_rust) / (np.linalg.norm(down_hf) * np.linalg.norm(down_rust)):.8f}"
)

# Total Block Output
out_rust = res1_rust + down_rust
out_hf = res1_hf + down_hf
print(
    f"\n🎯 BLOCK 0 TOTAL OUTPUT CosSim: {np.dot(out_hf, out_rust) / (np.linalg.norm(out_hf) * np.linalg.norm(out_rust)):.8f}"
)
print(f"  - HF Block 0 Norm:   {np.linalg.norm(out_hf):.6f}")
print(f"  - Rust Block 0 Norm: {np.linalg.norm(out_rust):.6f}")
