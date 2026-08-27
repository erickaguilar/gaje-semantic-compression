import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
hf_model.eval()

# Get q_proj weight from HF (shape: out_f, in_f = 576, 576)
w_q_hf = hf_model.model.layers[0].self_attn.q_proj.weight.detach().numpy()

# Test prompt token 'is' (pos 4)
prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, add_special_tokens=False)
emb_all = hf_model.model.embed_tokens(torch.tensor([input_ids]))
x_norm = hf_model.model.layers[0].input_layernorm(emb_all)[0, -1, :].detach().numpy()

# 1. Standard PyTorch q_proj output:
q_hf_std = x_norm @ w_q_hf.T  # (576,)

# 2. If weights were permuted (like GGUF) and unpermuted:
# Let's check if w_q_hf is already split or interleaved in PyTorch!
head_dim = 64
n_head = 9

# PyTorch LLaMA RoPE applies split rotary to [x1, x2] where x1 is q[:32] and x2 is q[32:64]
# Let's check RoPE math:
q0_hf = q_hf_std[:64]

q0_rot_split = q0_hf.copy()
rope_base = 100000.0
pos = 4
for i in range(32):
    freq = 1.0 / (rope_base ** ((2.0 * i) / 64.0))
    theta = pos * freq
    s_v, c_v = np.sin(theta), np.cos(theta)
    v0 = q0_rot_split[i]
    v1 = q0_rot_split[i + 32]
    q0_rot_split[i] = v0 * c_v - v1 * s_v
    q0_rot_split[i + 32] = v0 * s_v + v1 * c_v

# Now get HF PyTorch RoPE output:
with torch.no_grad():
    pos_ids = torch.tensor([[0, 1, 2, 3, 4]])
    cos_hf, sin_hf = hf_model.model.rotary_emb(emb_all, pos_ids)
    q_t_4d = torch.tensor(q_hf_std).view(1, 1, 9, 64).transpose(1, 2)
    q_rot_hf_t, _ = apply_rotary_pos_emb(
        q_t_4d, q_t_4d, cos_hf[:, -1:, :], sin_hf[:, -1:, :]
    )
    q0_rot_hf = q_rot_hf_t[0, 0, 0, :].detach().numpy()

cos_sim = np.dot(q0_rot_split, q0_rot_hf) / (
    np.linalg.norm(q0_rot_split) * np.linalg.norm(q0_rot_hf) + 1e-9
)
print(f"🔬 HF Weight + Split RoPE Alignment CosSim: {cos_sim:.8f}")
