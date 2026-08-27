import torch
import numpy as np
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    LlamaRotaryEmbedding,
)


def test_rope_standalone_math():
    head_dim = 64
    n_head = 9
    rope_base = 100000.0
    pos = 4

    # Create dummy Q, K
    torch.manual_seed(42)
    q_hf = torch.randn(1, n_head, 1, head_dim)
    k_hf = torch.randn(1, 3, 1, head_dim)

    rot_emb = LlamaRotaryEmbedding(
        config=type(
            "Cfg",
            (),
            {
                "head_dim": head_dim,
                "max_position_embeddings": 2048,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": rope_base,
                    "factor": 1.0,
                },
            },
        )()
    )
    cos, sin = rot_emb(q_hf, position_ids=torch.tensor([[pos]]))

    q_rot_hf, k_rot_hf = apply_rotary_pos_emb(q_hf, k_hf, cos, sin)

    q_hf_vec = q_hf[0, 0, 0, :].numpy()
    q_rot_hf_vec = q_rot_hf[0, 0, 0, :].numpy()

    # Rust RoPE math simulation
    q_rust = q_hf_vec.copy()
    for i in range(head_dim // 2):
        freq = 1.0 / (rope_base ** ((2.0 * i) / head_dim))
        theta = pos * freq
        sin_v, cos_v = np.sin(theta), np.cos(theta)
        v0 = q_rust[i]
        v1 = q_rust[i + head_dim // 2]
        q_rust[i] = v0 * cos_v - v1 * sin_v
        q_rust[i + head_dim // 2] = v0 * sin_v + v1 * cos_v

    cos_sim = float(
        np.dot(q_rot_hf_vec, q_rust)
        / (np.linalg.norm(q_rot_hf_vec) * np.linalg.norm(q_rust) + 1e-9)
    )
    mae = float(np.mean(np.abs(q_rot_hf_vec - q_rust)))

    assert cos_sim > 0.9999, f"RoPE CosSim below threshold: {cos_sim}"
    assert mae < 1e-5, f"RoPE MAE above threshold: {mae}"
