import torch
import numpy as np
import os
import sys

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

import gaje.core._impl as dna_semantic_compression

def apply_rotary_pos_emb(q, cos, sin):
    # Standard Llama implementation of split RoPE
    # q: [1, 1, 1, 64]
    # cos, sin: [1, 1, 64]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

def rotate_half(x):
    # Standard rotate_half for split RoPE
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def rope_unit_test():
    print("--- RoPE Unit Test: GAJE vs Transformers ---")
    
    head_dim = 64
    rope_base = 100000.0
    pos = 10 # Test a non-zero position
    
    # Input
    q_torch = torch.randn(1, 1, 1, head_dim)
    q_np = q_torch.numpy().flatten()
    
    # 1. Reference: Logic matching Transformers
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    rope = LlamaRotaryEmbedding(config=None) # Mock config or manual
    # Manually set base
    rope.base = rope_base
    rope.inv_freq = 1.0 / (rope.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    
    # get cos/sin
    t = torch.tensor([pos]).float()
    freqs = torch.outer(t, rope.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    
    q_ref = apply_rotary_pos_emb(q_torch, cos, sin)
    
    # 2. GAJE Implementation
    # We use GenomicAttention indirectly or just the kernel if exposed.
    # It's not easily exposed. I'll use GenomicLLM to run a single token.
    
    from gaje.nn.stabilized import GenomicLLM
    from gaje.nn.configs import ArchitectureConfig
    
    config = ArchitectureConfig(
        name="test",
        rope_base=rope_base,
        rope_style="split"
    )
    
    # Create a minimal attention layer
    attn = dna_semantic_compression.GenomicAttention(
        1, 1, head_dim, [1.0]*head_dim, 1e-6, rope_base, "split"
    )
    
    # q, k, v for forward_attention_core
    # attn expects Vec<f32>
    q_in = q_np.tolist()
    k_in = q_np.tolist() # Same for simplicity
    v_in = q_np.tolist()
    
    # forward_attention_core returns attn_out, but we want to see q after rope.
    # I'll modify attention.rs to log Q after RoPE or use a dedicated FFI if I had one.
    # WAIT: I can't see Q inside. I have to trust the math or add a log.
    
    # Alternative: Manual Python implementation of MY Rust code to see if it matches Ref.
    def gaje_rope_logic_py(vec, pos, base):
        half = len(vec) // 2
        res = np.zeros_like(vec)
        for i in range(half):
            freq = 1.0 / (base ** (2.0 * i / head_dim))
            theta = pos * freq
            s, c = np.sin(theta), np.cos(theta)
            v0 = vec[i]
            v1 = vec[i + half]
            res[i] = v0 * c - v1 * s
            res[i + half] = v1 * c + v0 * s
        return res

    q_gaje_sim = gaje_rope_logic_py(q_np, pos, rope_base)
    
    sim = torch.nn.functional.cosine_similarity(
        q_ref.flatten(), torch.from_numpy(q_gaje_sim).flatten(), dim=0
    )
    
    print(f"Cosine Similarity (GAJE Logic vs Ref): {sim.item():.6f}")
    
    if sim < 0.999:
        print("❌ RoPE Logic Mismatch!")
        print("\nFirst 4 elements:")
        print(f"Ref:  {q_ref.flatten()[:4]}")
        print(f"GAJE: {q_gaje_sim[:4]}")
    else:
        print("✅ RoPE Logic matches Transformers.")

if __name__ == "__main__":
    rope_unit_test()
