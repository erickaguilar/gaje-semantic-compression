import numpy as np
import os
import sys

def rotate_half_np(x):
    # Standard rotate_half for split RoPE using numpy
    # x shape: (batch, seq, heads, head_dim)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate((-x2, x1), axis=-1)

def apply_rotary_pos_emb_np(q, cos, sin):
    # q: [1, 1, 1, 64]
    # cos, sin: [1, 1, 64]
    return (q * cos) + (rotate_half_np(q) * sin)

def get_rope_cos_sin_np(head_dim, rope_base, pos):
    inv_freq = 1.0 / (rope_base ** (np.arange(0, head_dim, 2).astype(np.float32) / head_dim))
    t = np.array([pos], dtype=np.float32)
    freqs = np.outer(t, inv_freq) # [1, 32]
    emb = np.concatenate((freqs, freqs), axis=-1) # [1, 64]
    cos = np.cos(emb)
    sin = np.sin(emb)
    return cos.reshape(1, 1, head_dim), sin.reshape(1, 1, head_dim)

def rope_unit_test():
    print("--- RoPE Unit Test: GAJE vs NumPy-Ref ---")
    
    head_dim = 64
    rope_base = 100000.0
    pos = 10 
    
    # Input
    q_np = np.random.randn(1, 1, 1, head_dim).astype(np.float32)
    
    # 1. Reference: NumPy Implementation of Llama RoPE
    cos, sin = get_rope_cos_sin_np(head_dim, rope_base, pos)
    q_ref = apply_rotary_pos_emb_np(q_np, cos, sin)
    
    # 2. GAJE Rust Logic Implementation (Simulated in Python for formula check)
    def gaje_rope_logic_py(vec, pos, base):
        half = len(vec) // 2
        res = np.zeros_like(vec)
        for i in range(half):
            # Formula from my Rust code:
            freq = 1.0 / (base ** (2.0 * i / head_dim))
            theta = pos * freq
            s, c = np.sin(theta), np.cos(theta)
            v0 = vec[i]
            v1 = vec[i + half]
            # Rust code:
            # vec[h_start + i] = v0 * cos - v1 * sin;
            # vec[h_start + i + half] = v1 * cos + v0 * sin;
            res[i] = v0 * c - v1 * s
            res[i + half] = v1 * c + v0 * s
        return res

    q_gaje_sim = gaje_rope_logic_py(q_np.flatten(), pos, rope_base)
    
    # Calculate Similarity
    a = q_ref.flatten()
    b = q_gaje_sim.flatten()
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    print(f"Cosine Similarity (GAJE Logic vs Ref): {sim:.6f}")
    
    if sim < 0.9999:
        print("❌ RoPE Logic Mismatch!")
        print("\nFirst 4 elements:")
        print(f"Ref:  {a[:4]}")
        print(f"GAJE: {b[:4]}")
        
        # Check if the error is rotation direction or formula
        print("\nChecking if sin/cos match...")
        for i in range(4):
            freq = 1.0 / (rope_base ** (2.0 * i / head_dim))
            theta = pos * freq
            print(f"Index {i}: theta={theta:.4f}, cos={np.cos(theta):.4f}, sin={np.sin(theta):.4f}")
            print(f"Ref Cos: {cos[0,0,i]:.4f}, Ref Sin: {sin[0,0,i]:.4f}")
            
    else:
        print("✅ RoPE Logic matches Llama/Transformers formula.")

if __name__ == "__main__":
    rope_unit_test()
