import numpy as np

def rope_interleaved(x, pos, base=10000.0):
    dim = x.shape[-1]
    res = np.zeros_like(x)
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    theta = pos * inv_freq
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    # Interleaved: [x0, x1, x2, x3] -> [(x0*cos - x1*sin), (x0*sin + x1*cos), ...]
    v0 = x[..., 0::2]
    v1 = x[..., 1::2]
    res[..., 0::2] = v0 * cos - v1 * sin
    res[..., 1::2] = v0 * sin + v1 * cos
    return res

def rope_split(x, pos, base=10000.0):
    dim = x.shape[-1]
    res = np.zeros_like(x)
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    theta = pos * inv_freq
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    # Split: [x0, x1, x2, x3] -> [(x0*cos - x2*sin), (x1*cos - x3*sin), (x0*sin + x2*cos), (x1*sin + x3*cos)]
    v0 = x[..., :dim//2]
    v1 = x[..., dim//2:]
    res[..., :dim//2] = v0 * cos - v1 * sin
    res[..., dim//2:] = v0 * sin + v1 * cos
    return res

def test_alignment():
    print("🧪 Testing RoPE Alignment Logic")
    dim = 64
    pos = 1
    x = np.random.randn(dim).astype(np.float32)
    
    ri = rope_interleaved(x, pos)
    rs = rope_split(x, pos)
    
    print(f"[*] Input head (first 4): {x[:4]}")
    print(f"[*] Interleaved (first 4): {ri[:4]}")
    print(f"[*] Split (first 4):       {rs[:4]}")
    
    # Comparison with Rust logic (from lib.rs)
    # Rust uses: 
    # v0 = vec[h_start + i]
    # v1 = vec[h_start + i + head_dim / 2]
    # res[h_start + i] = v0 * cos - v1 * sin
    # res[h_start + i + head_dim / 2] = v0 * sin + v1 * cos
    # This is EXACTLY rope_split.
    
    print("\n[!] Conclusion: Python uses Interleaved, Rust uses Split.")
    print("[!] Action: Need to unify to Split and potentially de-permute weights.")

if __name__ == "__main__":
    test_alignment()
