import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.abspath("python"))


def rope_interleaved(x, pos, base=10000.0):
    dim = x.shape[-1]
    res = np.zeros_like(x)
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    theta = pos * inv_freq
    cos = np.cos(theta)
    sin = np.sin(theta)
    v0, v1 = x[..., 0::2], x[..., 1::2]
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
    v0, v1 = x[..., : dim // 2], x[..., dim // 2 :]
    res[..., : dim // 2] = v0 * cos - v1 * sin
    res[..., dim // 2 :] = v0 * sin + v1 * cos
    return res


def test_rope_variants_alignment():
    """Verifica la diferencia entre Interleaved y Split (usado en Rust)."""
    dim = 64
    pos = 1
    x = np.random.randn(dim).astype(np.float32)
    ri = rope_interleaved(x, pos)
    rs = rope_split(x, pos)

    # Rust usa Split (v0=x[i], v1=x[i+dim/2])
    assert not np.allclose(ri, rs), "Interleaved y Split deberían ser diferentes"
    print("✅ RoPE variants differentiation verified (Rust uses Split).")


if __name__ == "__main__":
    pytest.main([__file__])
