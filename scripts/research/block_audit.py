import numpy as np
import gguf
import os
import sys

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM
import gaje.core._impl as dna_semantic_compression


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def rms_norm_np(x, w, eps=1e-6):
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps) * w


def reference_block_forward_verbose(x, weights):
    # 1. Attn Norm
    x_norm = rms_norm_np(x, weights["attn_norm"])

    # 2. Q, K, V Projections
    q = x_norm @ weights["q"].T
    k = x_norm @ weights["k"].T
    v = x_norm @ weights["v"].T

    # GQA: Expand K and V to match Q heads
    n_head = 9
    n_head_kv = 3
    head_dim = 64
    n_groups = n_head // n_head_kv

    np.repeat(k.reshape(n_head_kv, head_dim), n_groups, axis=0).flatten()
    v_exp = np.repeat(v.reshape(n_head_kv, head_dim), n_groups, axis=0).flatten()

    # 3. Attention (trivial for seq_len=1)
    # attn_out = v_exp
    # 4. Output Projection
    attn_out = v_exp @ weights["o"].T

    # 5. Residual
    h = x + attn_out

    # 6. FFN Norm
    h_norm = rms_norm_np(h, weights["ffn_norm"])

    # 7. FFN
    gate = h_norm @ weights["gate"].T
    up = h_norm @ weights["up"].T
    gate_act = gate * (1.0 / (1.0 + np.exp(-gate)))
    ffn_h = gate_act * up
    ffn_out = ffn_h @ weights["down"].T

    # Final
    y = h + ffn_out

    return {
        "x_norm": x_norm,
        "q": q,
        "attn_out": attn_out,
        "h": h,
        "h_norm": h_norm,
        "ffn_out": ffn_out,
        "y": y,
    }


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def block_audit():
    # ... (same as before)
    gguf_path = "models/gguf/smollm2-135m-f16.gguf"
    gaje_path = "models/production/silver_adult_clean_v1.gaje"

    print(f"Loading GGUF for weights: {gguf_path}")
    reader = gguf.GGUFReader(gguf_path)

    def get_w(name):
        t = next(t for t in reader.tensors if t.name == name)
        data = np.array(t.data).astype(np.float32)
        if len(t.shape) == 2:
            return data.reshape(t.shape[::-1])
        return data

    weights = {
        "attn_norm": get_w("blk.0.attn_norm.weight"),
        "q": get_w("blk.0.attn_q.weight"),
        "k": get_w("blk.0.attn_k.weight"),
        "v": get_w("blk.0.attn_v.weight"),
        "o": get_w("blk.0.attn_output.weight"),
        "ffn_norm": get_w("blk.0.ffn_norm.weight"),
        "gate": get_w("blk.0.ffn_gate.weight"),
        "up": get_w("blk.0.ffn_up.weight"),
        "down": get_w("blk.0.ffn_down.weight"),
    }

    print("Loading GAJE model...")
    llm = GenomicLLM.load_genomic(gaje_path)
    block = llm.blocks[0]

    # Fixed Input
    np.random.seed(42)
    x_in = np.random.randn(576).astype(np.float32)

    # 1. Run Reference
    print("\n--- Running Reference (NumPy) ---")
    ref = reference_block_forward_verbose(x_in, weights)

    # 2. Run GAJE
    print("--- Running GAJE (Rust) ---")

    # A. Check Projections
    x_norm_gaje = np.array(
        dna_semantic_compression.rms_norm_py(
            x_in.tolist(), weights["attn_norm"].tolist(), 1e-6
        )
    )

    # We can't easily run q_proj from block.rust_block without calling forward.
    # But we can call forward on block.rust_block and check q_abs_sum from logs

    y_gaje = np.array(block.rust_block.forward(x_in.tolist(), 0))

    print("\n[BLOCK INTERMEDIATE AUDIT]")
    print(f"RMSNorm Similarity:    {cosine(ref['x_norm'], x_norm_gaje):.6f}")
    print(f"Final Similarity:      {cosine(ref['y'], y_gaje):.6f}")

    # Magnitude Check
    print("\n[MAGNITUDE CHECK]")
    print(f"Ref x_norm abs_sum: {np.sum(np.abs(ref['x_norm'])):.4f}")
    print(f"Ref q_proj abs_sum: {np.sum(np.abs(ref['q'])):.4f}")
    print(f"Ref y      abs_sum: {np.sum(np.abs(ref['y'])):.4f}")
    print(f"GAJE y     abs_sum: {np.sum(np.abs(y_gaje)):.4f}")

    # Q Proj check from Rust logs
    # I saw: [Debug Block 0] q_abs_sum: 2883.7935
    # Ref q_proj abs_sum: 133.3637
    # Ratio: 21.6

    print("\n[PROJECT DISCREPANCY]")
    print(f"Q-Proj Ratio (GAJE/Ref): {2883.7935 / 133.3637:.2f}x")


if __name__ == "__main__":
    block_audit()
