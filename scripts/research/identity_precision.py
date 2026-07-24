import numpy as np
import gguf
import os
import sys

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

import gaje.core._impl as dna_semantic_compression


def calculate_cosine_similarity(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)


def identity_test_4bit_precision():
    gguf_path = "models/gguf/smollm2-135m-f16.gguf"
    print(f"Loading GGUF: {gguf_path}")
    reader = gguf.GGUFReader(gguf_path)

    tensor_name = "blk.0.attn_q.weight"
    tensor = next(t for t in reader.tensors if t.name == tensor_name)
    w_original = np.frombuffer(tensor.data, dtype=np.float16).astype(np.float32)
    w_original = w_original.reshape(tensor.shape[::-1])

    # Quantize using our patched logic
    block_size = 32
    print("Quantizing to 4-bit...")
    dna, centroids, anchors = dna_semantic_compression.genomize_f32_native(
        w_original.tobytes(), block_size, -1.0, 4
    )

    # Reconstruct
    out_features, in_features = w_original.shape
    n_blocks = in_features // block_size
    stride = block_size // 2
    w_rec = np.zeros_like(w_original)

    for i in range(out_features):
        row_dna = dna[i * n_blocks * stride : (i + 1) * n_blocks * stride]
        for b in range(n_blocks):
            block_dna = row_dna[b * stride : (b + 1) * stride]
            block_c = centroids[(i * n_blocks + b) * 16 : (i * n_blocks + b) * 16 + 16]
            for k in range(stride):
                byte = block_dna[k]
                w_rec[i, b * block_size + k * 2] = block_c[(byte >> 4) & 0x0F]
                w_rec[i, b * block_size + k * 2 + 1] = block_c[byte & 0x0F]

    sim = calculate_cosine_similarity(w_original, w_rec)
    mse = np.mean((w_original - w_rec) ** 2)
    rel_err = np.mean(np.abs(w_original - w_rec)) / np.mean(np.abs(w_original))

    print("\n[4-BIT PRECISION REPORT]")
    print(f"Cosine Similarity: {sim:.6f}")
    print(f"MSE:              {mse:.6f}")
    print(f"Relative Error:    {rel_err:.6f}")

    # Sample weights
    print("\nSample (Original vs Reconstructed):")
    print(f"Orig: {w_original[0, :8]}")
    print(f"Rec:  {w_rec[0, :8]}")


if __name__ == "__main__":
    identity_test_4bit_precision()
