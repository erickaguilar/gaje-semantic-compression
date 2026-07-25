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


def identity_test_4bit():
    gguf_path = "models/gguf/smollm2-135m-f16.gguf"
    if not os.path.exists(gguf_path):
        print(f"Error: {gguf_path} not found")
        return

    print(f"Loading GGUF: {gguf_path}")
    reader = gguf.GGUFReader(gguf_path)

    # attn_q is 4-bit in GAJE
    tensor_name = "blk.0.attn_q.weight"
    tensor_data = None
    tensor_shape = None
    for t in reader.tensors:
        if t.name == tensor_name:
            tensor_data = np.frombuffer(t.data, dtype=np.float16).astype(np.float32)
            tensor_shape = t.shape[::-1]
            break

    if tensor_data is None:
        print(f"Error: Tensor {tensor_name} not found")
        return

    w_original = tensor_data.reshape(tensor_shape)
    print(f"Original shape: {w_original.shape}")

    # Quantize to 4-bit GAJE
    block_size = 32
    anchor_threshold = -1.0

    print("Quantizing to 4-bit GAJE...")
    dna_db, centroids, anchors_bin = dna_semantic_compression.genomize_f32_native(
        w_original.tobytes(),
        block_size,
        anchor_threshold,
        4,  # BIT DEPTH 4
    )

    # Reconstruct from GAJE
    print("Reconstructing...")
    out_features, in_features = w_original.shape

    n_blocks = in_features // block_size
    stride = block_size // 2  # 2 weights per byte for 4-bit
    w_reconstructed = np.zeros_like(w_original)

    for i in range(out_features):
        row_dna = dna_db[i * n_blocks * stride : (i + 1) * n_blocks * stride]
        for b in range(n_blocks):
            block_dna = row_dna[b * stride : (b + 1) * stride]
            c_off = (i * n_blocks + b) * 16  # 16 centroids for 4-bit
            block_centroids = centroids[c_off : c_off + 16]

            for k in range(stride):
                byte = block_dna[k]
                # High nibble: first element
                # Low nibble: second element
                # Matching genomic_dot_product_4bit
                idx1 = (byte >> 4) & 0x0F
                idx2 = byte & 0x0F

                w_reconstructed[i, b * block_size + k * 2] = block_centroids[idx1]
                w_reconstructed[i, b * block_size + k * 2 + 1] = block_centroids[idx2]

    sim = calculate_cosine_similarity(w_original, w_reconstructed)
    print("\n[4-BIT IDENTITY TEST RESULT]")
    print(f"Tensor:            {tensor_name}")
    print(f"Cosine Similarity: {sim:.4f}")

    if sim < 0.90:
        print("❌ FAILED: Significant 4-bit corruption detected!")
    else:
        print("✅ PASSED: 4-bit Quantization/Reconstruction is stable.")


if __name__ == "__main__":
    identity_test_4bit()
