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


def identity_test():
    # Usamos el modelo F16 para evitar errores de de-cuantización previa
    gguf_path = "models/gguf/smollm2-135m-f16.gguf"
    if not os.path.exists(gguf_path):
        print(f"Error: {gguf_path} not found")
        return

    print(f"Loading GGUF: {gguf_path}")
    reader = gguf.GGUFReader(gguf_path)

    # Target tensor
    tensor_name = "blk.0.attn_v.weight"
    tensor_data = None
    tensor_shape = None
    for t in reader.tensors:
        if t.name == tensor_name:
            # F16 data
            tensor_data = np.frombuffer(t.data, dtype=np.float16).astype(np.float32)
            tensor_shape = t.shape[::-1]  # GGUF dimensions are usually reversed
            break

    if tensor_data is None:
        print(f"Error: Tensor {tensor_name} not found")
        return

    w_original = tensor_data.reshape(tensor_shape)
    print(f"Original shape: {w_original.shape}")

    # Quantize using GAJE logic
    block_size = 32
    anchor_threshold = -1.0  # No anchors for pure 2-bit test

    print("Quantizing to 2-bit GAJE...")
    dna_db, centroids, anchors_bin = dna_semantic_compression.genomize_f32_native(
        w_original.tobytes(), block_size, anchor_threshold, 2
    )

    # Reconstruct from GAJE
    print("Reconstructing...")
    out_features, in_features = w_original.shape

    n_blocks = in_features // block_size
    stride = block_size // 4
    w_reconstructed = np.zeros_like(w_original)

    for i in range(out_features):
        row_dna = dna_db[i * n_blocks * stride : (i + 1) * n_blocks * stride]
        for b in range(n_blocks):
            block_dna = row_dna[b * stride : (b + 1) * stride]
            c_off = (i * n_blocks + b) * 4
            block_centroids = centroids[c_off : c_off + 4]

            # Dequantize block
            for k in range(stride):
                byte = block_dna[k]
                for s in range(4):
                    shift = (3 - s) * 2
                    bits = (byte >> shift) & 0b11

                    # Mapping from dequantize_embedding_core / genomize_f32_core
                    # 0b00 -> c0, 0b01 -> c1, 0b11 -> c2, 0b10 -> c3
                    if bits == 0b00:
                        cent = block_centroids[0]
                    elif bits == 0b01:
                        cent = block_centroids[1]
                    elif bits == 0b11:
                        cent = block_centroids[2]
                    elif bits == 0b10:
                        cent = block_centroids[3]
                    else:
                        cent = 0.0

                    w_reconstructed[i, b * block_size + k * 4 + s] = cent

    sim = calculate_cosine_similarity(w_original, w_reconstructed)
    mse = np.mean((w_original - w_reconstructed) ** 2)

    print("\n[IDENTITY TEST RESULT]")
    print(f"Tensor:            {tensor_name}")
    print(f"Cosine Similarity: {sim:.4f}")
    print(f"MSE:              {mse:.6f}")

    if sim < 0.80:
        print("❌ FAILED: Significant corruption detected!")
        # Debug small sample
        print("\nSample (Original vs Reconstructed):")
        print(f"Orig: {w_original[0, :8]}")
        print(f"Rec:  {w_reconstructed[0, :8]}")
    else:
        print("✅ PASSED: Quantization/Reconstruction is stable.")


if __name__ == "__main__":
    identity_test()
