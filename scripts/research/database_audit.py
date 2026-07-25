import numpy as np
import gguf
import os
import sys

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM


def calculate_cosine_similarity(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)


def database_audit():
    gaje_path = "models/production/silver_adult_sovereign.gaje"
    gguf_path = "models/gguf/smollm2-135m-f16.gguf"  # Assuming this is the base

    if not os.path.exists(gaje_path):
        print(f"Error: {gaje_path} not found")
        return

    print(f"Auditing GAJE: {gaje_path} vs GGUF: {gguf_path}")

    # Load GAJE
    llm = GenomicLLM.load_genomic(gaje_path)
    # The first block's attn_v is layer 0's attn_v
    # Wait, how to access specific layers?
    # llm.rust_llm has blocks.

    # Load GGUF
    reader = gguf.GGUFReader(gguf_path)
    tensor_name = "blk.0.attn_v.weight"
    tensor_data = None
    for t in reader.tensors:
        if t.name == tensor_name:
            tensor_data = np.frombuffer(t.data, dtype=np.float16).astype(np.float32)
            tensor_shape = t.shape[::-1]
            break

    if tensor_data is None:
        print(f"Error: Tensor {tensor_name} not found in GGUF")
        return
    w_original = tensor_data.reshape(tensor_shape)

    # Extract reconstructed weights from GAJE model
    # blk.0 is index 0
    out_features, in_features = w_original.shape
    w_reconstructed = np.zeros_like(w_original)

    target_block = llm.rust_llm.blocks[0]
    target_linear = target_block.v_gen  # GenomicLinear

    for i in range(out_features):
        try:
            row = target_linear.get_row(i)
            w_reconstructed[i, :] = np.array(row)
        except Exception as e:
            print(f"Error reading row {i}: {e}")
            break

    sim = calculate_cosine_similarity(w_original, w_reconstructed)
    print("\n[DATABASE AUDIT RESULT]")
    print(f"Tensor:            {tensor_name}")
    print(f"Cosine Similarity: {sim:.4f}")

    if sim < 0.10:
        print(
            "❌ CRITICAL: The weights in the GAJE file are COMPLETELY CORRUPTED (~random)."
        )
    elif sim < 0.80:
        print("⚠️ WARNING: Significant drift or degradation.")
    else:
        print("✅ OK: Weights are preserved.")


if __name__ == "__main__":
    database_audit()
