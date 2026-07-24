import numpy as np
import gguf
import os
import sys
import time

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


def reimport_and_verify():
    gguf_path = "models/gguf/smollm2-135m-f16.gguf"
    output_path = "models/production/silver_adult_clean_v1.gaje"

    if not os.path.exists(gguf_path):
        print(f"Error: {gguf_path} not found")
        return

    print(f"🚀 Iniciando Re-importación Limpia: {gguf_path}")
    t0 = time.time()

    # Initialize from GGUF (This performs genomization)
    model = GenomicLLM(model_path=gguf_path)

    print(f"[*] Modelo importado en {time.time() - t0:.2f}s")

    # Save to .gaje
    print(f"[*] Guardando en {output_path}...")
    model.save(output_path)

    print("[*] Modelo guardado. Iniciando Auditoría de Identidad...")

    # Verify a few layers
    reader = gguf.GGUFReader(gguf_path)
    test_layers = ["blk.0.attn_v.weight", "blk.0.attn_q.weight", "token_embd.weight"]

    for layer_name in test_layers:
        print(f"\nAuditando Capa: {layer_name}")
        # Get original
        tensor = next(t for t in reader.tensors if t.name == layer_name)
        w_original = np.frombuffer(tensor.data, dtype=np.float16).astype(np.float32)
        w_original = w_original.reshape(tensor.shape[::-1])

        # Get reconstructed from current model memory
        if layer_name == "token_embd.weight":
            target_layer = model.embeddings
        elif "attn_v" in layer_name:
            target_layer = model.blocks[0].attn_layer.v_gen
        elif "attn_q" in layer_name:
            target_layer = model.blocks[0].attn_layer.q_gen
        else:
            continue

        out_f, in_f = w_original.shape
        w_rec = np.zeros_like(w_original)
        for i in range(out_f):
            w_rec[i, :] = target_layer.get_row(i)

        sim = calculate_cosine_similarity(w_original, w_rec)
        print(f"  -> Cosine Similarity: {sim:.4f}")
        if sim < 0.90:
            print(f"  ❌ FALLO DE IDENTIDAD en {layer_name}")
        else:
            print("  ✅ IDENTIDAD OK")

    print("\n" + "=" * 40)
    print("✨ RE-IMPORTACIÓN FINALIZADA ✨")
    print("=" * 40)


if __name__ == "__main__":
    reimport_and_verify()
