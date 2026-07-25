import os
import sys
import numpy as np
import json
import gguf
import struct

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

import gaje.core._impl as dna_semantic_compression
from gaje.nn import constants as C


def run_identity_test():
    GGUF_PATH = "models/gguf/smollm2-135m-f16.gguf"
    OUTPUT_PATH = "models/test_artifacts/identity_test.gaje"

    print("🚀 Iniciando TEST DE IDENTIDAD (0-Loss) - FIX")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    reader = gguf.GGUFReader(GGUF_PATH)

    # Tomamos solo la primera capa para el test
    layer_name = "blk.0.attn_q.weight"
    tensor = next(t for t in reader.tensors if t.name == layer_name)
    # FLATTEN ES CRUCIAL
    w_orig = tensor.data.astype(np.float32).flatten()

    block_size = 32
    n_elements = w_orig.size
    n_blocks = n_elements // block_size

    dna_bytes = bytearray()
    all_centroids = []

    print(
        f"[~] Construyendo pesos de identidad para {layer_name} (Size: {n_elements})..."
    )
    for b in range(n_blocks):
        block_data = w_orig[b * block_size : (b + 1) * block_size]
        # Usamos los primeros 16 valores como centroides (para 4 bits)
        c = block_data[:16].tolist()
        all_centroids.extend(c)

        # DNA: índices 0..15 dos veces para cubrir 32 elementos
        for k in range(8):
            dna_bytes.append(((2 * k) << 4) | (2 * k + 1))
        for k in range(8):
            dna_bytes.append(((2 * k) << 4) | (2 * k + 1))

    # 2. Escribir a BD
    writer = dna_semantic_compression.GajeDatabaseWriter(OUTPUT_PATH)
    batch = writer.begin_batch()

    config = {
        "config": {"name": "Identity-Test"},
        "n_embd": 576,
        "n_head": 9,
        "n_blocks": 30,
        "vocab_size": 49152,
    }
    batch.write_metadata(C.META_KEY_CONFIG, json.dumps(config))

    base_name = "blk.0.attn_q"
    batch.write_tensor(f"{base_name}.dna", bytes(dna_bytes))
    batch.write_tensor(
        f"{base_name}.centroids", np.array(all_centroids, dtype=np.float32).tobytes()
    )

    # Anchor buffer dummy
    anchors_buf = (
        b"GAJE" + struct.pack("<I", 0) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
    )
    batch.write_tensor(f"{base_name}.anchors", anchors_buf)

    batch.commit()
    del batch
    del writer
    print(f"✅ Archivo de test generado en {OUTPUT_PATH}")

    # 3. Leer y Verificar
    print("[~] Verificando lectura desde Rust...")
    reader_gaje = dna_semantic_compression.GajeDatabaseReader(OUTPUT_PATH)
    dna_read = reader_gaje.read_tensor(f"{base_name}.dna")
    centroids_read = np.frombuffer(
        reader_gaje.read_tensor(f"{base_name}.centroids"), dtype=np.float32
    )

    # Comparar DNA
    if list(dna_read) == list(dna_bytes):
        print("✅ DNA: Los bytes coinciden exactamente.")
    else:
        print("❌ DNA: DISCREPANCIA.")

    # Comparar Centroides
    if np.allclose(centroids_read, all_centroids):
        print("✅ Centroides: Coincidencia exacta.")
    else:
        print("❌ Centroides: DISCREPANCIA.")


if __name__ == "__main__":
    run_identity_test()
