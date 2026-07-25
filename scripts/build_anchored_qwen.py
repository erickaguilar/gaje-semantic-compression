import os
import sys
import numpy as np
from gguf import GGUFReader
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.core import _impl as dna_semantic_compression


def build_anchored_model():
    gguf_path = os.path.join(PROJECT_ROOT, "data", "models", "qwen2-0_5b-instruct-fp16.gguf")
    output_path = os.path.join(PROJECT_ROOT, "models", "qwen2-0_5b-anchored.gaje")

    if not os.path.exists(gguf_path):
        print(f"Error: {gguf_path} not found.")
        return

    print(f"🧬 Genomizando Qwen2-0.5B con Stability Anchors (Anclas f16 5%)...")
    reader = GGUFReader(gguf_path)

    # Crear base de datos sqlite nativa .gaje
    writer = dna_semantic_compression.GajeDatabaseWriter(output_path)
    db_writer = writer.begin_batch()

    # 1. Configuración de Arquitectura
    arch_config = {
        "architecture": "qwen2",
        "n_layer": 24,
        "n_head": 14,
        "n_kv_head": 2,
        "n_embd": 896,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "block_size": 32,
        "quantization_bits": 2,
        "anchor_threshold": 0.05,
    }
    import json
    db_writer.write_metadata("config.json", json.dumps(arch_config))

    # 2. Tokenizador
    tokenizer_data = reader.fields.get("tokenizer.ggml.tokens")
    if tokenizer_data:
        tokens_list = [bytes(p).decode('utf-8', errors='ignore') for p in tokenizer_data.parts]
        db_writer.write_metadata("tokenizer.tokens", json.dumps(tokens_list))

    # 3. Genomización de Tensores
    tensors = {}
    for tensor in reader.tensors:
        tensors[tensor.name] = tensor

    print("[*] Genomizando capas con 5% Stability Anchors en Rust...")
    for name, tensor in tqdm(tensors.items()):
        base_name = name.replace("blk.", "blk_").replace(".weight", "")
        w_f32 = tensor.data.astype(np.float32).flatten()

        is_critical = ".attn_" in name or "token_embd" in name or "lm_head" in name
        anchor_rate = 0.05 if is_critical else 0.01
        bit_depth = 2

        (
            dna_bytes,
            centroids,
            anchors_buf,
        ) = dna_semantic_compression.genomize_f32_native(
            w_f32.tobytes(), 32, anchor_rate, bit_depth
        )

        db_writer.write_tensor_compressed(f"{base_name}.dna", dna_bytes)
        db_writer.write_tensor(
            f"{base_name}.centroids", np.array(centroids, dtype=np.float32).tobytes()
        )
        db_writer.write_tensor(f"{base_name}.anchors", anchors_buf)

    del db_writer
    print(f"✅ Organismo Genómico Anclado guardado exitosamente en: {output_path}")


if __name__ == "__main__":
    build_anchored_model()
