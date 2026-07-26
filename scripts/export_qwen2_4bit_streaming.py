import os
import sys
import gc
import json
import numpy as np
import gguf
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

import gaje.core._impl as dna_semantic_compression
from gaje.nn.stabilized import GenomicLayer, dequantize_q8_0
from gaje.nn.configs import ARCHITECTURES

gguf_path = os.path.join(PROJECT_ROOT, "data", "models", "qwen2-0_5b-instruct-fp16.gguf")
out_path = os.path.join(PROJECT_ROOT, "models", "production", "qwen2_0_5b_4bit.gaje")

os.makedirs(os.path.dirname(out_path), exist_ok=True)

print(f"🚀 Exportación Streaming Ultra-Baja RAM de Qwen2-0.5B (4-bit Uniforme)")
print(f"  - Origen GGUF: {gguf_path}")
print(f"  - Destino GAJE: {out_path}")

reader = gguf.GGUFReader(gguf_path)
cfg = ARCHITECTURES["qwen2"]
cfg.attn_bit_depth = 4
cfg.ffn_bit_depth = 4
cfg.ffn_anchor_threshold = -1.0

# Extract hyperparams from GGUF fields
n_embd = 896
n_head = 14
n_head_kv = 2
head_dim = 64
n_blocks = 24
eps = 1e-6
rope_base = 1000000.0

for field_name, field in reader.fields.items():
    if "embedding_length" in field_name:
        n_embd = int(field.parts[field.data[0]][0])
    elif "head_count" in field_name and "head_count_kv" not in field_name:
        n_head = int(field.parts[field.data[0]][0])
    elif "head_count_kv" in field_name:
        n_head_kv = int(field.parts[field.data[0]][0])
    elif "block_count" in field_name:
        n_blocks = int(field.parts[field.data[0]][0])
    elif "layer_norm_rms_epsilon" in field_name:
        eps = float(field.parts[field.data[0]][0])

head_dim = n_embd // n_head
print(f"[*] Parámetros: n_embd={n_embd}, n_head={n_head}, n_head_kv={n_head_kv}, head_dim={head_dim}, n_blocks={n_blocks}")

# Map tensor names
tensors_by_name = {t.name: t for t in reader.tensors}

db_writer = dna_semantic_compression.GajeDatabaseWriter(out_path)

# 1. Metadata
metadata = {
    "config": {
        "name": "qwen2",
        "version": "0.9.5",
        "tokenizer_id": "Qwen/Qwen2-0.5B-Instruct",
        "rope_base": rope_base,
        "ffn_act": "swiglu",
        "use_genomic_norm": False,
    },
    "n_embd": n_embd,
    "n_head": n_head,
    "n_head_kv": n_head_kv,
    "n_blocks": n_blocks,
    "vocab_size": 151936,
    "eps": eps,
}
db_writer.write_metadata("config", json.dumps(metadata))

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
if hasattr(tokenizer, "backend_tokenizer"):
    db_writer.write_metadata("tokenizer", tokenizer.backend_tokenizer.to_str())

def save_layer_data(name, layer):
    db_writer.write_tensor_compressed(
        f"{name}.dna",
        np.frombuffer(layer.linear.database, dtype=np.uint8).tobytes(),
    )
    db_writer.write_tensor_compressed(
        f"{name}.centroids",
        np.array(layer.linear.centroids, dtype=np.float32).tobytes(),
    )
    db_writer.write_tensor_compressed(
        f"{name}.anchors", layer.anchors_f16_bytes
    )
    bias_val = getattr(layer, "bias", None)
    if bias_val is not None and len(bias_val) > 0:
        db_writer.write_tensor_compressed(
            f"{name}.bias",
            np.array(bias_val, dtype=np.float32).tobytes(),
        )

def process_and_write(name, tensor_obj, bit_depth=4, bias_obj=None):
    b_data = bias_obj.data if bias_obj else None
    layer = GenomicLayer(
        name,
        tensor_obj,
        bias_f32_or_tensor=b_data,
        bit_depth=bit_depth,
        n_head=n_head,
        head_dim=head_dim,
        config=cfg,
    )
    save_layer_data(name, layer)
    del layer
    gc.collect()

# Save Embeddings & LM Head
print("[*] Guardando token_embd y lm_head...")
process_and_write("token_embd", tensors_by_name["token_embd.weight"], bit_depth=32)

output_norm_tensor = tensors_by_name["output_norm.weight"]
out_norm_bytes = np.frombuffer(output_norm_tensor.data, dtype=np.float32).tobytes() if output_norm_tensor.tensor_type == gguf.GGMLQuantizationType.F32 else np.frombuffer(output_norm_tensor.data, dtype=np.float16).astype(np.float32).tobytes()
db_writer.write_tensor_compressed("output_norm", out_norm_bytes)

lm_head_tensor = tensors_by_name.get("lm_head.weight", tensors_by_name["token_embd.weight"])
process_and_write("lm_head", lm_head_tensor, bit_depth=32)

# Stream each block to keep RAM minimal (< 1.5 GB)
for i in range(n_blocks):
    p = f"blk.{i}."
    
    # Norms
    for norm_suffix in ["attn_norm", "ffn_norm"]:
        g_norm_key = f"blk.{i}.attn_norm.weight" if norm_suffix == "attn_norm" else f"blk.{i}.ffn_norm.weight"
        if g_norm_key in tensors_by_name:
            t = tensors_by_name[g_norm_key]
            norm_bytes = np.frombuffer(t.data, dtype=np.float32).tobytes() if t.tensor_type == gguf.GGMLQuantizationType.F32 else np.frombuffer(t.data, dtype=np.float16).astype(np.float32).tobytes()
            db_writer.write_tensor_compressed(p + norm_suffix, norm_bytes)

    # Attention & FFN layers
    layer_specs = [
        ("attn_q", f"blk.{i}.attn_q.weight", f"blk.{i}.attn_q.bias", 4),
        ("attn_k", f"blk.{i}.attn_k.weight", f"blk.{i}.attn_k.bias", 4),
        ("attn_v", f"blk.{i}.attn_v.weight", f"blk.{i}.attn_v.bias", 4),
        ("attn_output", f"blk.{i}.attn_output.weight", f"blk.{i}.attn_output.bias", 4),
        ("ffn_gate", f"blk.{i}.ffn_gate.weight", None, 4),
        ("ffn_up", f"blk.{i}.ffn_up.weight", None, 4),
        ("ffn_down", f"blk.{i}.ffn_down.weight", None, 4),
    ]

    for layer_name, gguf_weight, gguf_bias, bdepth in layer_specs:
        w_obj = tensors_by_name[gguf_weight]
        b_obj = tensors_by_name.get(gguf_bias, None) if gguf_bias else None
        process_and_write(p + layer_name, w_obj, bit_depth=bdepth, bias_obj=b_obj)

    print(f"  [~] Bloque {i+1}/{n_blocks} procesado y liberado de RAM.")
    gc.collect()

# Copiar también a models/qwen2_0_5b_4bit.gaje
import shutil
shutil.copy(out_path, os.path.join(PROJECT_ROOT, "models", "qwen2_0_5b_4bit.gaje"))

print(f"\n✅ Exportación Streaming Finalizada Exitosamente: {out_path}")
