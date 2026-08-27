import gc
import json
import os
import shutil
import sys
import numpy as np
import gguf
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

import gaje.core._impl as dna_semantic_compression  # noqa: E402
from gaje.nn.configs import ARCHITECTURES  # noqa: E402
from gaje.nn.stabilized import GenomicLayer, dequantize_q8_0  # noqa: E402

gguf_path = os.path.join(
    PROJECT_ROOT, "data", "models", "qwen2-0_5b-instruct-fp16.gguf"
)
out_path = os.path.join(PROJECT_ROOT, "models", "production", "qwen2_0_5b_4bit.gaje")

os.makedirs(os.path.dirname(out_path), exist_ok=True)

print("🚀 Exportación Streaming Ultra-Baja RAM de Qwen2-0.5B (4-bit Uniforme)")
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
print(
    f"[*] Parámetros: n_embd={n_embd}, n_head={n_head}, n_head_kv={n_head_kv}, head_dim={head_dim}, n_blocks={n_blocks}"
)

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
    db_writer.write_tensor_compressed(f"{name}.anchors", layer.anchors_f16_bytes)
    bias_val = getattr(layer, "bias", None)
    if bias_val is not None and len(bias_val) > 0:
        db_writer.write_tensor_compressed(
            f"{name}.bias",
            np.array(bias_val, dtype=np.float32).tobytes(),
        )


def process_and_write(name, tensor_obj, bit_depth=4, bias_obj=None):
    if isinstance(bias_obj, np.ndarray):
        b_data = bias_obj
    elif bias_obj is not None and hasattr(bias_obj, "data"):
        b_data = bias_obj.data
    else:
        b_data = None

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
out_norm_bytes = (
    np.frombuffer(output_norm_tensor.data, dtype=np.float32).tobytes()
    if output_norm_tensor.tensor_type == gguf.GGMLQuantizationType.F32
    else np.frombuffer(output_norm_tensor.data, dtype=np.float16)
    .astype(np.float32)
    .tobytes()
)
db_writer.write_tensor_compressed("output_norm", out_norm_bytes)

lm_head_tensor = tensors_by_name.get(
    "lm_head.weight", tensors_by_name["token_embd.weight"]
)
process_and_write("lm_head", lm_head_tensor, bit_depth=32)


def get_tensor_f32_matrix(tensor_obj, n_h=None, h_d=None, is_q_k=False):
    if tensor_obj.tensor_type == gguf.GGMLQuantizationType.F32:
        raw_data = np.frombuffer(tensor_obj.data, dtype=np.float32)
    elif tensor_obj.tensor_type == gguf.GGMLQuantizationType.F16:
        raw_data = np.frombuffer(tensor_obj.data, dtype=np.float16).astype(np.float32)
    elif tensor_obj.tensor_type == gguf.GGMLQuantizationType.Q8_0:
        return dequantize_q8_0(tensor_obj, n_h, h_d, is_q_k, rope_style="split")
    else:
        raise ValueError(f"Unsupported quantization type: {tensor_obj.tensor_type}")

    out_f = tensor_obj.shape[1] if len(tensor_obj.shape) == 2 else tensor_obj.shape[0]
    in_f = tensor_obj.shape[0] if len(tensor_obj.shape) == 2 else 1
    w_matrix = raw_data.reshape(out_f, in_f)
    if is_q_k and n_h is not None and h_d is not None:
        from gaje.utils.quantization import unpermute_to_split

        w_matrix = unpermute_to_split(w_matrix, n_h, h_d)
    return w_matrix


# Stream each block to keep RAM minimal (< 1.5 GB)
for i in range(n_blocks):
    p = f"blk.{i}."

    # Norms
    for norm_suffix in ["attn_norm", "ffn_norm"]:
        g_norm_key = (
            f"blk.{i}.attn_norm.weight"
            if norm_suffix == "attn_norm"
            else f"blk.{i}.ffn_norm.weight"
        )
        if g_norm_key in tensors_by_name:
            t = tensors_by_name[g_norm_key]
            norm_bytes = (
                np.frombuffer(t.data, dtype=np.float32).tobytes()
                if t.tensor_type == gguf.GGMLQuantizationType.F32
                else np.frombuffer(t.data, dtype=np.float16)
                .astype(np.float32)
                .tobytes()
            )
            db_writer.write_tensor_compressed(p + norm_suffix, norm_bytes)

    # 1. Fused QKV Layer (896 + 128 + 128 = 1152 rows)
    w_q_obj = tensors_by_name[f"blk.{i}.attn_q.weight"]
    w_k_obj = tensors_by_name[f"blk.{i}.attn_k.weight"]
    w_v_obj = tensors_by_name[f"blk.{i}.attn_v.weight"]

    w_q = get_tensor_f32_matrix(w_q_obj, n_head, head_dim, is_q_k=True)
    w_k = get_tensor_f32_matrix(w_k_obj, n_head_kv, head_dim, is_q_k=True)
    w_v = get_tensor_f32_matrix(w_v_obj, is_q_k=False)

    w_qkv = np.concatenate([w_q, w_k, w_v], axis=0)

    # Biases if present
    b_q_obj = tensors_by_name.get(f"blk.{i}.attn_q.bias", None)
    b_k_obj = tensors_by_name.get(f"blk.{i}.attn_k.bias", None)
    b_v_obj = tensors_by_name.get(f"blk.{i}.attn_v.bias", None)

    b_qkv = None
    if b_q_obj and b_k_obj and b_v_obj:
        bq = np.frombuffer(b_q_obj.data, dtype=np.float32)
        bk = np.frombuffer(b_k_obj.data, dtype=np.float32)
        bv = np.frombuffer(b_v_obj.data, dtype=np.float32)
        b_qkv = np.concatenate([bq, bk, bv], axis=0)

    process_and_write(p + "attn_qkv", w_qkv, bit_depth=4, bias_obj=b_qkv)

    # 2. Attention Output
    w_o_obj = tensors_by_name[f"blk.{i}.attn_output.weight"]
    b_o_obj = tensors_by_name.get(f"blk.{i}.attn_output.bias", None)
    process_and_write(p + "attn_output", w_o_obj, bit_depth=4, bias_obj=b_o_obj)

    # 3. Fused GateUp Layer (4864 + 4864 = 9728 rows)
    w_gate_obj = tensors_by_name[f"blk.{i}.ffn_gate.weight"]
    w_up_obj = tensors_by_name[f"blk.{i}.ffn_up.weight"]

    w_gate = get_tensor_f32_matrix(w_gate_obj)
    w_up = get_tensor_f32_matrix(w_up_obj)

    w_gate_up = np.concatenate([w_gate, w_up], axis=0)
    process_and_write(p + "ffn_gate_up", w_gate_up, bit_depth=4, bias_obj=None)

    # 4. FFN Down
    w_down_obj = tensors_by_name[f"blk.{i}.ffn_down.weight"]
    process_and_write(p + "ffn_down", w_down_obj, bit_depth=4, bias_obj=None)

    print(f"  [~] Bloque {i + 1}/{n_blocks} fusionado (4 capas) y liberado de RAM.")
    gc.collect()


shutil.copy(out_path, os.path.join(PROJECT_ROOT, "models", "qwen2_0_5b_4bit.gaje"))

print(f"\n✅ Exportación Fusionada v0.9.7 Finalizada Exitosamente: {out_path}")
