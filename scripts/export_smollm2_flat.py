#!/usr/bin/env python3
"""
🚀 Exportador Flat Zero-Copy v0.9.7 para SmolLM2 135M (.gaje.flat)
------------------------------------------------------------------
Convierte data/models/smollm2-135m-instruct-fp16.gguf al formato binario
plano alineado a 64 bytes para carga mmap instantánea en 0.15s.
"""

import os
import sys
import gc
import json
import struct
import numpy as np
import gguf
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.core._impl import genomize_f32_native  # noqa: E402


def get_tensor_f32_matrix(tensor_obj):
    if tensor_obj.tensor_type == gguf.GGMLQuantizationType.F16:
        w_f16 = np.frombuffer(tensor_obj.data, dtype=np.float16)
        w_f32 = w_f16.astype(np.float32)
    elif tensor_obj.tensor_type == gguf.GGMLQuantizationType.F32:
        w_f32 = np.frombuffer(tensor_obj.data, dtype=np.float32)
    else:
        raise ValueError(f"Tipo no soportado: {tensor_obj.tensor_type}")

    shape = tensor_obj.shape
    if len(shape) == 2:
        out_dim, in_dim = int(shape[1]), int(shape[0])
        w_f32 = w_f32.reshape(out_dim, in_dim)
    return w_f32


def export_smollm2_flat():
    gguf_path = os.path.join(
        PROJECT_ROOT, "data", "models", "smollm2-135m-instruct-fp16.gguf"
    )
    out_path = os.path.join(
        PROJECT_ROOT, "models", "production", "smollm2_4bit.gaje.flat"
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("🚀 Exportador Flat Zero-Copy v0.9.7 para SmolLM2 135M")
    print(f"  - Origen GGUF: {gguf_path}")
    print(f"  - Destino Flat: {out_path}")

    reader = gguf.GGUFReader(gguf_path)

    n_embd = 576
    n_head = 9
    n_head_kv = 3
    n_blocks = 30
    eps = 1e-5
    rope_base = 10000.0

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
        f"[*] Parámetros SmolLM2: n_embd={n_embd}, n_head={n_head}, n_head_kv={n_head_kv}, head_dim={head_dim}, n_blocks={n_blocks}"
    )

    tensors_by_name = {t.name: t for t in reader.tensors}

    try:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
        tokenizer_str = (
            tokenizer.backend_tokenizer.to_str()
            if hasattr(tokenizer, "backend_tokenizer")
            else ""
        )
    except Exception:
        tokenizer_str = ""

    metadata_dict = {
        "config": {
            "name": "smollm2",
            "version": "0.9.7",
            "tokenizer_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "rope_base": rope_base,
            "ffn_act": "swiglu",
            "use_genomic_norm": False,
        },
        "n_embd": n_embd,
        "n_head": n_head,
        "n_head_kv": n_head_kv,
        "n_blocks": n_blocks,
        "vocab_size": 49152,
        "eps": eps,
        "tokenizer": tokenizer_str,
    }

    metadata_json_bytes = json.dumps(metadata_dict).encode("utf-8")

    tensor_directory = []
    blob_bytes = bytearray()

    def add_blob_data(name, raw_data_bytes):
        offset = len(blob_bytes)
        blob_bytes.extend(raw_data_bytes)
        padding = (64 - (len(blob_bytes) % 64)) % 64
        blob_bytes.extend(b"\x00" * padding)
        return offset, len(raw_data_bytes)

    def process_layer_data(name, tensor_f32, bit_depth=4, bias_obj=None):
        out_features, in_features = tensor_f32.shape[0], tensor_f32.shape[1]
        block_size = 64

        gen_ret = genomize_f32_native(
            tensor_f32.copy().tobytes(),
            block_size,
            -1.0,
            bit_depth,
        )

        dna_db = gen_ret[0]
        anchors_f16 = gen_ret[1]
        centroids = gen_ret[2]

        dna_bytes = dna_db if isinstance(dna_db, (bytes, bytearray)) else bytes(dna_db)
        anchors_bytes = (
            anchors_f16
            if isinstance(anchors_f16, (bytes, bytearray))
            else np.array(anchors_f16, dtype=np.float32).tobytes()
        )
        centroids_bytes = (
            centroids
            if isinstance(centroids, (bytes, bytearray))
            else np.array(centroids, dtype=np.float32).tobytes()
        )

        bias_bytes = b""
        if bias_obj is not None:
            b_f32 = np.frombuffer(bias_obj.data, dtype=np.float32)
            bias_bytes = b_f32.tobytes()

        off_dna, len_dna = add_blob_data(f"{name}_dna", dna_bytes)
        off_anc, len_anc = add_blob_data(f"{name}_anc", anchors_bytes)
        off_cen, len_cen = add_blob_data(f"{name}_cen", centroids_bytes)
        off_bias, len_bias = (
            add_blob_data(f"{name}_bias", bias_bytes) if bias_bytes else (0, 0)
        )

        tensor_directory.append(
            {
                "name": name,
                "out_features": out_features,
                "in_features": in_features,
                "block_size": block_size,
                "bit_depth": bit_depth,
                "offsets": {
                    "dna": [off_dna, len_dna],
                    "anc": [off_anc, len_anc],
                    "cen": [off_cen, len_cen],
                    "bias": [off_bias, len_bias],
                },
            }
        )

    # Export Embeddings
    print("  [~] Cuantizando token_embd a 4-bit...")
    process_layer_data(
        "token_embd",
        get_tensor_f32_matrix(tensors_by_name["token_embd.weight"]),
        bit_depth=4,
    )

    # Export Blocks
    for i in range(n_blocks):
        p = f"blk.{i}."

        # RMSNorms
        attn_norm = get_tensor_f32_matrix(
            tensors_by_name[f"blk.{i}.attn_norm.weight"]
        ).flatten()
        off_anorm, len_anorm = add_blob_data(f"{p}attn_norm", attn_norm.tobytes())
        tensor_directory.append(
            {"name": f"{p}attn_norm", "offsets": {"data": [off_anorm, len_anorm]}}
        )

        ffn_norm = get_tensor_f32_matrix(
            tensors_by_name[f"blk.{i}.ffn_norm.weight"]
        ).flatten()
        off_fnorm, len_fnorm = add_blob_data(f"{p}ffn_norm", ffn_norm.tobytes())
        tensor_directory.append(
            {"name": f"{p}ffn_norm", "offsets": {"data": [off_fnorm, len_fnorm]}}
        )

        # 1. Fused QKV
        w_q = get_tensor_f32_matrix(tensors_by_name[f"blk.{i}.attn_q.weight"])
        w_k = get_tensor_f32_matrix(tensors_by_name[f"blk.{i}.attn_k.weight"])
        w_v = get_tensor_f32_matrix(tensors_by_name[f"blk.{i}.attn_v.weight"])
        w_qkv = np.concatenate([w_q, w_k, w_v], axis=0)
        process_layer_data(p + "attn_qkv", w_qkv, bit_depth=4)

        # 2. Attn Output
        w_o = get_tensor_f32_matrix(tensors_by_name[f"blk.{i}.attn_output.weight"])
        process_layer_data(p + "attn_output", w_o, bit_depth=4)

        # 3. Fused GateUp
        w_gate = get_tensor_f32_matrix(tensors_by_name[f"blk.{i}.ffn_gate.weight"])
        w_up = get_tensor_f32_matrix(tensors_by_name[f"blk.{i}.ffn_up.weight"])
        w_gate_up = np.concatenate([w_gate, w_up], axis=0)
        process_layer_data(p + "ffn_gate_up", w_gate_up, bit_depth=4)

        # 4. FFN Down
        w_down = get_tensor_f32_matrix(tensors_by_name[f"blk.{i}.ffn_down.weight"])
        process_layer_data(p + "ffn_down", w_down, bit_depth=4)

        print(f"  [~] Bloque {i + 1}/{n_blocks} empaquetado.")
        gc.collect()

    # LM Head (con soporte para weight tying)
    print("  [~] Cuantizando output / lm_head a 4-bit...")
    lm_head_tensor = tensors_by_name.get(
        "output.weight",
        tensors_by_name.get("lm_head.weight", tensors_by_name["token_embd.weight"]),
    )
    process_layer_data("lm_head", get_tensor_f32_matrix(lm_head_tensor), bit_depth=4)

    # Final Norm (con fallback si no existe output_norm.weight)
    if "output_norm.weight" in tensors_by_name:
        output_norm = get_tensor_f32_matrix(
            tensors_by_name["output_norm.weight"]
        ).flatten()
    elif "norm.weight" in tensors_by_name:
        output_norm = get_tensor_f32_matrix(tensors_by_name["norm.weight"]).flatten()
    else:
        output_norm = np.ones(n_embd, dtype=np.float32)
    off_onorm, len_onorm = add_blob_data("output_norm", output_norm.tobytes())
    tensor_directory.append(
        {"name": "output_norm", "offsets": {"data": [off_onorm, len_onorm]}}
    )

    dir_json_bytes = json.dumps(tensor_directory).encode("utf-8")

    magic = b"GAJE"
    version = 0x000907
    flags = 0x0003
    num_tensors = len(tensor_directory)
    meta_len = len(metadata_json_bytes)
    dir_len = len(dir_json_bytes)

    header_fixed_size = 4096
    weights_offset = header_fixed_size + meta_len + dir_len
    if weights_offset % 4096 != 0:
        weights_offset = ((weights_offset // 4096) + 1) * 4096

    weights_len = len(blob_bytes)

    header_bin = bytearray(4096)
    struct.pack_into(
        "<4sIIIQQQQ",
        header_bin,
        0,
        magic,
        version,
        flags,
        num_tensors,
        meta_len,
        dir_len,
        weights_offset,
        weights_len,
    )

    with open(out_path, "wb") as f:
        f.write(header_bin)
        f.write(metadata_json_bytes)
        f.write(dir_json_bytes)

        current_pos = 4096 + meta_len + dir_len
        padding_needed = weights_offset - current_pos
        if padding_needed > 0:
            f.write(b"\x00" * padding_needed)

        f.write(blob_bytes)

    print(f"\n✅ Exportación Flat Zero-Copy para SmolLM2 Finalizada: {out_path}")
    print(
        f"  - Tamaño Total Archivo: {os.path.getsize(out_path) / (1024 * 1024):.2f} MB"
    )


if __name__ == "__main__":
    export_smollm2_flat()
