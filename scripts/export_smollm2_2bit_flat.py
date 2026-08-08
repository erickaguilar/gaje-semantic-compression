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

from gaje.nn.configs import ARCHITECTURES  # noqa: E402
from gaje.nn.stabilized import GenomicLayer, dequantize_q8_0  # noqa: E402


def export_smollm2_2bit_flat():
    gguf_path = os.path.join(
        PROJECT_ROOT, "data", "models", "smollm2-135m-instruct-fp16.gguf"
    )
    out_path = os.path.join(
        PROJECT_ROOT, "models", "production", "smollm2_2bit_flat.gaje.flat"
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("🚀 Exportador Flat Zero-Copy v0.9.7 para SmolLM2 135M (2-BIT)")
    print(f"  - Origen GGUF: {gguf_path}")
    print(f"  - Destino Flat: {out_path}")

    reader = gguf.GGUFReader(gguf_path)
    cfg = ARCHITECTURES["llama"]
    cfg.block_size = 32
    cfg.attn_bit_depth = 2
    cfg.ffn_bit_depth = 2
    cfg.ffn_anchor_threshold = -1.0

    n_embd = 576
    n_head = 9
    n_head_kv = 3
    n_blocks = 30
    eps = 1e-5
    rope_base = 100000.0

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
        elif "rope.freq_base" in field_name:
            rope_base = float(field.parts[field.data[0]][0])

    head_dim = n_embd // n_head
    print(
        f"[*] Parámetros SmolLM2: n_embd={n_embd}, n_head={n_head}, n_head_kv={n_head_kv}, head_dim={head_dim}, n_blocks={n_blocks}, rope_base={rope_base}"
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
            "rope_style": "split",
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

    def process_and_add_layer(name, layer):
        dna_b = (
            bytes(layer.linear.database)
            if hasattr(layer.linear, "database")
            else bytes(layer.dna_database)
        )
        c_b = np.array(
            layer.linear.centroids
            if hasattr(layer.linear, "centroids")
            else layer.dna_centroids,
            dtype=np.float32,
        ).tobytes()
        anc_b = layer.anchors_f16_bytes

        dna_off, dna_len = add_blob_data(f"{name}.dna", dna_b)
        c_off, c_len = add_blob_data(f"{name}.centroids", c_b)
        anc_off, anc_len = add_blob_data(f"{name}.anchors", anc_b)

        bias_val = getattr(layer, "bias", None)
        bias_off, bias_len = 0, 0
        if bias_val is not None and len(bias_val) > 0:
            b_bytes = np.array(bias_val, dtype=np.float32).tobytes()
            bias_off, bias_len = add_blob_data(f"{name}.bias", b_bytes)

        tensor_directory.append(
            {
                "name": name,
                "bit_depth": int(layer.bit_depth),
                "out_features": int(layer.out_features),
                "in_features": int(layer.in_features),
                "dna_off": int(dna_off),
                "dna_len": int(dna_len),
                "c_off": int(c_off),
                "c_len": int(c_len),
                "anc_off": int(anc_off),
                "anc_len": int(anc_len),
                "bias_off": int(bias_off),
                "bias_len": int(bias_len),
            }
        )

    def process_layer_data(
        name, tensor_obj, bit_depth=2, bias_obj=None, n_head=None, head_dim=None
    ):
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
            config=cfg,
            n_head=n_head,
            head_dim=head_dim,
        )
        process_and_add_layer(name, layer)
        del layer
        gc.collect()

    def get_tensor_f32_matrix(tensor_obj, n_h=None, h_d=None, is_q_k=False):
        if tensor_obj.tensor_type == gguf.GGMLQuantizationType.F32:
            raw_data = np.frombuffer(tensor_obj.data, dtype=np.float32)
        elif tensor_obj.tensor_type == gguf.GGMLQuantizationType.F16:
            raw_data = np.frombuffer(tensor_obj.data, dtype=np.float16).astype(
                np.float32
            )
        elif tensor_obj.tensor_type == gguf.GGMLQuantizationType.Q8_0:
            return dequantize_q8_0(tensor_obj, n_h, h_d, is_q_k, rope_style="split")
        else:
            raise ValueError(f"Tipo no soportado: {tensor_obj.tensor_type}")

        shape = tensor_obj.shape
        if len(shape) == 2:
            out_dim, in_dim = int(shape[1]), int(shape[0])
            raw_data = raw_data.reshape(out_dim, in_dim)

        if is_q_k and n_h is not None and h_d is not None:
            from gaje.utils.quantization import unpermute_to_split

            raw_data = unpermute_to_split(raw_data, n_h, h_d)
        return raw_data

    # Export Embeddings
    print("  [~] Exportando token_embd a 32-bit (FP32)...")
    process_layer_data(
        "token_embd",
        tensors_by_name["token_embd.weight"],
        bit_depth=32,
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
            {
                "name": f"{p}attn_norm",
                "bit_depth": 32,
                "out_features": int(n_embd),
                "in_features": 1,
                "dna_off": int(off_anorm),
                "dna_len": int(len_anorm),
                "c_off": 0,
                "c_len": 0,
                "anc_off": 0,
                "anc_len": 0,
                "bias_off": 0,
                "bias_len": 0,
            }
        )

        ffn_norm = get_tensor_f32_matrix(
            tensors_by_name[f"blk.{i}.ffn_norm.weight"]
        ).flatten()
        off_fnorm, len_fnorm = add_blob_data(f"{p}ffn_norm", ffn_norm.tobytes())
        tensor_directory.append(
            {
                "name": f"{p}ffn_norm",
                "bit_depth": 32,
                "out_features": int(n_embd),
                "in_features": 1,
                "dna_off": int(off_fnorm),
                "dna_len": int(len_fnorm),
                "c_off": 0,
                "c_len": 0,
                "anc_off": 0,
                "anc_len": 0,
                "bias_off": 0,
                "bias_len": 0,
            }
        )

        # 1. Attn Q, K, V (2-bit)
        process_layer_data(
            p + "attn_q",
            tensors_by_name[f"blk.{i}.attn_q.weight"],
            bit_depth=2,
            n_head=n_head,
            head_dim=head_dim,
        )
        process_layer_data(
            p + "attn_k",
            tensors_by_name[f"blk.{i}.attn_k.weight"],
            bit_depth=2,
            n_head=n_head_kv,
            head_dim=head_dim,
        )
        process_layer_data(
            p + "attn_v", tensors_by_name[f"blk.{i}.attn_v.weight"], bit_depth=2
        )

        # 2. Attn Output (2-bit)
        process_layer_data(
            p + "attn_output",
            tensors_by_name[f"blk.{i}.attn_output.weight"],
            bit_depth=2,
        )

        # 3. FFN Gate & Up (2-bit)
        process_layer_data(
            p + "ffn_gate", tensors_by_name[f"blk.{i}.ffn_gate.weight"], bit_depth=2
        )
        process_layer_data(
            p + "ffn_up", tensors_by_name[f"blk.{i}.ffn_up.weight"], bit_depth=2
        )

        # 4. FFN Down (2-bit)
        process_layer_data(
            p + "ffn_down",
            tensors_by_name[f"blk.{i}.ffn_down.weight"],
            bit_depth=2,
        )

        print(f"  [~] Bloque {i + 1}/{n_blocks} empaquetado (2-bit).")
        gc.collect()

    # LM Head (con soporte para weight tying)
    print("  [~] Exportando output / lm_head a 32-bit (FP32)...")
    lm_head_obj = tensors_by_name.get(
        "output.weight",
        tensors_by_name.get("lm_head.weight", tensors_by_name["token_embd.weight"]),
    )
    process_layer_data("lm_head", lm_head_obj, bit_depth=32)

    # Final Norm
    norm_tensor = tensors_by_name.get(
        "output_norm.weight", tensors_by_name.get("norm.weight", None)
    )
    if norm_tensor is not None:
        output_norm = get_tensor_f32_matrix(norm_tensor).flatten()
    else:
        output_norm = np.ones(n_embd, dtype=np.float32)

    off_onorm, len_onorm = add_blob_data("output_norm", output_norm.tobytes())
    tensor_directory.append(
        {
            "name": "output_norm",
            "bit_depth": 32,
            "out_features": int(n_embd),
            "in_features": 1,
            "dna_off": int(off_onorm),
            "dna_len": int(len_onorm),
            "c_off": 0,
            "c_len": 0,
            "anc_off": 0,
            "anc_len": 0,
            "bias_off": 0,
            "bias_len": 0,
        }
    )

    dir_json_bytes = json.dumps(tensor_directory).encode("utf-8")

    magic = b"GAJE"
    version = 0x000907
    flags = 0x0002  # Flag for 2-bit
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

    print(f"\n✅ Exportación Flat Zero-Copy para SmolLM2 2-Bit Finalizada: {out_path}")
    print(
        f"  - Tamaño Total Archivo: {os.path.getsize(out_path) / (1024 * 1024):.2f} MB"
    )


if __name__ == "__main__":
    export_smollm2_2bit_flat()
