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


def export_qwen2_5_1_5b_flat():
    gguf_path = os.path.join(
        PROJECT_ROOT, "data", "models", "qwen2.5-1.5b-instruct-fp16.gguf"
    )
    out_path = os.path.join(
        PROJECT_ROOT, "models", "production", "qwen2_5_1_5b_q4_0.gaje.flat"
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("🚀 Exportador Flat Zero-Copy v0.9.7 para Qwen2.5 1.5B Instruct (.gaje.flat)")
    print(f"  - Origen GGUF: {gguf_path}")
    print(f"  - Destino Flat: {out_path}")

    reader = gguf.GGUFReader(gguf_path)
    cfg = ARCHITECTURES["qwen2"]
    cfg.tokenizer_id = "Qwen/Qwen2.5-1.5B-Instruct"
    cfg.attn_bit_depth = 4
    cfg.ffn_bit_depth = 4
    cfg.ffn_anchor_threshold = -1.0

    n_embd = 1536
    n_head = 12
    n_head_kv = 2
    n_blocks = 28
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
        elif "rope.freq_base" in field_name:
            rope_base = float(field.parts[field.data[0]][0])

    head_dim = n_embd // n_head
    print(
        f"[*] Parámetros Qwen2.5: n_embd={n_embd}, n_head={n_head}, n_head_kv={n_head_kv}, head_dim={head_dim}, n_blocks={n_blocks}, eps={eps}, rope_base={rope_base}"
    )

    tensors_by_name = {t.name: t for t in reader.tensors}

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        tokenizer_str = (
            tokenizer.backend_tokenizer.to_str()
            if hasattr(tokenizer, "backend_tokenizer")
            else ""
        )
    except Exception:
        tokenizer_str = ""

    metadata_dict = {
        "config": {
            "name": "qwen2",
            "version": "0.9.7",
            "tokenizer_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "rope_base": rope_base,
            "rope_style": "interleaved",
            "ffn_act": "swiglu",
            "use_genomic_norm": False,
        },
        "n_embd": n_embd,
        "n_head": n_head,
        "n_head_kv": n_head_kv,
        "n_blocks": n_blocks,
        "vocab_size": 151936,
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
        name, tensor_obj, bit_depth=4, bias_obj=None, quant_format=1
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
            quant_format=quant_format if bit_depth == 4 else 0,
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
            raise ValueError(f"Unsupported quantization type: {tensor_obj.tensor_type}")

        out_f = (
            tensor_obj.shape[1] if len(tensor_obj.shape) == 2 else tensor_obj.shape[0]
        )
        in_f = tensor_obj.shape[0] if len(tensor_obj.shape) == 2 else 1
        w_matrix = raw_data.reshape(out_f, in_f)
        if is_q_k and n_h is not None and h_d is not None:
            from gaje.utils.quantization import unpermute_to_split

            w_matrix = unpermute_to_split(w_matrix, n_h, h_d)
        return w_matrix

    print("[*] Empaquetando token_embd y lm_head...")
    process_layer_data("token_embd", tensors_by_name["token_embd.weight"], bit_depth=32)

    output_norm_tensor = tensors_by_name["output_norm.weight"]
    out_norm_bytes = (
        np.frombuffer(output_norm_tensor.data, dtype=np.float32).tobytes()
        if output_norm_tensor.tensor_type == gguf.GGMLQuantizationType.F32
        else np.frombuffer(output_norm_tensor.data, dtype=np.float16)
        .astype(np.float32)
        .tobytes()
    )
    norm_off, norm_len = add_blob_data("output_norm", out_norm_bytes)
    tensor_directory.append(
        {
            "name": "output_norm",
            "bit_depth": 32,
            "out_features": n_embd,
            "in_features": 1,
            "dna_off": norm_off,
            "dna_len": norm_len,
            "c_off": 0,
            "c_len": 0,
            "anc_off": 0,
            "anc_len": 0,
            "bias_off": 0,
            "bias_len": 0,
        }
    )

    lm_head_tensor = tensors_by_name.get(
        "lm_head.weight", tensors_by_name["token_embd.weight"]
    )
    process_layer_data("lm_head", lm_head_tensor, bit_depth=32)

    for i in range(n_blocks):
        p = f"blk.{i}."
        for norm_suffix in ["attn_norm", "ffn_norm"]:
            g_norm_key = (
                f"blk.{i}.attn_norm.weight"
                if norm_suffix == "attn_norm"
                else f"blk.{i}.ffn_norm.weight"
            )
            if g_norm_key in tensors_by_name:
                t = tensors_by_name[g_norm_key]
                norm_b = (
                    np.frombuffer(t.data, dtype=np.float32).tobytes()
                    if t.tensor_type == gguf.GGMLQuantizationType.F32
                    else np.frombuffer(t.data, dtype=np.float16)
                    .astype(np.float32)
                    .tobytes()
                )
                no_off, no_len = add_blob_data(p + norm_suffix, norm_b)
                tensor_directory.append(
                    {
                        "name": p + norm_suffix,
                        "bit_depth": 32,
                        "out_features": n_embd,
                        "in_features": 1,
                        "dna_off": no_off,
                        "dna_len": no_len,
                        "c_off": 0,
                        "c_len": 0,
                        "anc_off": 0,
                        "anc_len": 0,
                        "bias_off": 0,
                        "bias_len": 0,
                    }
                )

        # 1. Fused QKV (4-bit)
        w_q = get_tensor_f32_matrix(
            tensors_by_name[f"blk.{i}.attn_q.weight"], n_head, head_dim, is_q_k=False
        )
        w_k = get_tensor_f32_matrix(
            tensors_by_name[f"blk.{i}.attn_k.weight"], n_head_kv, head_dim, is_q_k=False
        )
        w_v = get_tensor_f32_matrix(
            tensors_by_name[f"blk.{i}.attn_v.weight"], is_q_k=False
        )
        w_qkv = np.concatenate([w_q, w_k, w_v], axis=0)

        b_q_obj = tensors_by_name.get(f"blk.{i}.attn_q.bias", None)
        b_k_obj = tensors_by_name.get(f"blk.{i}.attn_k.bias", None)
        b_v_obj = tensors_by_name.get(f"blk.{i}.attn_v.bias", None)
        b_qkv = None
        if b_q_obj and b_k_obj and b_v_obj:
            bq = np.frombuffer(b_q_obj.data, dtype=np.float32)
            bk = np.frombuffer(b_k_obj.data, dtype=np.float32)
            bv = np.frombuffer(b_v_obj.data, dtype=np.float32)
            b_qkv = np.concatenate([bq, bk, bv], axis=0)

        process_layer_data(p + "attn_qkv", w_qkv, bit_depth=4, bias_obj=b_qkv)

        # 2. Attn Output (4-bit)
        w_o_obj = tensors_by_name[f"blk.{i}.attn_output.weight"]
        b_o_obj = tensors_by_name.get(f"blk.{i}.attn_output.bias", None)
        process_layer_data(p + "attn_output", w_o_obj, bit_depth=4, bias_obj=b_o_obj)

        # 3. Fused GateUp (4-bit)
        w_gate = get_tensor_f32_matrix(tensors_by_name[f"blk.{i}.ffn_gate.weight"])
        w_up = get_tensor_f32_matrix(tensors_by_name[f"blk.{i}.ffn_up.weight"])
        w_gate_up = np.concatenate([w_gate, w_up], axis=0)
        process_layer_data(p + "ffn_gate_up", w_gate_up, bit_depth=4, bias_obj=None)

        # 4. FFN Down (4-bit)
        process_layer_data(
            p + "ffn_down",
            tensors_by_name[f"blk.{i}.ffn_down.weight"],
            bit_depth=4,
            bias_obj=None,
        )

        print(f"  [~] Bloque {i+1}/{n_blocks} empaquetado en binario plano (4-bit).")
        gc.collect()

    dir_json_bytes = json.dumps(tensor_directory).encode("utf-8")

    magic = b"GAJE"
    version = 0x000908
    flags = 0x0003  # bit 0: fused_qkv, bit 1: fused_gateup
    num_tensors = len(tensor_directory)
    meta_len = len(metadata_json_bytes)
    dir_len = len(dir_json_bytes)

    header_fixed_size = 4096
    weights_offset = header_fixed_size + meta_len + dir_len
    if weights_offset % 4096 != 0:
        weights_offset = ((weights_offset // 4096) + 1) * 4096

    weights_len = len(blob_bytes)

    group_size = 32
    quant_format = 1  # 1 = Q4_0 (Variant B: scale + min + qs)

    header_bin = bytearray(4096)
    struct.pack_into(
        "<4sIIIQQQQIIIIIIII",
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
        group_size,
        quant_format,
        4,  # arch_family = Qwen2_5
        n_embd,
        n_head,
        n_head_kv,
        n_blocks,
        0,  # arch_qk_permute = false
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

    print(
        f"\n✅ Exportación Flat Zero-Copy Qwen2.5-1.5B Finalizada Exitosamente: {out_path}"
    )
    print(f"  - Tamaño Total Archivo: {os.path.getsize(out_path) / (1024*1024):.2f} MB")
    print(f"  - Offset de Pesos (Alineado a 4KB): {weights_offset} bytes")


if __name__ == "__main__":
    export_qwen2_5_1_5b_flat()
