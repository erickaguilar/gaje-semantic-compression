import os
import sys
import gc
import json
import struct
import numpy as np
import gguf
import argparse
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

import gaje.core._impl as core  # noqa: E402
from gaje.nn.configs import ARCHITECTURES  # noqa: E402
from gaje.nn.stabilized import dequantize_q8_0  # noqa: E402


def build_q2_0():
    """Exporta un modelo .gaje.flat Q2_0 (2 bits/peso + scale/min por bloque de 32).

    El cuerpo se cuantiza con `quantize_q2_0_native` (Q2_0Block: 12 bytes/bloque,
    ~3 bits/peso); embeddings, lm_head y normas quedan en FP32. Cabecera
    quant_format=3, bit_depth=2 y centroids vacíos por tensor -> el lector Rust
    construye WeightDatabase::GenomicQ2_0.
    """
    parser = argparse.ArgumentParser(description="Exportador Flat Q2_0 (2 bits/peso)")
    parser.add_argument("--input", type=str, help="Ruta al archivo origen GGUF")
    parser.add_argument("--output", type=str, help="Ruta al archivo destino .gaje.flat")
    parser.add_argument("--tokenizer", type=str, help="Hugging Face tokenizer ID")
    parser.add_argument(
        "--quant-embed",
        action="store_true",
        help="Cuantizar token_embd y lm_head a Q8_0 (como el Q4_0 de referencia) en vez de FP32",
    )
    args = parser.parse_args()

    gguf_path = args.input or os.path.join(
        PROJECT_ROOT, "data", "models", "qwen2-0_5b-instruct-fp16.gguf"
    )
    out_path = args.output or os.path.join(
        PROJECT_ROOT, "models", "production", "qwen2_0_5b_q2_0.gaje.flat"
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not os.path.exists(gguf_path):
        print(f"❌ Error: No existe el GGUF en {gguf_path}")
        sys.exit(1)

    print("🧬 Exportador Flat Q2_0 (2 bits/peso + scale/min por bloque de 32)")
    print(f"  - Origen GGUF: {gguf_path}")
    print(f"  - Destino Flat: {out_path}")

    reader = gguf.GGUFReader(gguf_path)
    arch_name = "qwen2"
    for field_name, field in reader.fields.items():
        if field_name == "general.architecture":
            val = field.parts[field.data[0]][0]
            arch_name = val.decode("utf-8") if isinstance(val, bytes) else str(val)

    n_embd, n_head, n_head_kv, n_blocks, eps, rope_base = (
        896,
        14,
        2,
        24,
        1e-6,
        1000000.0,
    )
    for field_name, field in reader.fields.items():
        val = field.parts[field.data[0]][0]
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        if "embedding_length" in field_name:
            n_embd = int(val)
        elif "head_count" in field_name and "head_count_kv" not in field_name:
            n_head = int(val)
        elif "head_count_kv" in field_name:
            n_head_kv = int(val)
        elif "block_count" in field_name:
            n_blocks = int(val)
        elif "layer_norm_rms_epsilon" in field_name:
            eps = float(val)
        elif "rope.frequency_base" in field_name:
            rope_base = float(val)

    head_dim = n_embd // n_head
    model_name_lower = os.path.basename(gguf_path).lower()
    if "qwen2.5" in model_name_lower or "qwen2.5" in arch_name.lower():
        arch_family, qk_permute = 4, False
    elif "qwen2" in model_name_lower or "qwen2" in arch_name.lower():
        arch_family, qk_permute = 3, False
    elif "smollm2" in model_name_lower or "smollm2" in arch_name.lower():
        arch_family, qk_permute = 2, True
    elif "smollm" in model_name_lower or "smollm" in arch_name.lower():
        arch_family, qk_permute = 2, True
    elif "gemma" in model_name_lower or "gemma" in arch_name.lower():
        arch_family, qk_permute = 5, False
    elif "llama" in model_name_lower or "llama" in arch_name.lower():
        arch_family, qk_permute = 1, True
    else:
        arch_family, qk_permute = 6, False

    print(
        f"[*] Parámetros: n_embd={n_embd}, n_head={n_head}, n_head_kv={n_head_kv}, "
        f"n_blocks={n_blocks} | familia={arch_family}, qk_permute={qk_permute}"
    )

    tokenizer_map = {
        1: "meta-llama/Llama-3.2-1B-Instruct",
        2: "HuggingFaceTB/SmolLM2-135M-Instruct",
        3: "Qwen/Qwen2-0.5B-Instruct",
        4: "Qwen/Qwen2.5-1.5B-Instruct",
        5: "google/gemma-2-2b-it",
        6: "Qwen/Qwen2-0.5B-Instruct",
    }
    if arch_family == 4 and "3b" in model_name_lower:
        tokenizer_id = args.tokenizer or "Qwen/Qwen2.5-3B-Instruct"
    else:
        tokenizer_id = args.tokenizer or tokenizer_map.get(
            arch_family, "Qwen/Qwen2-0.5B-Instruct"
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception as e:
        print(f"[!] Warning: tokenizer {tokenizer_id} falló ({e}). Usando Qwen2-0.5B.")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    tokenizer_str = (
        tokenizer.backend_tokenizer.to_str()
        if hasattr(tokenizer, "backend_tokenizer")
        else ""
    )

    tensors_by_name = {t.name: t for t in reader.tensors}
    embd_tensor = tensors_by_name["token_embd.weight"]
    vocab_size = int(
        embd_tensor.shape[1] if len(embd_tensor.shape) == 2 else embd_tensor.shape[0]
    )
    if vocab_size < int(embd_tensor.shape[0]) and len(embd_tensor.shape) == 2:
        vocab_size = int(embd_tensor.shape[0])

    arch_key = {"2": "smollm", "1": "llama", "5": "llama"}.get(
        str(arch_family), "qwen2"
    )
    _cfg = ARCHITECTURES.get(arch_key, ARCHITECTURES["qwen2"])

    metadata_dict = {
        "config": {
            "name": arch_name,
            "version": "0.9.8-q2",
            "tokenizer_id": tokenizer_id,
            "rope_base": rope_base,
            "ffn_act": "swiglu" if arch_family != 5 else "geglu",
            "use_genomic_norm": False,
        },
        "n_embd": n_embd,
        "n_head": n_head,
        "n_head_kv": n_head_kv,
        "n_blocks": n_blocks,
        "vocab_size": vocab_size,
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

    def add_linear_tensor(name, weights_f32, bit_depth=2):
        """Cuantiza una matriz f32 a bloques Q2_0 y la registra con centroids vacíos."""
        if weights_f32.ndim == 2:
            out_f, in_f = weights_f32.shape
        else:
            out_f, in_f = weights_f32.shape[0], 1
        flat = np.ascontiguousarray(weights_f32.reshape(-1), dtype=np.float32)
        # La capa con el tensor debe tener dimensión divisible por 32 (como Q4_0)
        if flat.size % 32 != 0:
            pad = 32 - (flat.size % 32)
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
            # Q2_0 exige exactitud: si no es divisible, truncar es inválido
            print(
                f"[!] {name}: {flat.size} no divisible por 32, rellenando a {flat.size}"
            )
        dna = core.quantize_q2_0_native(flat.tobytes())

        dna_off, dna_len = add_blob_data(f"{name}.dna", dna)
        c_off, c_len = add_blob_data(f"{name}.centroids", b"")  # vacíos -> GenomicQ2_0
        anc_off, anc_len = add_blob_data(f"{name}.anchors", b"")
        tensor_directory.append(
            {
                "name": name,
                "bit_depth": bit_depth,
                "out_features": int(out_f),
                "in_features": int(in_f),
                "dna_off": int(dna_off),
                "dna_len": int(dna_len),
                "c_off": int(c_off),
                "c_len": int(c_len),
                "anc_off": int(anc_off),
                "anc_len": int(anc_len),
                "bias_off": 0,
                "bias_len": 0,
            }
        )
        del dna
        gc.collect()

    def add_f32_tensor(name, weights_f32, out_f=None, in_f=1):
        if out_f is None:
            out_f = weights_f32.shape[0] if weights_f32.ndim > 1 else len(weights_f32)
        raw = np.ascontiguousarray(weights_f32, dtype=np.float32).tobytes()
        off, ln = add_blob_data(f"{name}.raw", raw)
        tensor_directory.append(
            {
                "name": name,
                "bit_depth": 32,
                "out_features": int(out_f),
                "in_features": int(in_f),
                "dna_off": int(off),
                "dna_len": int(ln),
                "c_off": 0,
                "c_len": 0,
                "anc_off": 0,
                "anc_len": 0,
                "bias_off": 0,
                "bias_len": 0,
            }
        )

    def add_q8_tensor(name, weights_f32):
        """Cuantiza a bloques Q8_0 (scale + i8, sin centroids) para embd/lm_head."""
        if weights_f32.ndim == 2:
            out_f, in_f = weights_f32.shape
        else:
            out_f, in_f = weights_f32.shape[0], 1
        flat = np.ascontiguousarray(weights_f32.reshape(-1), dtype=np.float32)
        if flat.size % 32 != 0:
            flat = np.concatenate(
                [flat, np.zeros(32 - (flat.size % 32), dtype=np.float32)]
            )
        dna = core.quantize_q8_0_native(flat.tobytes())
        dna_off, dna_len = add_blob_data(f"{name}.dna", dna)
        c_off, c_len = add_blob_data(f"{name}.centroids", b"")
        anc_off, anc_len = add_blob_data(f"{name}.anchors", b"")
        tensor_directory.append(
            {
                "name": name,
                "bit_depth": 8,
                "out_features": int(out_f),
                "in_features": int(in_f),
                "dna_off": int(dna_off),
                "dna_len": int(dna_len),
                "c_off": int(c_off),
                "c_len": int(c_len),
                "anc_off": int(anc_off),
                "anc_len": int(anc_len),
                "bias_off": 0,
                "bias_len": 0,
            }
        )
        del dna
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
    embd_mat = get_tensor_f32_matrix(tensors_by_name["token_embd.weight"])
    if args.quant_embed:
        add_q8_tensor("token_embd", embd_mat)
    else:
        add_f32_tensor(
            "token_embd", embd_mat, out_f=embd_mat.shape[0], in_f=embd_mat.shape[1]
        )
    del embd_mat
    gc.collect()

    out_norm_t = tensors_by_name["output_norm.weight"]
    out_norm = (
        np.frombuffer(out_norm_t.data, dtype=np.float32)
        if out_norm_t.tensor_type == gguf.GGMLQuantizationType.F32
        else np.frombuffer(out_norm_t.data, dtype=np.float16).astype(np.float32)
    )
    add_f32_tensor("output_norm", out_norm, out_f=n_embd)

    lm_head_t = tensors_by_name.get(
        "lm_head.weight", tensors_by_name["token_embd.weight"]
    )
    lm_mat = get_tensor_f32_matrix(lm_head_t)
    if args.quant_embed:
        add_q8_tensor("lm_head", lm_mat)
    else:
        add_f32_tensor("lm_head", lm_mat, out_f=lm_mat.shape[0], in_f=lm_mat.shape[1])
    del lm_mat
    gc.collect()

    for i in range(n_blocks):
        p = f"blk.{i}."
        for norm_suffix in ["attn_norm", "ffn_norm"]:
            g_norm_key = (
                f"blk.{i}.attn_norm.weight"
                if norm_suffix == "attn_norm"
                else f"blk.{i}.ffn_norm.weight"
            )
            t = tensors_by_name[g_norm_key]
            nb = (
                np.frombuffer(t.data, dtype=np.float32)
                if t.tensor_type == gguf.GGMLQuantizationType.F32
                else np.frombuffer(t.data, dtype=np.float16).astype(np.float32)
            )
            add_f32_tensor(p + norm_suffix, nb, out_f=n_embd)

        w_q = get_tensor_f32_matrix(
            tensors_by_name[f"blk.{i}.attn_q.weight"],
            n_head,
            head_dim,
            is_q_k=qk_permute,
        )
        w_k = get_tensor_f32_matrix(
            tensors_by_name[f"blk.{i}.attn_k.weight"],
            n_head_kv,
            head_dim,
            is_q_k=qk_permute,
        )
        w_v = get_tensor_f32_matrix(
            tensors_by_name[f"blk.{i}.attn_v.weight"], is_q_k=False
        )
        w_qkv = np.concatenate([w_q, w_k, w_v], axis=0)
        add_linear_tensor(p + "attn_qkv", w_qkv)
        del w_q, w_k, w_v, w_qkv

        w_o = get_tensor_f32_matrix(tensors_by_name[f"blk.{i}.attn_output.weight"])
        add_linear_tensor(p + "attn_output", w_o)
        del w_o

        w_gate = get_tensor_f32_matrix(tensors_by_name[f"blk.{i}.ffn_gate.weight"])
        w_up = get_tensor_f32_matrix(tensors_by_name[f"blk.{i}.ffn_up.weight"])
        w_gate_up = np.concatenate([w_gate, w_up], axis=0)
        add_linear_tensor(p + "ffn_gate_up", w_gate_up)
        del w_gate, w_up, w_gate_up

        w_down = get_tensor_f32_matrix(tensors_by_name[f"blk.{i}.ffn_down.weight"])
        add_linear_tensor(p + "ffn_down", w_down)
        del w_down
        gc.collect()
        print(f"  [~] Bloque {i+1}/{n_blocks} empaquetado a Q2_0.")

    dir_json_bytes = json.dumps(tensor_directory).encode("utf-8")

    header_bin = bytearray(4096)
    struct.pack_into(
        "<4sIIIQQQQIIIIIIII",
        header_bin,
        0,
        b"GAJE",
        0x000908,
        0x0003,
        len(tensor_directory),
        len(metadata_json_bytes),
        len(dir_json_bytes),
        0,
        0,  # weights_offset/weights_len rellenados tras calcular
        32,
        3,  # group_size, quant_format=3 (Q2_0)
        arch_family,
        n_embd,
        n_head,
        n_head_kv,
        n_blocks,
        1 if qk_permute else 0,
    )

    header_fixed_size = 4096
    weights_offset = header_fixed_size + len(metadata_json_bytes) + len(dir_json_bytes)
    if weights_offset % 4096 != 0:
        weights_offset = ((weights_offset // 4096) + 1) * 4096
    weights_len = len(blob_bytes)

    struct.pack_into("<QQ", header_bin, 32, weights_offset, weights_len)

    with open(out_path, "wb") as f:
        f.write(header_bin)
        f.write(metadata_json_bytes)
        f.write(dir_json_bytes)
        current_pos = 4096 + len(metadata_json_bytes) + len(dir_json_bytes)
        if weights_offset > current_pos:
            f.write(b"\x00" * (weights_offset - current_pos))
        f.write(blob_bytes)

    print(f"\n✅ Exportación Flat Q2_0 finalizada: {out_path}")
    print(f"  - Tamaño: {os.path.getsize(out_path) / (1024*1024):.2f} MB")


if __name__ == "__main__":
    build_q2_0()
