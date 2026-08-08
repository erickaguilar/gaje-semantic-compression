import os
import sys
import gc
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM  # noqa: E402
from gaje.nn.configs import ARCHITECTURES  # noqa: E402


def build_smollm2_fp32():
    """Genera un organismo SmolLM2-135M con todas sus capas en FP32 puro (sin compresión)."""
    gguf_path = os.path.join(
        PROJECT_ROOT, "data", "models", "smollm2-135m-instruct-fp16.gguf"
    )
    fp32_output_path = os.path.join(PROJECT_ROOT, "models", "smollm2-135m-fp32.gaje")

    print(
        f"🧬 [gaje_diff] Generando baseline SmolLM2 FP32 nativo en: {fp32_output_path}..."
    )
    cfg = ARCHITECTURES["llama"]
    old_attn = cfg.attn_bit_depth
    old_ffn = cfg.ffn_bit_depth
    cfg.attn_bit_depth = 32
    cfg.ffn_bit_depth = 32

    try:
        llm = GenomicLLM(gguf_path)
        gc.collect()
        llm.save(fp32_output_path)
        print(f"✅ Baseline FP32 creado exitosamente en: {fp32_output_path}")
    finally:
        cfg.attn_bit_depth = old_attn
        cfg.ffn_bit_depth = old_ffn

    return fp32_output_path


def build_smollm2_mixed():
    """Genera un organismo SmolLM2-135M Mixed-Bit (4-bit atención, 2-bit FFN)."""
    gguf_path = os.path.join(
        PROJECT_ROOT, "data", "models", "smollm2-135m-instruct-fp16.gguf"
    )
    mixed_output_path = os.path.join(PROJECT_ROOT, "models", "smollm2-135m-mixed.gaje")

    print(
        f"🧬 [gaje_diff] Generando SmolLM2 Mixed-Bit (4-bit Attn, 2-bit FFN) en: {mixed_output_path}..."
    )
    cfg = ARCHITECTURES["llama"]
    old_attn = cfg.attn_bit_depth
    old_ffn = cfg.ffn_bit_depth
    cfg.attn_bit_depth = 4
    cfg.ffn_bit_depth = 2

    try:
        llm = GenomicLLM(gguf_path)
        gc.collect()
        llm.save(mixed_output_path)
        print(f"✅ Organismo Mixed-Bit creado exitosamente en: {mixed_output_path}")
    finally:
        cfg.attn_bit_depth = old_attn
        cfg.ffn_bit_depth = old_ffn

    return mixed_output_path


def build_smollm2_2bit_anchored(density=0.05):
    """Genera un organismo SmolLM2-135M en 2-bit puro con un porcentaje específico de Stability Anchors."""
    gguf_path = os.path.join(
        PROJECT_ROOT, "data", "models", "smollm2-135m-instruct-fp16.gguf"
    )
    output_path = os.path.join(
        PROJECT_ROOT, "models", f"smollm2-135m-2bit-anchored-{int(density*100)}pct.gaje"
    )

    print(
        f"🧬 [gaje_diff] Generando SmolLM2 2-bit ({density*100:.1f}% Anchors) en: {output_path}..."
    )
    cfg = ARCHITECTURES["llama"]
    old_attn = cfg.attn_bit_depth
    old_ffn = cfg.ffn_bit_depth
    old_anchor = cfg.anchor_threshold
    old_ffn_anchor = cfg.ffn_anchor_threshold
    
    cfg.attn_bit_depth = 2
    cfg.ffn_bit_depth = 2
    cfg.anchor_threshold = density
    cfg.ffn_anchor_threshold = density

    try:
        llm = GenomicLLM(gguf_path)
        gc.collect()
        llm.save(output_path)
        print(f"✅ Organismo 2-bit ({density*100:.1f}% Anchors) creado exitosamente.")
    finally:
        cfg.attn_bit_depth = old_attn
        cfg.ffn_bit_depth = old_ffn
        cfg.anchor_threshold = old_anchor
        cfg.ffn_anchor_threshold = old_ffn_anchor

    return output_path


def run_gaje_diff():
    print("\n=======================================================")
    print("🔬 GAJE DIFF: Certificación Capa por Capa (SmolLM2 FP32 y 2-Bit)")
    print("=======================================================")

    fp32_path = build_smollm2_fp32()
    mixed_path = build_smollm2_mixed()

    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    print(f"[*] Cargando PyTorch FP32 de referencia ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    hf_model.eval()

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    inputs = torch.tensor([input_ids])
    with torch.no_grad():
        hf_outputs = hf_model(inputs)
        hf_logits = hf_outputs.logits[0, -1, :].numpy()

    # --- EVALUACIÓN FP32 ---
    print(f"[*] Evaluando GAJE Rust FP32 Organism desde '{fp32_path}'...")
    gaje_llm_fp32 = GenomicLLM.load_genomic(fp32_path)
    gaje_llm_fp32.rust_llm.set_k_wta_ratio(0.0)

    gaje_logits_fp32 = None
    for p_idx, tok_id in enumerate(input_ids):
        clear_cache = p_idx == 0
        gaje_logits_fp32 = np.array(gaje_llm_fp32.rust_llm.forward(tok_id, clear_cache))

    cos_fp32 = np.dot(hf_logits, gaje_logits_fp32) / (
        np.linalg.norm(hf_logits) * np.linalg.norm(gaje_logits_fp32) + 1e-9
    )

    # --- EVALUACIÓN MIXED-BIT ---
    print(
        f"[*] Evaluando GAJE Rust Mixed-Bit (4b Attn, 2b FFN) desde '{mixed_path}'..."
    )
    gaje_llm_mixed = GenomicLLM.load_genomic(mixed_path)
    gaje_llm_mixed.rust_llm.set_k_wta_ratio(0.0)

    gaje_logits_mixed = None
    for p_idx, tok_id in enumerate(input_ids):
        clear_cache = p_idx == 0
        gaje_logits_mixed = np.array(
            gaje_llm_mixed.rust_llm.forward(tok_id, clear_cache)
        )

    cos_mixed = np.dot(hf_logits, gaje_logits_mixed) / (
        np.linalg.norm(hf_logits) * np.linalg.norm(gaje_logits_mixed) + 1e-9
    )

    hf_top1 = int(np.argmax(hf_logits))
    fp32_top1 = int(np.argmax(gaje_logits_fp32))
    mixed_top1 = int(np.argmax(gaje_logits_mixed))

    # --- EVALUACIÓN MULTI-DENSIDAD 2-BIT ---
    densities = [-1.0, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results_2bit = []

    for density in densities:
        try:
            path_2bit = build_smollm2_2bit_anchored(density)
            print(f"[*] Evaluando 2-bit con {density*100:.1f}% anclas..." if density >= 0 else "[*] Evaluando 2-bit puro (sin anclas)...")
            llm_2bit = GenomicLLM.load_genomic(path_2bit)
            llm_2bit.rust_llm.set_k_wta_ratio(0.0)

            logits_2bit = None
            for p_idx, tok_id in enumerate(input_ids):
                clear_cache = p_idx == 0
                logits_2bit = np.array(llm_2bit.rust_llm.forward(tok_id, clear_cache))

            cos_2 = np.dot(hf_logits, logits_2bit) / (
                np.linalg.norm(hf_logits) * np.linalg.norm(logits_2bit) + 1e-9
            )
            top1_2 = int(np.argmax(logits_2bit))
            match = "✅ SÍ" if top1_2 == hf_top1 else "❌ NO"
            pred_tok = tokenizer.decode([top1_2])

            name_str = "Puro (Sin Anclas)" if density < 0 else f"{density*100:.1f}%"
            results_2bit.append({
                "density": name_str,
                "cossim": f"{cos_2:.6f}",
                "pred": f"'{pred_tok}' ({top1_2})",
                "match": match
            })
            
            # Delete and collect
            del llm_2bit
            gc.collect()
        except Exception as e:
            print(f"❌ Error evaluando {density*100:.1f}%: {e}")

    print("\n=======================================================")
    print("GAJE Validation Report: Curva de Estabilidad 2-Bit")
    print("=======================================================")
    print("Baseline:")
    print(f"  - HF PyTorch FP32: '{tokenizer.decode([hf_top1])}' ({hf_top1})")
    print(f"  - GAJE FP32: CosSim={cos_fp32:.6f}, Pred='{tokenizer.decode([fp32_top1])}'")
    print(f"  - GAJE Mixed-Bit: CosSim={cos_mixed:.6f}, Pred='{tokenizer.decode([mixed_top1])}'")
    print("\nCurva de Similitud 2-Bit:")
    print("| Densidad de Anclas | Cosine Similarity | Predicción Top-1 | Match vs HF |")
    print("| :---: | :---: | :--- | :---: |")
    for r in results_2bit:
        print(f"| {r['density']} | {r['cossim']} | {r['pred']} | {r['match']} |")


if __name__ == "__main__":
    run_gaje_diff()
