import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM  # noqa: E402


def run_diagnostics_smollm2(model_path, model_name="SmolLM2-135M Organism"):
    print("\n=======================================================")
    print(f"🔬 DIAGNÓSTICO TOP-K AGREEMENT & TEACHER FORCING: {model_name}")
    print("=======================================================")

    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    print(f"[*] Cargando modelo PyTorch FP32 de referencia ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    hf_model.eval()

    print(f"[*] Cargando Organismo GAJE desde '{model_path}'...")
    gaje_llm = GenomicLLM.load_genomic(model_path)

    prompt = "<|im_start|>user\nWhat is the capital of France? Answer in one word.<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    print("\n--- TEST 1: TOP-K LOGITS RANKING EN PREFILL ---")
    inputs = torch.tensor([input_ids])
    with torch.no_grad():
        hf_outputs = hf_model(inputs)
        hf_logits = hf_outputs.logits[0, -1, :].numpy()

    # Prefill en GAJE
    gaje_logits = None
    for p_idx, tok_id in enumerate(input_ids):
        clear_cache = p_idx == 0
        gaje_logits = np.array(gaje_llm.rust_llm.forward(tok_id, clear_cache))

    # Medir CosSim de Logits
    cos_sim = np.dot(hf_logits, gaje_logits) / (
        np.linalg.norm(hf_logits) * np.linalg.norm(gaje_logits) + 1e-9
    )
    print(f"📊 Cosine Similarity de Logits Finales: {cos_sim:.6f}")

    # Top-K Ranking Comparison
    hf_top10 = np.argsort(hf_logits)[-10:][::-1]
    gaje_top10 = np.argsort(gaje_logits)[-10:][::-1]

    print("\n🏆 Top-10 Tokens Predichos por PyTorch FP32:")
    for rank, idx in enumerate(hf_top10, 1):
        tok_str = repr(tokenizer.decode([idx]))
        print(
            f"   #{rank:2d}: Token {idx:<6d} {tok_str:<20s} (logit = {hf_logits[idx]:.4f})"
        )

    print("\n🧬 Top-10 Tokens Predichos por GAJE Rust:")
    for rank, idx in enumerate(gaje_top10, 1):
        tok_str = repr(tokenizer.decode([idx]))
        print(
            f"   #{rank:2d}: Token {idx:<6d} {tok_str:<20s} (logit = {gaje_logits[idx]:.4f})"
        )

    # Métrica de Coincidencia
    top1_match = hf_top10[0] == gaje_top10[0]
    top5_intersect = len(set(hf_top10[:5]).intersection(set(gaje_top10[:5])))
    top10_intersect = len(set(hf_top10[:10]).intersection(set(gaje_top10[:10])))

    target_token_id = hf_top10[0]
    gaje_rank_of_target = np.where(gaje_top10 == target_token_id)[0]
    gaje_rank_str = (
        f"#{gaje_rank_of_target[0] + 1}"
        if len(gaje_rank_of_target) > 0
        else "Fuera del Top-10"
    )

    print("\n📈 MÉTRICAS DE RANKING:")
    print(f"   - Top-1 Match:          {'✅ SÍ' if top1_match else '❌ NO'}")
    print(
        f"   - Top-5 Intersección:   {top5_intersect}/5 ({top5_intersect / 5 * 100:.1f}%)"
    )
    print(
        f"   - Top-10 Intersección:  {top10_intersect}/10 ({top10_intersect / 10 * 100:.1f}%)"
    )
    print(
        f"   - Ranking de Target ('{tokenizer.decode([target_token_id])}'): {gaje_rank_str} en GAJE"
    )

    print("\n--- TEST 2: TEACHER FORCING DIVERGENCE OVER 10 STEPS ---")
    current_input_ids = list(input_ids)

    for step in range(10):
        inputs_tf = torch.tensor([current_input_ids])
        with torch.no_grad():
            outputs_tf = hf_model(inputs_tf)
            hf_l = outputs_tf.logits[0, -1, :].numpy()

        hf_next_tok = int(np.argmax(hf_l))
        gaje_l = np.array(
            gaje_llm.rust_llm.forward(
                current_input_ids[-1],
                step == 0 and len(current_input_ids) == len(input_ids),
            )
        )

        step_cos = np.dot(hf_l, gaje_l) / (
            np.linalg.norm(hf_l) * np.linalg.norm(gaje_l) + 1e-9
        )
        hf_top1 = np.argmax(hf_l)
        gaje_top1 = np.argmax(gaje_l)
        match = hf_top1 == gaje_top1

        print(
            f"Step {step + 1:02d} | Token Forzado HF: {hf_next_tok:<6d} ({repr(tokenizer.decode([hf_next_tok])):<12s}) | CosSim: {step_cos:.6f} | Match Top-1: {'✅' if match else '❌ (GAJE=' + repr(tokenizer.decode([gaje_top1])) + ')'}"
        )

        current_input_ids.append(hf_next_tok)


if __name__ == "__main__":
    target_model = os.path.join(PROJECT_ROOT, "models", "smollm2-135m-anchored.gaje")
    run_diagnostics_smollm2(target_model, "SmolLM2-135M (4-bit uniform)")
