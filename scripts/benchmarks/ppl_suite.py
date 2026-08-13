"""PPL GAJE por modelo + paridad FP16 (HuggingFace) sobre corpus ES limpio."""
import os
import sys
import random
import numpy as np
import argparse
import json
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM

MODELS = {
    "qwen2_0_5b_q4_0_q8_0_embd": "models/production/qwen2_0_5b_q4_0_q8_0_embd.gaje.flat",
    "qwen2_5_1_5b_q4_0": "models/production/qwen2_5_1_5b_q4_0.gaje.flat",
    "qwen2_5_1_5b_q4_0_q8_0_embd": "models/production/qwen2_5_1_5b_q4_0_q8_0_embd.gaje.flat",
    "qwen2_5_3b_q4_0_q8_0_embd": "models/production/qwen2_5_3b_q4_0_q8_0_embd.gaje.flat",
    "smollm2_4bit": "models/production/smollm2_4bit.gaje.flat",
}
HF_REF = {
    "qwen2_0_5b": "Qwen/Qwen2-0.5B-Instruct",
    "qwen2_5_1_5b": "Qwen/Qwen2.5-1.5B-Instruct",
}


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def ppl_gaje(model, tokens):
    logits_seq = model.forward(tokens, clear_cache=True)
    logits_seq = logits_seq[:-1]
    target = tokens[1:]
    lp = []
    for i, tid in enumerate(target):
        probs = softmax(logits_seq[i])
        p = np.clip(probs[tid], 1e-10, 1.0)
        lp.append(np.log(p))
    return float(np.exp(-np.mean(lp))) if lp else None


def ppl_hf(model, tokenizer, tokens, device):
    import torch

    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(ids).logits[:, :-1, :]
    targets = torch.tensor(tokens[1:], dtype=torch.long, device=device)
    return float(torch.nn.functional.cross_entropy(logits[0], targets).exp())


def clean_line(ln):
    s = ln.strip()
    if not s or len(s) < 10:
        return None
    if s.startswith("#") or s.startswith("---") or s.startswith("==="):
        return None
    if "DATASET" in s.upper() or "COHERENCIA Y LÓGICA" in s.upper():
        return None
    return s


def load_samples(max_len, samples_n):
    lines = []
    for fn in ["dataset_es_ext.txt", "coherence_es.txt"]:
        with open(os.path.join(ROOT, "data/datasets", fn), encoding="utf-8") as f:
            lines += [c for ln in f if (c := clean_line(ln)) is not None]
    random.seed(42)
    random.shuffle(lines)
    return [ln[:max_len] for ln in lines[:samples_n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--skip_hf", action="store_true")
    ap.add_argument(
        "--only", type=str, default=None, help="Substring de modelo a evaluar (GAJE)"
    )
    args = ap.parse_args()

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    samples = load_samples(args.max_len, args.samples)
    print(f"[*] device={device} | {len(samples)} muestras | max_len={args.max_len}")

    out = {}

    for tag, path in MODELS.items():
        if args.only and args.only not in tag:
            continue
        t0 = time.time()
        llm = GenomicLLM.load_genomic(os.path.join(ROOT, path))
        tok = llm.tokenizer
        ppls = []
        for ln in samples:
            enc = tok.encode(ln, add_special_tokens=False)
            ids = enc.ids if hasattr(enc, "ids") else enc
            if len(ids) < 3:
                continue
            try:
                v = ppl_gaje(llm, ids)
                if v and not np.isinf(v):
                    ppls.append(v)
            except Exception:
                pass
        out[tag] = {"ppl_gaje": round(float(np.mean(ppls)), 2) if ppls else None}
        print(f"  GAJE {tag:<34} PPL={out[tag]['ppl_gaje']}  ({time.time()-t0:.0f}s)")

    if not args.skip_hf:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        for key, hf_id in HF_REF.items():
            if args.only and args.only not in key:
                continue
            t0 = time.time()
            print(f"[~] HF FP16 {hf_id}...")
            hf = (
                AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float16)
                .to(device)
                .eval()
            )
            ht = AutoTokenizer.from_pretrained(hf_id)
            ppls = []
            for ln in samples:
                ids = ht.encode(ln, add_special_tokens=False)[: args.max_len]
                if len(ids) < 3:
                    continue
                try:
                    ppls.append(ppl_hf(hf, ht, ids, device))
                except Exception:
                    pass
            out[key] = {"ppl_fp16": round(float(np.mean(ppls)), 2) if ppls else None}
            print(
                f"  HF FP16 {key:<26} PPL={out[key]['ppl_fp16']}  ({time.time()-t0:.0f}s)"
            )

    with open(os.path.join(os.path.dirname(__file__), "ppl_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== RESUMEN PPL ===")
    for k, v in out.items():
        print(f"  {k:<34} {v}")


if __name__ == "__main__":
    main()
