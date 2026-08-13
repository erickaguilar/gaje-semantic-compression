import os
import sys
import random
import numpy as np
import torch
import argparse

ROOT = "/home/erickaguilar/Documentos/gaje-semantic-compression"
sys.path.insert(0, os.path.join(ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def ppl_gaje(model, tokens):
    logits_seq = model.forward(tokens, clear_cache=True)
    logits_seq = logits_seq[:-1]
    target_tokens = tokens[1:]
    log_probs = []
    for i, target_id in enumerate(target_tokens):
        probs = softmax(logits_seq[i])
        p = np.clip(probs[target_id], 1e-10, 1.0)
        log_probs.append(np.log(p))
    if not log_probs:
        return None
    return float(np.exp(-np.mean(log_probs)))


def ppl_hf(model, tokenizer, tokens, device):
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(ids).logits[:, :-1, :]
    logits = logits[0]  # [seq-1, vocab]
    targets = torch.tensor(tokens[1:], dtype=torch.long, device=device)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    return float(loss.exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gaje",
        default=os.path.join(
            ROOT, "models/production/qwen2_0_5b_q4_0_q8_0_embd.gaje.flat"
        ),
    )
    ap.add_argument("--hf_id", default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument(
        "--files", nargs="+", default=["dataset_es_ext.txt", "coherence_es.txt"]
    )
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    def clean_line(ln):
        s = ln.strip()
        if not s or len(s) < 10:
            return None
        if s.startswith("#") or s.startswith("---") or s.startswith("==="):
            return None
        if "DATASET" in s.upper() or "COHERENCIA Y LÓGICA" in s.upper():
            return None
        return s

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] device: {device}")

    print("[~] Cargando modelo GAJE (Q4_0 + FP32 embd)...")
    gaje = GenomicLLM.load_genomic(args.gaje)
    gaje_tok = gaje.tokenizer

    print(f"[~] Cargando HF FP16: {args.hf_id}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(args.hf_id)
    hf = (
        AutoModelForCausalLM.from_pretrained(args.hf_id, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )

    # vocab alignment
    gaje_vocab = (
        set(gaje_tok.get_vocab().keys()) if hasattr(gaje_tok, "get_vocab") else set()
    )
    hf_vocab = set(hf_tok.get_vocab().keys())
    print(
        f"[*] GAJE vocab: {len(gaje_vocab)} | HF vocab: {len(hf_vocab)} | intersección: {len(gaje_vocab & hf_vocab)}"
    )

    random.seed(42)
    lines = []
    for fn in args.files:
        path = os.path.join(ROOT, "data/datasets", fn)
        with open(path, encoding="utf-8") as f:
            lines += [c for ln in f if (c := clean_line(ln)) is not None]
    random.shuffle(lines)
    samples = lines[: args.samples]

    gaje_ppls, hf_ppls = [], []
    for ln in samples:
        enc = gaje_tok.encode(ln, add_special_tokens=False)
        g_tok = (enc.ids if hasattr(enc, "ids") else enc)[: args.max_len]
        h_tok = hf_tok.encode(ln, add_special_tokens=False)[: args.max_len]
        if len(g_tok) < 3 or len(h_tok) < 3:
            continue
        try:
            g = ppl_gaje(gaje, g_tok)
            h = ppl_hf(hf, hf_tok, h_tok, device)
        except Exception as e:
            print("  skip:", ln[:30], e)
            continue
        gaje_ppls.append(g)
        hf_ppls.append(h)
        print(f"  {g:9.2f} vs {h:9.2f}   {ln[:45]}")

    if not gaje_ppls:
        print("Sin muestras válidas")
        return
    gaje_ppls, hf_ppls = np.array(gaje_ppls), np.array(hf_ppls)
    print("\n=== RESUMEN PPL (menor = mejor) ===")
    print(
        f"GAJE Q4_0+FP32 : {gaje_ppls.mean():.4f}  (media de {len(gaje_ppls)} muestras)"
    )
    print(f"HF FP16        : {hf_ppls.mean():.4f}")
    print(f"Ratio GAJE/FP16: {gaje_ppls.mean()/hf_ppls.mean():.4f}")
    print(f"Correlación    : {np.corrcoef(gaje_ppls, hf_ppls)[0,1]:.4f}")
    if gaje_ppls.mean() / hf_ppls.mean() < 1.05:
        print(
            "\n✨ SORPRESA CONFIRMADA: PPL dentro de 5% de FP16 a pesar de cuerpo 4-bit"
        )


if __name__ == "__main__":
    main()
