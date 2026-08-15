"""PPL unificada entre motores: GAJE, HuggingFace FP16 y llama.cpp.

Protocolo común:
  1. Tokenizar el corpus UNA vez con el tokenizer HF (Qwen2-0.5B).
  2. Truncar a un tope común de tokens y de-tokenizar a texto unificado.
  3. Cada motor calcula PPL de SECUENCIA COMPLETA (sin ventana) sobre el MISMO texto.

Uso:
    python scripts/benchmarks/ppl_unified.py [--max_tokens 2048]
    # luego llama-perplexity -m <gguf> -f unified_corpus.txt -c 8192 --seed 42
"""
import os
import sys
import argparse
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM

HF_ID = "Qwen/Qwen2-0.5B-Instruct"
GAJE_PATH = "models/production/qwen2_0_5b_q4_0_q8_0_embd.gaje.flat"


def build_corpus(max_tokens, out_text):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_ID)
    raw = open(out_text, encoding="utf-8").read()
    ids = tok.encode(raw, add_special_tokens=False)
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
    unified_text = tok.decode(ids, skip_special_tokens=True)
    # guardar el texto truncado para que llama.cpp use el MISMO contenido
    unified_path = os.path.join(os.path.dirname(out_text), "corpus_unified.txt")
    with open(unified_path, "w", encoding="utf-8") as f:
        f.write(unified_text)
    return ids, unified_path


def ppl_gaje(model, tokens):
    logits = model.forward(tokens, clear_cache=True)[:-1]
    target = tokens[1:]
    lp = []
    for i, tid in enumerate(target):
        row = logits[i]
        e = np.exp(row - np.max(row))
        p = np.clip(e[tid] / e.sum(), 1e-10, 1.0)
        lp.append(np.log(p))
    return float(np.exp(-np.mean(lp))), len(lp)


def ppl_hf(model, tokenizer, tokens, device):
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(ids).logits[:, :-1, :]
    target = torch.tensor(tokens[1:], dtype=torch.long, device=device)
    loss = torch.nn.functional.cross_entropy(logits[0], target)
    return float(loss.exp()), len(tokens) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--which", choices=["gaje", "hf", "all"], default="all")
    ap.add_argument(
        "--corpus", default=os.path.join(ROOT, "data/datasets/_tmp_unified_raw.txt")
    )
    args = ap.parse_args()

    # corpus crudo (mismas líneas ya limpias)
    lines = []
    for fn in ["dataset_es_ext.txt", "coherence_es.txt"]:
        with open(os.path.join(ROOT, "data/datasets", fn), encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if len(s) >= 10 and not s.startswith(("#", "-", "=")):
                    if (
                        "DATASET" not in s.upper()
                        and "COHERENCIA Y LÓGICA" not in s.upper()
                    ):
                        lines.append(s)
    with open(args.corpus, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    ids, unified_path = build_corpus(args.max_tokens, args.corpus)
    print(f"[*] tokens unificados: {len(ids)} -> {unified_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # GAJE (cargar solo, liberar antes de HF)
    if args.which in ("gaje", "all"):
        llm = GenomicLLM.load_genomic(os.path.join(ROOT, GAJE_PATH))
        enc = llm.tokenizer.encode(
            open(unified_path, encoding="utf-8").read(), add_special_tokens=False
        )
        g_ids = enc.ids if hasattr(enc, "ids") else enc
        g_ids = g_ids[: args.max_tokens]
        ppl_g, n_g = ppl_gaje(llm, g_ids)
        print(f"GAJE Q4_0+Q8_0-emb : PPL={ppl_g:.3f}  tokens={n_g}")
        del llm
        import gc

        gc.collect()

    # HF FP16
    if args.which in ("hf", "all"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf = (
            AutoModelForCausalLM.from_pretrained(HF_ID, dtype=torch.float16)
            .to(device)
            .eval()
        )
        ht = AutoTokenizer.from_pretrained(HF_ID)
        h_ids = ht.encode(
            open(unified_path, encoding="utf-8").read(), add_special_tokens=False
        )[: args.max_tokens]
        ppl_h, n_h = ppl_hf(hf, ht, h_ids, device)
        print(f"HF FP16            : PPL={ppl_h:.3f}  tokens={n_h}")

    if args.which == "all":
        print(f"Ratio GAJE/FP16    : {ppl_g/ppl_h:.3f}")


if __name__ == "__main__":
    main()
