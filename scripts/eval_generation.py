#!/usr/bin/env python3
"""Harness de EVALUACIÓN GENERATIVA FIJA — compara modelos con métricas objetivas.

Motivo: en este proyecto se juzgó la calidad con ejemplos sueltos (cherry-picking),
lo que llevó a conclusiones equivocadas (p.ej. "distill 1520 => coherente"). Este
harness aplica los MISMOS prompts, la MISMA temperatura y métricas objetivas a todos
los modelos para decidir con datos si el fine-tune del cuerpo mejora o degrada.

Métricas por modelo (media sobre prompts):
  - distinct-1 / distinct-2 : diversidad de tokens/bigramas (bajo = degenerado).
  - repeticion               : fracción de bigramas repetidos dentro de la salida.
  - len_tokens               : longitud media de la salida.
  - degeneradas              : % de respuestas degeneradas (distinct-1 < umbral o
                               repetición muy alta).

Uso:
    python scripts/eval_generation.py [--models a,lista,de,paths] [--temp 0.4] [--max_new 40]
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM  # noqa: E402

DEFAULT_MODELS = [
    "models/production/smollm2_4bit.gaje.flat",
    "models/production/smollm2_4bit_quality.gaje.flat",
    "models/production/smollm2_4bit_clean.gaje.flat",
    "models/production/smollm2_4bit_quality_big.gaje.flat",
    "models/production/smollm2_4bit_trained.gaje.flat",
]

PROMPTS = [
    "¿Cuál es la capital de Francia?",
    "¿Cuál es la capital de España?",
    "La capital de España es",
    "¿Qué es la fotosíntesis?",
    "¿Qué es el ADN?",
    "El planeta más grande del Sistema Solar es",
    "Si x = 12 y el padre tiene 36 años, ¿cuánto tiene el hijo?",
    "Traduce al inglés: 'hola, ¿cómo estás?'",
]


def distinct_ngrams(tokens, n):
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def repetition_rate(tokens):
    """Fracción de bigramas que se repiten al menos una vez dentro de la salida."""
    bigrams = [tuple(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
    if not bigrams:
        return 0.0
    seen = set()
    repeats = 0
    for bg in bigrams:
        if bg in seen:
            repeats += 1
        seen.add(bg)
    return repeats / len(bigrams)


def is_degenerate(tokens):
    if len(tokens) < 3:
        return True
    d1 = distinct_ngrams(tokens, 1)
    rep = repetition_rate(tokens)
    return d1 < 0.25 or rep > 0.7


def generate(model, prompt, temperature, max_new):
    try:
        return "".join(model.generate(prompt, max_new_tokens=max_new, temperature=temperature))
    except Exception as e:  # noqa: BLE001
        return f"[error: {e}]"


def tokenize(model, text):
    ids = model.tokenizer.encode(text, add_special_tokens=False)
    return list(getattr(ids, "ids", ids))


def eval_model(path, temperature, max_new):
    m = GenomicLLM.load_genomic(path)
    per_prompt = []
    for p in PROMPTS:
        out = generate(m, p, temperature, max_new)
        toks = tokenize(m, out)
        d1 = distinct_ngrams(toks, 1)
        d2 = distinct_ngrams(toks, 2)
        rep = repetition_rate(toks)
        per_prompt.append({
            "prompt": p,
            "out": out,
            "tokens": len(toks),
            "d1": d1,
            "d2": d2,
            "rep": rep,
            "deg": is_degenerate(toks),
        })
    n = len(per_prompt)
    agg = {
        "model": path,
        "d1": sum(x["d1"] for x in per_prompt) / n,
        "d2": sum(x["d2"] for x in per_prompt) / n,
        "rep": sum(x["rep"] for x in per_prompt) / n,
        "len": sum(x["tokens"] for x in per_prompt) / n,
        "deg": sum(x["deg"] for x in per_prompt) / n,
    }
    return agg, per_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--temp", type=float, default=0.4)
    ap.add_argument("--max_new", type=int, default=40)
    args = ap.parse_args()

    models = [os.path.join(PROJECT_ROOT, p) for p in args.models.split(",") if p]
    results = []
    for path in models:
        agg, per = eval_model(path, args.temp, args.max_new)
        results.append(agg)
        print(f"\n=== {path} (temp={args.temp}) ===")
        for x in per:
            print(f"  {x['prompt'][:45]!r} -> len={x['tokens']:3d} "
                  f"d1={x['d1']:.2f} d2={x['d2']:.2f} rep={x['rep']:.2f} "
                  f"{'DEG' if x['deg'] else 'ok '}")
            print(f"      {x['out'][:90]!r}")

    print("\n\n=== RESUMEN AGREGADO (d1/d2 más alto y rep/deg más bajo = mejor) ===")
    print(f"{'modelo':<55} {'d1':>5} {'d2':>5} {'rep':>5} {'len':>5} {'deg%':>5}")
    for r in results:
        print(f"{os.path.basename(r['model']):<55} {r['d1']:.3f} {r['d2']:.3f} "
              f"{r['rep']:.3f} {r['len']:5.1f} {r['deg']:5.0%}")


if __name__ == "__main__":
    main()