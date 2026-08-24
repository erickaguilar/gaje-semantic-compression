#!/usr/bin/env python3
"""Gate de Nacimiento: valida que un organismo .flat genere con coherencia minima.

Uso:
    python scripts/validate_flat_birth.py models/production/gaje_ultra_7b.flat

Ejecuta prompts factuales con decodificacion GREEDY (determinista, sin sampling).
Si el organismo esta corrupto, produce saladillo y el gate falla -> exit code 1.
Disenado para encadenarse tras export_gaje_flat.py / transmute_qwen_models.py:
ningun modelo debe llegar a produccion sin pasar este gate (Mandato de Verdad
Empirica).

Opciones:
    --quick     Solo 2 prompts (para CI rapido)
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath("python"))

PROBES = [
    # (prompt, subcadenas aceptables en la respuesta, case-insensitive)
    ("The capital of Germany is", ["berlin"]),
    ("The capital of France is", ["paris"]),
    ("The capital of Japan is", ["tokyo", "kyoto"]),
    ("Water boils at", ["100", "212"]),
]


def run_probe(llm, prompt, accept, max_tokens=12):
    """Genera greedy y verifica que alguna subcadena esperada aparezca."""
    tok = llm.tokenizer
    enc = tok.encode(prompt, add_special_tokens=False)
    if hasattr(enc, "ids"):
        enc = enc.ids

    llm.rust_llm.clear_cache_py()
    for tid in enc[:-1]:
        llm.rust_llm.forward(tid, False)

    logits = llm.rust_llm.forward(enc[-1], False)
    out_ids = []
    for _ in range(max_tokens):
        nid = int(np.argmax(logits))
        if nid in getattr(tok, "eos_ids", []) or nid == tok.token_to_id("<|im_end|>"):
            break
        out_ids.append(nid)
        logits = llm.rust_llm.forward(nid, False)

    text = tok.decode(out_ids).lower()
    hit = any(a in text for a in accept)
    return hit, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Modelo no encontrado: {args.model}")
        return 2

    from gaje.nn.stabilized import GenomicLLM

    print("=" * 64)
    print("🚪 GATE DE NACIMIENTO:", os.path.basename(args.model))
    print("=" * 64)

    t0 = time.time()
    llm = GenomicLLM.load_genomic(args.model)
    print(f"[carga] {time.time() - t0:.1f}s")

    probes = PROBES[:2] if args.quick else PROBES
    passed = 0
    for prompt, accept in probes:
        hit, text = run_probe(llm, prompt, accept)
        status = "✅" if hit else "❌"
        preview = text[:48].replace("\n", " ")
        print(f"{status} {prompt!r} -> {preview!r}")
        passed += hit

    score = passed / len(probes) * 100
    print("-" * 64)
    print(f"GATE: {passed}/{len(probes)} ({score:.0f}%)")
    ok = passed == len(probes)
    print(
        "VEREDICTO:",
        "🏆 ORGANISMO VIABLE"
        if ok
        else "☠️  NACIMIENTO RECHAZADO (pesos corruptos o degenerados)",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
