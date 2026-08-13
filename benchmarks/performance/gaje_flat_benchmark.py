"""Benchmark de rendimiento nativo para modelos GAJE `.gaje.flat`.

Mide:
  1. Cold-start (mmap) - tiempo de carga del archivo binario plano
  2. Prefill (procesamiento del prompt / KV-cache)
  3. Decode (generación autoregresiva greedy)

Uso:
    python benchmarks/performance/gaje_flat_benchmark.py \
        --model models/production/qwen2_5_1_5b_q4_0.gaje.flat \
        --tokenizer temp_tokenizer/tokenizer.json \
        --prompt "El planeta más grande del Sistema Solar es" \
        --tokens 64

ADVERTENCIA DE ENTORNO WSL2:
  * Si el modelo reside en /mnt/<letra> (filesystem Windows vía 9p), el mmap
    perezoso dispara lecturas muy lentas. Copia el .flat a ext4 nativo primero.
  * La RAM limitada de la VM hace que el decode pueda convoy con SWAP/thrash.
Ejecútalo en hardware/Linux nativo para números representativos del README.
"""
import argparse
import os
import sys
import time

import numpy as np

MODELPATH = os.path.abspath("python")
if MODELPATH not in sys.path:
    sys.path.insert(0, MODELPATH)

from gaje.core import _impl as core  # noqa: E402

dna_semantic_compression = core


def rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:") or line.startswith("VmSize:"):
                parts = line.split()
                yield parts[0].rstrip(":"), int(parts[1])


def measure(prefix, n, seconds):
    rate = n / seconds if seconds > 0 else 0.0
    print(f"[{prefix:<12}] {n} tok in {seconds:.3f} s  ({rate:.2f} tok/s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="El planeta más grande del Sistema Solar es")
    ap.add_argument("--tokens", type=int, default=64)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    ids = tok.encode(args.prompt, add_special_tokens=False).ids

    print("=" * 64)
    print(f"GAJE FLAT BENCHMARK")
    print(f"  model  : {args.model}")
    print(f"  prompt : {args.prompt!r} ({len(ids)} tokens)")
    print("=" * 64)

    # 1. Cold start (mmap perezoso)
    t0 = time.perf_counter()
    llm = dna_semantic_compression.load_genomic_auto(args.model)
    load_s = time.perf_counter() - t0
    rss = dict(rss_kb())
    print(f"[{'cold-start mmap':<12}] {load_s * 1000:.1f} ms  "
          f"(mRSS={rss.get('VmRSS:', 0):,} kB, VmSize={rss.get('VmSize:', 0):,} kB)")

    # 2. Prefill
    llm.clear_cache_py()
    t0 = time.perf_counter()
    for tid in ids:
        llm.forward(tid, False)
    measure("prefill", len(ids), time.perf_counter() - t0)

    # 3. Decode greed
    llm.clear_cache_py()
    for tid in ids:
        llm.forward(tid, False)
    logits = llm.forward(ids[-1], False)

    t0 = time.perf_counter()
    for _ in range(args.tokens):
        nid = int(np.argmax(logits))
        logits = llm.forward(nid, False)
    measure("decode", args.tokens, time.perf_counter() - t0)

    print("=" * 64)


if __name__ == "__main__":
    main()