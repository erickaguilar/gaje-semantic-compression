"""Benchmark oficial del motor GAJE: compresión, memoria y velocidad por modelo.

Reproduce la ruta de producción: generate_native_py(tokens, 512, 0.2, 1.1, eos).
Uso:
    python scripts/benchmarks/engine_benchmark.py
    python scripts/benchmarks/engine_benchmark.py --gen_tokens 64
"""

import os
import sys
import time
import json
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM


# Parámetros nominales públicos (miles de millones de parámetros)
# Qwen2-0.5B=0.494B, Qwen2.5-1.5B=1.54B, Qwen2.5-3B=3.09B, SmolLM2-135M=0.135B
NOMINAL = {
    "qwen2_0_5b": 0.494,
    "qwen2_5_1_5b": 1.54,
    "qwen2_5_3b": 3.09,
    "smollm2": 0.135,
}

PROD_DIR = os.path.join(ROOT, "models", "production")

PROMPT = (
    "Explica en español, con detalle y en varios pasos, qué es la inteligencia "
    "artificial y cómo funciona un modelo de lenguaje basado en transformers."
)


def rss_kb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return 0


def vms_kb():
    for line in open("/proc/self/status"):
        if line.startswith("VmSize:"):
            return int(line.split()[1])
    return 0


def gb(n):
    return n / 1024.0 / 1024.0


def bytes_per_param(file_bytes, params_b):
    return file_bytes / (params_b * 1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_tokens", type=int, default=64)
    args = ap.parse_args()

    files = sorted(os.listdir(PROD_DIR))
    files = [f for f in files if f.endswith(".gaje.flat")]

    results = []
    for fn in files:
        path = os.path.join(PROD_DIR, fn)
        key = next((k for k in NOMINAL if k in fn), None)
        params_b = NOMINAL.get(key)
        file_bytes = os.path.getsize(path)

        entry = {
            "file": fn,
            "size_mb": round(file_bytes / 1024 / 1024, 1),
            "params_b": params_b,
            "bytes_per_param": (
                round(bytes_per_param(file_bytes, params_b), 3) if params_b else None
            ),
            "vms_load_gb": None,
            "rss_before_gb": round(gb(rss_kb()), 2),
            "rss_after_load_gb": None,
            "rss_after_gen_gb": None,
            "load_ms": None,
            "ttft_ms": None,
            "gen_ms": None,
            "gen_tokens": None,
            "tok_per_s": None,
        }

        print(f"\n=== {fn} ({entry['size_mb']} MB) ===")
        t0 = time.time()
        llm = GenomicLLM.load_genomic(path)
        entry["load_ms"] = round((time.time() - t0) * 1000, 1)
        entry["vms_load_gb"] = round(gb(vms_kb()), 2)
        entry["rss_after_load_gb"] = round(gb(rss_kb()), 2)
        print(
            f"  load: {entry['load_ms']} ms | VMS {entry['vms_load_gb']} GB | RSS {entry['rss_after_load_gb']} GB"
        )

        eos_ids = [2, 151643, 151645]
        if (
            hasattr(llm.tokenizer, "eos_token_id")
            and llm.tokenizer.eos_token_id is not None
        ):
            eos_ids.append(llm.tokenizer.eos_token_id)

        if not (
            hasattr(llm, "rust_llm")
            and llm.rust_llm is not None
            and hasattr(llm.rust_llm, "generate_native_py")
        ):
            print("  ⚠️ sin ruta nativa generate_native_py, se omite velocidad")
            results.append(entry)
            continue

        enc = llm.tokenizer.encode(PROMPT, add_special_tokens=False)
        tokens = enc.ids if hasattr(enc, "ids") else enc

        # TTFT (1 token) — incluye prefill + primer decode, caché fría
        t0 = time.time()
        llm.rust_llm.generate_native_py(tokens, 1, 0.2, 1.1, eos_ids)
        entry["ttft_ms"] = round((time.time() - t0) * 1000, 1)

        # Throughput de generación (ruta de producción)
        t0 = time.time()
        gen_ids = llm.rust_llm.generate_native_py(
            tokens, args.gen_tokens, 0.2, 1.1, eos_ids
        )
        entry["gen_ms"] = round((time.time() - t0) * 1000, 1)
        entry["gen_tokens"] = len(gen_ids)
        if entry["gen_ms"] > 0:
            entry["tok_per_s"] = round(len(gen_ids) / (entry["gen_ms"] / 1000.0), 2)
        entry["rss_after_gen_gb"] = round(gb(rss_kb()), 2)
        print(
            f"  TTFT {entry['ttft_ms']} ms | {entry['gen_tokens']} tok en "
            f"{entry['gen_ms']} ms → {entry['tok_per_s']} tok/s | RSS {entry['rss_after_gen_gb']} GB"
        )

        results.append(entry)

    with open(
        os.path.join(os.path.dirname(__file__), "engine_benchmark_results.json"), "w"
    ) as f:
        json.dump(results, f, indent=2)
    print("\n=== RESUMEN ===")
    for e in results:
        print(
            f"{e['file']:<40} {e['size_mb']:>8} MB  {e['bytes_per_param']} b/p  "
            f"{e['tok_per_s']} tok/s"
        )


if __name__ == "__main__":
    main()
