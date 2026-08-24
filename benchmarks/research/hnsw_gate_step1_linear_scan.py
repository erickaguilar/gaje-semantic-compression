#!/usr/bin/env python3
"""PASO 1 del micro-plan HNSW: Gate empirico del escaneo lineal en .gmem.

Mide la latencia REAL de retrieve_context (ruta RAG de produccion) a escala
creciente para determinar a que N se rompe el gate sub-milisegundo.

Veredicto:
  - Si toda escala util mantiene < 1 ms  -> el escaneo lineal basta, HNSW no
    se construye (patron Q2_0: frente congelado con datos).
  - Si se rompe -> procede el paso 2 (elegir crate vs grafo propio).
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath("python"))
from gaje.core._impl import IslandOrchestrator  # noqa: E402

DIM = 768
K_PER_NICHE = 3
SCALES = [10_000, 50_000, 100_000, 200_000, 400_000]
N_QUERIES = 15
NICHES = ["documental", "episodic", "conversational"]


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0


def build_entries(orch, n_total):
    """Reparte n_total entradas entre los 3 nichos con vectores sinteticos."""
    per_niche = n_total // len(NICHES)
    rng = np.random.default_rng(42)
    t0 = time.time()
    gid = 0
    for niche in NICHES:
        for _ in range(per_niche):
            v = rng.standard_normal(DIM).astype(np.float32)
            orch.add_memory_py(niche, gid, v.tolist(), f"doc{gid}")
            gid += 1
    print(
        f"    [{n_total:>7,} entradas] construccion: {time.time() - t0:.1f}s | "
        f"RSS: {rss_mb():,.0f} MB",
        flush=True,
    )


def measure(orch):
    """Latencia p50/p95/max sobre queries sinteticas (caché caliente)."""
    rng = np.random.default_rng(7)
    latencies = []
    for _ in range(N_QUERIES):
        q = rng.standard_normal(DIM).astype(np.float32)
        t0 = time.perf_counter()
        orch.retrieve_context_py(q.tolist(), K_PER_NICHE)
        latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    return p50, p95, latencies[-1]


def main():
    print("=" * 68)
    print("GATE EMPIRICO PASO 1: escaneo lineal .gmem vs escala")
    print(f"  dim={DIM} | k/nicho={K_PER_NICHE} | {N_QUERIES} queries por escala")
    print("=" * 68)

    results = []
    orch = IslandOrchestrator(DIM)
    prev_n = 0
    for n_total in SCALES:
        build_entries(orch, n_total - prev_n)
        prev_n = n_total

        # Warm-up (faults de pagina fuera de la medicion)
        q = np.random.default_rng(1).standard_normal(DIM).astype(np.float32)
        orch.retrieve_context_py(q.tolist(), K_PER_NICHE)

        p50, p95, mx = measure(orch)
        gate = "✅ PASS" if p50 < 1.0 else "❌ FAIL"
        results.append({"n": n_total, "p50": p50, "p95": p95, "max": mx})
        print(
            f"    N={n_total:>8,} | p50={p50:8.2f} ms | p95={p95:8.2f} ms | "
            f"max={mx:8.2f} ms | gate sub-ms: {gate}",
            flush=True,
        )

    # Extrapolacion lineal a 1M basada en el ultimo tramo medido
    last, prev = results[-1], results[-2]
    slope_ms_per_entry = (last["p50"] - prev["p50"]) / (last["n"] - prev["n"])
    est_1m = slope_ms_per_entry * 1_000_000
    print("-" * 68)
    print(f"Pendiente medida: {slope_ms_per_entry * 1000:.3f} us/entrada (lineal)")
    print(f"Extrapolacion O(N) a 1M entradas: ~{est_1m:,.0f} ms")

    failing = [r for r in results if r["p50"] >= 1.0]
    verdict = {
        "gate_pass": not failing,
        "first_failure_at": failing[0]["n"] if failing else None,
        "results": results,
        "est_1m_p50_ms": round(est_1m, 1),
        "rss_final_mb": round(rss_mb()),
    }
    out = "benchmarks/logs/hnsw_gate_step1_results.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(verdict, f, indent=2)
    print(
        f"\nVeredicto: {'✅ GATE PASA — lineal suficiente' if not failing else '❌ GATE FALLA desde N=' + format(failing[0]['n'], ',')}"
    )
    print(f"Resultados guardados en {out}")


if __name__ == "__main__":
    main()
