#!/usr/bin/env python3
"""FASE 4a — Gate H1: latencia de transicion nodo→nodo del Agentic Graph.

Compara tres formas de orquestar una cadena de N nodos con presupuesto
identico de trabajo por nodo:

  A) Nativo Rust puro     : StateGraph::run completo dentro de Rust.
                            Mide SOLO la maquinaria del grafo (dispatch,
                            clone de estado, trait dispatch).
  B) PyO3 boundary        : cada nodo cruza la frontera FFI sin serializar
                            (payload por referencia &str).
  C) Python + JSON handoff: baseline estilo LangGraph — cada paso serializa
                            el estado a JSON, lo cruza y deserializa.

Gate H1 (PHASE_4_AGENTIC_GRAPH_EXECUTION_PLAN.md):
  - Transicion nativa < 10 us p50
  - Nativo >= 100x mas rapido que el handoff serializado (C)
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath("python"))
from gaje.core._impl import boundary_step_py, graph_bench_native_py  # noqa: E402

CHAIN_LEN = 8
ITERATIONS = 20_000


def bench_boundary(iters):
    """Nodo via PyO3 sin serializacion (payload &str directo)."""
    t0 = time.perf_counter()
    hops = 0
    for i in range(iters):
        _, hops = boundary_step_py(f"q{i}", hops)
    return (time.perf_counter() - t0) * 1e9 / iters


def validate_state(d):
    """Validacion de esquema estilo Pydantic/LangGraph (verificacion de tipos
    completa de todos los campos de AgentState por transicion)."""
    if not isinstance(d, dict):
        raise TypeError("state must be dict")
    if not isinstance(d.get("user_query"), str):
        raise TypeError("user_query must be str")
    if not isinstance(d.get("hops"), int):
        raise TypeError("hops must be int")
    if "context" in d and not isinstance(d["context"], list):
        raise TypeError("context must be list")
    if "tool_outputs" in d and not isinstance(d["tool_outputs"], list):
        raise TypeError("tool_outputs must be list")
    return d


def bench_json_handoff(iters, chain_len):
    """Orquestacion Python estilo framework (LangGraph / CrewAI):
    serializacion JSON + validacion de esquema Pydantic + handoff FFI."""
    state = {
        "user_query": "benchmark-query-agentic-swarm",
        "intent": "SwarmIntent::DeepReasoning",
        "context": ["[MemoryRAG] Vector 768d match", "[DocRAG] Protocolo GAJE"],
        "tool_outputs": [("math_tool", "1024"), ("db_query", "record_ok")],
        "response": None,
        "hops": 0,
    }
    t0 = time.perf_counter()
    for _ in range(iters):
        for _ in range(chain_len):
            blob = json.dumps(state)
            d = validate_state(json.loads(blob))
            uq, hops = boundary_step_py(d["user_query"], d["hops"])
            state["user_query"] = uq
            state["hops"] = hops
    total_steps = iters * chain_len
    return (time.perf_counter() - t0) * 1e9 / total_steps


def main():
    print("=" * 66)
    print("FASE 4a — GATE H1: transicion nodo->nodo")
    print(f"  cadena={CHAIN_LEN} nodos | iteraciones={ITERATIONS:,}")
    print("=" * 66)

    # A. Nativo Rust
    res = graph_bench_native_py(CHAIN_LEN, ITERATIONS)
    native_ns = res.ns_per_transition
    print(f"[A] Nativo Rust      : {native_ns:10.1f} ns/transicion  "
          f"({res.total_ms:.1f} ms total, {res.transitions:,} transiciones)")

    # B. PyO3 boundary sin serializar
    pyo_ns = bench_boundary(ITERATIONS * CHAIN_LEN)
    print(f"[B] PyO3 boundary    : {pyo_ns:10.1f} ns/transicion")

    # C. Python + JSON handoff (baseline LangGraph-style)
    iters_c = max(200, ITERATIONS // 50)  # es lento por diseño; menos muestras
    json_ns = bench_json_handoff(iters_c, CHAIN_LEN)
    print(f"[C] Python+JSON      : {json_ns:10.1f} ns/transicion")

    ratio_native_vs_json = json_ns / native_ns if native_ns > 0 else float("inf")
    ratio_native_vs_pyo = pyo_ns / native_ns if native_ns > 0 else float("inf")

    print("-" * 66)
    gate_latency = native_ns < 10_000
    gate_speedup = ratio_native_vs_json >= 100
    print(f"GATE latencia   : {native_ns:.1f} ns < 10,000 ns -> "
          f"{'✅ PASS' if gate_latency else '❌ FAIL'}")
    print(f"GATE speedup    : {ratio_native_vs_json:.0f}x vs JSON handoff "
          f"(>= 100x) -> {'✅ PASS' if gate_speedup else '❌ FAIL'}")
    print(f"  (vs PyO3 boundary sin serializar: {ratio_native_vs_pyo:.0f}x)")

    results = {
        "chain_len": CHAIN_LEN,
        "iterations": ITERATIONS,
        "native_ns_per_transition": round(native_ns, 1),
        "pyo3_boundary_ns": round(pyo_ns, 1),
        "python_json_handoff_ns": round(json_ns, 1),
        "speedup_vs_json_handoff": round(ratio_native_vs_json, 1),
        "gate_h1_pass": bool(gate_latency and gate_speedup),
    }
    out = "benchmarks/logs/graph_4a_gate_results.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados: {out}")


if __name__ == "__main__":
    main()
