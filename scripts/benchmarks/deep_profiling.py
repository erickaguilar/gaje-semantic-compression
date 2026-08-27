import os
import sys
import time
import numpy as np
import psutil
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.core import _impl as dna_semantic_compression  # noqa: E402


def run_deep_profiling():
    print("=================================================================")
    print("🔍 GAJE-Flow v0.9.7: DEEP PROFILING & BOTTLENECK ANALYSIS")
    print("=================================================================")

    flat_path = os.path.join(
        PROJECT_ROOT, "models", "production", "qwen2_0_5b_4bit.gaje.flat"
    )
    tokenizer_id = "Qwen/Qwen2-0.5B-Instruct"

    print(f"[*] Cargando Tokenizador ({tokenizer_id})...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    print(f"[*] Cargando Modelo GAJE Zero-Copy Flat Mmap desde:\n    {flat_path}")
    t0_load = time.perf_counter()
    gaje_llm = dna_semantic_compression.load_genomic_auto(flat_path)
    gaje_llm.set_k_wta_ratio(0.0)
    load_time_ms = (time.perf_counter() - t0_load) * 1000.0
    print(f"✅ Cargado en {load_time_ms:.2f} ms")

    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    print(f"  - RAM Activa Inicial: {ram_mb:.2f} MB")

    prompts = [
        "A cuál país pertenece la capital París?",
        "Explica qué es la fotosíntesis en las plantas en una oración simple.",
        "Write a Python function to calculate the Fibonacci sequence using iteration.",
    ]

    for p_idx, prompt in enumerate(prompts, 1):
        print("\n==================================================")
        print(f"🧪 PROMPING EXPERIMENTO #{p_idx}: {prompt!r}")
        print("==================================================")

        # 1. TOKENIZACIÓN
        t0_tok = time.perf_counter()
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        tok_ms = (time.perf_counter() - t0_tok) * 1000.0
        print(
            f"1. Tokenización Python: {tok_ms:.3f} ms ({len(tokens)} tokens de entrada)"
        )

        # 2. PREFILL (Token por Token en Python vs Native)
        gaje_llm.clear_cache_py()

        # Prefill token por token manual
        t0_prefill = time.perf_counter()
        prefill_token_times = []
        for tid in tokens:
            t0_t = time.perf_counter()
            _ = gaje_llm.forward(tid, False)
            dt = (time.perf_counter() - t0_t) * 1000.0
            prefill_token_times.append(dt)
        total_prefill_ms = (time.perf_counter() - t0_prefill) * 1000.0
        avg_prefill_tok_ms = total_prefill_ms / len(tokens)

        print("\n2. DESGLOSE DE PREFILL (Prompt Processing):")
        print(
            f"  - Tiempo Total de Prefill: {total_prefill_ms:.2f} ms ({len(tokens)} tokens)"
        )
        print(f"  - Latencia por Token Prefill: {avg_prefill_tok_ms:.2f} ms/tok")
        print(
            f"  - Primer token de prefill:  {prefill_token_times[0]:.2f} ms (cold cache / allocation)"
        )
        if len(prefill_token_times) > 1:
            print(
                f"  - Tokens subsiguientes avg:  {np.mean(prefill_token_times[1:]):.2f} ms"
            )

        # 3. DECODE AUTORREGRESIVO (Comparativa: C++ Native Loop vs Python Step Loop)
        N_GEN = 30
        print(f"\n3. DESGLOSE DE DECODE AUTORREGRESIVO ({N_GEN} tokens):")

        # A) Bucle Step por Step en Python
        gaje_llm.clear_cache_py()
        # Repetimos prefill
        for tid in tokens:
            last_logits = gaje_llm.forward(tid, False)

        curr_token = int(np.argmax(last_logits))
        py_decode_times = []

        t0_py_loop = time.perf_counter()
        for _ in range(N_GEN):
            t0_step = time.perf_counter()
            logits = gaje_llm.forward(curr_token, False)
            curr_token = int(np.argmax(logits))
            py_decode_times.append((time.perf_counter() - t0_step) * 1000.0)
        total_py_decode_ms = (time.perf_counter() - t0_py_loop) * 1000.0
        avg_py_decode_ms = np.mean(py_decode_times)
        py_tok_sec = 1000.0 / avg_py_decode_ms

        print("  [A] Python Step Loop:")
        print(f"      - Total: {total_py_decode_ms:.2f} ms")
        print(
            f"      - Latencia Decode: {avg_py_decode_ms:.2f} ms/tok ({py_tok_sec:.2f} tok/s)"
        )
        print(
            f"      - Min: {np.min(py_decode_times):.2f} ms | Max: {np.max(py_decode_times):.2f} ms | Std: {np.std(py_decode_times):.2f} ms"
        )

        # B) Generación Nativa C++/Rust Pure Loop (generate_native_py)
        gaje_llm.clear_cache_py()
        t0_native = time.perf_counter()
        native_tokens = gaje_llm.generate_native_py(
            tokens, N_GEN, 0.3, 1.1, [2, 151643, 151645]
        )
        total_native_ms = (time.perf_counter() - t0_native) * 1000.0

        # Separar tiempo aproximado de prefill e inferencia pura
        gen_tokens_count = len(native_tokens)
        native_tok_sec = (
            (gen_tokens_count / (total_native_ms / 1000.0))
            if total_native_ms > 0
            else 0
        )
        native_decode_ms = (
            (total_native_ms / gen_tokens_count) if gen_tokens_count > 0 else 0
        )

        print("  [B] Native Rust Loop (`generate_native_py`):")
        print(
            f"      - Total E2E: {total_native_ms:.2f} ms ({gen_tokens_count} tokens)"
        )
        print(
            f"      - Latencia E2E Promedio: {native_decode_ms:.2f} ms/tok ({native_tok_sec:.2f} tok/s)"
        )

        # C) Overhead de Python
        overhead_ms = total_py_decode_ms - (avg_py_decode_ms * N_GEN)
        print("  [C] Overhead de FFI / Python GIL per step:")
        print(f"      - Overhead Total estimado: {overhead_ms:.2f} ms")
        py_ffi_overhead_per_tok = avg_py_decode_ms - (
            (total_native_ms - total_prefill_ms) / max(gen_tokens_count, 1)
        )
        print(
            f"      - FFI + Argmax Overhead estimado: {py_ffi_overhead_per_tok:.2f} ms por token"
        )

    print("\n=================================================================")
    print("📌 CONCLUSIONES DEL DEEP PROFILING")
    print("=================================================================")


if __name__ == "__main__":
    run_deep_profiling()
