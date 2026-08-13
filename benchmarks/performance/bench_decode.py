"""Benchmark forense de la regresión de throughput (0.2 vs ~3 tok/s).

Separa en fases medibles para responder a la pregunta:
  "¿Que convirtio ~3 tok/s en 0.2 tok/s?"

Fases:
  A. Interrogatorio del entorno: memoria fisica / swap (confirma thrashing)
  B. Cold-start (mmap)
  C. Prefill: cada token del prompt, timing individual
  D. Decode: cada token autoregresivo (greedy), timing individual
  E. Analisis de tendencia: el decode degrada con el largo de contexto? (KV cache)
  F. Sampleo estocastico vs greedy (ruido del sampler)

Uso:
  python benchmarks/performance/bench_decode.py \
      --model models/production/qwen2_0_5b_4bit.gaje[.flat] \
      --tokenizer temp_tokenizer/tokenizer.json \
      --prompt "El planeta mas grande del Sistema Solar es" \
      --tokens 96

Requisitos:
  - Motor compilado (maturin develop --release --features python)
  - Linux nativo o Windows nativo (NO WSL2 sobre /mnt: mmap 9p es lentisimo
    y distorsiona las mediciones). Ver notas en gaje_flat_benchmark.py.
"""
import argparse
import os
import platform
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
PYTHON_SRC = os.path.join(PROJECT_ROOT, "python")
if PYTHON_SRC not in sys.path:
    sys.path.insert(0, PYTHON_SRC)

from gaje.core import _impl as core  # noqa: E402


# ---------------------------------------------------------------------------
# Entorno: memoria / swap
# ---------------------------------------------------------------------------
def probe_environment():
    """Reporta RAM fisica, memoria en uso y swap. Usa psutil si esta disponible."""
    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_total_mb"] = vm.total / (1024 * 1024)
        info["ram_used_mb"] = vm.used / (1024 * 1024)
        info["ram_avail_mb"] = vm.available / (1024 * 1024)
        if psutil.swap_memory() is not None:
            sw = psutil.swap_memory()
            info["swap_total_mb"] = sw.total / (1024 * 1024)
            info["swap_used_mb"] = sw.used / (1024 * 1024)
            info["swap_pct"] = sw.percent
    except Exception as exc:  # pragma: no cover - psutil opcional
        info["psutil_error"] = str(exc)
    return info


def process_rss_mb():
    """RSS del proceso actual (Windows/Linux)."""
    import os as _os

    if _os.name == "nt":
        try:
            import psutil

            return psutil.Process(_os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:
            return None
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        return None
    return None


def measure(prefix, n_items, seconds):
    rate = (n_items / seconds) if seconds > 0 else 0.0
    per = (seconds * 1000.0 / n_items) if n_items > 0 else 0.0
    print(f"  [{prefix:<28}] {n_items:>4} it | {seconds:>9.3f} s | {per:>8.1f} ms/it | {rate:>7.2f} it/s")
    return rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="El planeta mas grande del Sistema Solar es")
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument(
        "--greedy",
        action="store_true",
        help="Usa greedy (argmax) en vez del sampler predeterminado.",
    )
    args = ap.parse_args()

    from tokenizers import Tokenizer

    env = probe_environment()
    print("=" * 72)
    print("BENCH DECODE FORENSE — GAJE")
    print(f"  plataforma : {env.get('platform')}")
    print(f"  python     : {env.get('python')}")
    if "ram_total_mb" in env:
        print(
            "  RAM        : {:.1f} MB total / {:.1f} MB en uso / {:.1f} MB disponible".format(
                env["ram_total_mb"], env["ram_used_mb"], env["ram_avail_mb"]
            )
        )
    if "swap_used_mb" in env:
        print(
            "  SWAP       : {:.1f} MB usado / {:.1f} MB total ({:.1f}%)".format(
                env["swap_used_mb"], env["swap_total_mb"], env["swap_pct"]
            )
        )
        if env["swap_used_mb"] > (0.5 * env["swap_total_mb"]):
            print(
                "  !! SWAP ALTA (>50%): el thrashing degrada el decode 10-20x. "
                "Confirma si otro proceso lo consume."
            )
    print("=" * 72)

    tok = Tokenizer.from_file(args.tokenizer)
    ids = tok.encode(args.prompt, add_special_tokens=False).ids
    print(f"model  : {os.path.basename(args.model)}")
    print(f"prompt : {args.prompt!r}  ({len(ids)} tokens)")
    print(f"decode : {args.tokens} tokens, modo={'greedy' if args.greedy else 'sampler'}")
    print("=" * 72)

    # B. Cold-start
    t0 = time.perf_counter()
    llm = core.load_genomic_auto(args.model)
    load_s = time.perf_counter() - t0
    rss = process_rss_mb()
    print(f"[{'cold-start mmap':<28}] {load_s * 1000:>9.1f} ms  RSS={rss:.1f} MB" if rss else
          f"[{'cold-start mmap':<28}] {load_s * 1000:>9.1f} ms")
    if hasattr(llm, "set_k_wta_ratio"):
        llm.set_k_wta_ratio(0.0)

    # C. Prefill token a token
    llm.clear_cache_py()
    prefill_times = []
    for tid in ids:
        t0 = time.perf_counter()
        llm.forward(tid, False)
        prefill_times.append((time.perf_counter() - t0) * 1000.0)
    p_arr = np.array(prefill_times)
    print("\n-- PREFILL (prompt) --")
    print(f"  total : {p_arr.sum():.1f} ms | media {p_arr.mean():.2f} ms/tok | "
          f"p50 {np.median(p_arr):.2f} | p95 {np.percentile(p_arr,95):.2f} | "
          f"max {p_arr.max():.2f} ms")
    print(f"  TTFT  (prefill hasta 1er token): {p_arr.sum():.1f} ms")

    # D + E. Decode autoregresivo token a token (sin clear_cache: KV acumula)
    logits = llm.forward(ids[-1], False)
    decode_times = []
    tokens_out = []
    for _ in range(args.tokens):
        if args.greedy:
            nid = int(np.argmax(logits))
        else:
            probs = np.array(logits, dtype=np.float64)
            m = probs - probs.max()
            exp = np.exp(m)
            probs = exp / exp.sum()
            nid = int(np.random.choice(len(probs), p=probs))
        t0 = time.perf_counter()
        logits = llm.forward(nid, False)
        decode_times.append((time.perf_counter() - t0) * 1000.0)
        tokens_out.append(nid)

    d_arr = np.array(decode_times)
    n_decode = len(d_arr)
    total_d = d_arr.sum()
    print("\n-- DECODE autoregresivo --")
    print(f"  tokens : {n_decode} | total {total_d:.1f} ms | "
          f"media {d_arr.mean():.2f} ms/tok ({1000.0/d_arr.mean():.2f} tok/s)")
    print(f"  p50 {np.median(d_arr):.2f} | p95 {np.percentile(d_arr,95):.2f} | "
          f"p99 {np.percentile(d_arr,99):.2f} | max {d_arr.max():.2f} ms")

    # E. Tendencia: comparar media de la mitad inicial vs final
    if n_decode >= 8:
        half = n_decode // 2
        first = d_arr[:half].mean()
        second = d_arr[half:].mean()
        delta = second / first if first > 0 else 0.0
        print(f"\n-- TENDENCIA KV CACHE (decode) --")
        print(f"  media 1a mitad : {first:.2f} ms/tok")
        print(f"  media 2a mitad : {second:.2f} ms/tok  (x{delta:.2f})")
        if delta > 1.3:
            print("  >> El decode DEGRADA con el largo de contexto -> patrón O(n^2) / mem-bandwidth.")
        else:
            print("  >> Decode estable frente al contexto -> descarta recomputación O(n^2).")

    # F. Desglose por bloques si el motor lo expone (GAJE_PROFILE_VERBOSE)
    rss_end = process_rss_mb()
    print(f"\nRSS proceso : {rss:.1f} MB (post-carga) -> {rss_end:.1f} MB (post-decode)")
    print("=" * 72)
    print("SUGERENCIA: reproduce el 0.2 tok/s con este benchmark. Si aqui da \n"
          "~3+ tok/s, el cuello de botella era ambiental (swap), no el motor.")


if __name__ == "__main__":
    main()