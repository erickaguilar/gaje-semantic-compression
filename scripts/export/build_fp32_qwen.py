import os
import sys
import gc
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM


def build_and_test_fp32():
    gguf_path = os.path.join(
        PROJECT_ROOT, "data", "models", "qwen2-0_5b-instruct-fp16.gguf"
    )
    output_path = os.path.join(PROJECT_ROOT, "models", "qwen2-0_5b-fp32.gaje")

    print("=" * 60)
    print("🧬 CONSTRUYENDO Y EVALUANDO FP32 PURO EN MOTOR RUST CON BIAS GUARDADO")
    print("=" * 60)

    print(f"[*] Cargando desde GGUF FP16: {gguf_path}")
    llm = GenomicLLM(gguf_path)
    gc.collect()

    print(
        f"\n[*] Guardando organismo FP32 completo con vectores bias en BD: {output_path}..."
    )
    llm.save(output_path)
    print("✅ BD FP32 guardada exitosamente.")

    print("\n[*] Re-cargando organismo FP32 desde BD...")
    llm_reloaded = GenomicLLM.load_genomic(output_path)

    prompts = [
        "The capital of France is",
        "Responde únicamente con una palabra: capital de Francia",
    ]

    print("\n" + "=" * 60)
    print("🎯 EVALUACIÓN DE RESPUESTAS DESDE BD FP32 (CON BIAS REAL)")
    print("=" * 60)

    for p in prompts:
        print(f"\n👤 Prompt: '{p}'")
        print("🤖 GAJE (BD FP32): ", end="", flush=True)
        t0 = time.time()
        output_text = ""
        for token in llm_reloaded.generate(p, max_new_tokens=10, temperature=0.1):
            print(token, end="", flush=True)
            output_text += token
        elapsed = time.time() - t0
        print(f"\n   ⏱️ Tiempo: {elapsed:.2f}s")


if __name__ == "__main__":
    build_and_test_fp32()
