import os
import sys
import gc

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM  # noqa: E402


def build_smollm2_model():
    gguf_path = os.path.join(
        PROJECT_ROOT, "data", "models", "smollm2-135m-instruct-fp16.gguf"
    )
    output_path = os.path.join(PROJECT_ROOT, "models", "smollm2-135m-anchored.gaje")

    print(f"🧬 Cargando y genomizando SmolLM2-135M desde GGUF FP16: {gguf_path}")
    llm = GenomicLLM(gguf_path)
    gc.collect()

    print(f"\n[*] Guardando organismo genómico SmolLM2-135M en: {output_path}...")
    llm.save(output_path)
    print(f"✅ Organismo Genómico SmolLM2-135M guardado exitosamente en: {output_path}")


if __name__ == "__main__":
    build_smollm2_model()
