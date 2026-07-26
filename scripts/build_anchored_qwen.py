import os
import sys
import gc

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM


def build_anchored_model():
    coherent_path = os.path.join(PROJECT_ROOT, "models", "qwen2-0_5b-coherent.gaje")
    output_path = os.path.join(PROJECT_ROOT, "models", "qwen2-0_5b-anchored.gaje")

    if not os.path.exists(coherent_path):
        print(f"Error: {coherent_path} not found.")
        return

    print(f"🧬 Cargando modelo base Qwen2-0.5B Coherente desde BD: {coherent_path}")
    llm = GenomicLLM.load_genomic(coherent_path)

    gc.collect()

    print(f"\n[*] Protegiendo lm_head al 100% de precisión y guardando en: {output_path}...")
    llm.save(output_path)
    print(f"✅ Organismo Genómico Anclado guardado exitosamente en: {output_path}")


if __name__ == "__main__":
    build_anchored_model()
