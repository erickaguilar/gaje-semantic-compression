import os
import sys
import gc

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM


def build_anchored_model():
    gguf_path = os.path.join(PROJECT_ROOT, "data", "models", "qwen2-0_5b-instruct-fp16.gguf")
    output_path = os.path.join(PROJECT_ROOT, "models", "qwen2-0_5b-anchored.gaje")

    if os.path.exists(gguf_path):
        print(f"🧬 Cargando y genomizando desde GGUF FP16 original (Preservando token_embd y lm_head al 100%): {gguf_path}")
        llm = GenomicLLM(gguf_path)
    else:
        coherent_path = os.path.join(PROJECT_ROOT, "models", "qwen2-0_5b-coherent.gaje")
        print(f"🧬 Cargando modelo base desde BD: {coherent_path}")
        llm = GenomicLLM.load_genomic(coherent_path)

    gc.collect()

    print(f"\n[*] Guardando organismo genómico anclado con Embeddings y LM Head protegidos en: {output_path}...")
    llm.save(output_path)
    print(f"✅ Organismo Genómico Anclado guardado exitosamente en: {output_path}")


if __name__ == "__main__":
    build_anchored_model()
