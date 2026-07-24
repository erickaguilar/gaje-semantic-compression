import os
import sys

# Asegurar uso de código local
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
)
from gaje.nn.stabilized import GenomicLLM


def run_coherence_demo():
    print("🧬 GAJE COHERENCE DEMO: Evaluación de Sentido Común")
    print("=" * 60)

    model_path = "models/gguf/qwen2-0_5b-q8_0.gguf"
    if not os.path.exists(model_path):
        print(
            f"⚠️ Modelo no encontrado en {model_path}. Por favor descarga uno para probar."
        )
        return

    print("[*] Cargando Micro-Genoma (v1.x)...")
    llm = GenomicLLM(model_path, num_blocks=4)

    test_prompts = [
        "What is the capital of Spain?",
        "Translate 'The sun is hot' to Spanish:",
        "Complete the sequence: 1, 2, 3, 4,",
    ]

    for prompt in test_prompts:
        print(f"\n💬 Prompt: {prompt}")
        print("🤖 Respuesta: ", end="", flush=True)
        for token in llm.generate(prompt, max_new_tokens=20, temperature=0.5):
            print(token, end="", flush=True)
        print("\n" + "-" * 40)


if __name__ == "__main__":
    run_coherence_demo()
