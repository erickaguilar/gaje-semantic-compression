import os
import sys

# Añadir el directorio python al path para importar los módulos locales
sys.path.append(os.path.abspath("python"))

from genomize_llm import GenomicLLM


def run_teacher_test():
    print("=" * 60)
    print("🧪 PRUEBA DE BASELINE: QWEN2 TEACHER (FLOAT32) 🧪")
    print("=" * 60)

    model_path = "/data/data/com.termux/files/home/models/gguf/qwen2-0_5b-q8_0.gguf"
    if not os.path.exists(model_path):
        print("❌ Modelo GGUF no encontrado.")
        return

    # Cargamos solo 2 bloques en modo F32 para rapidez
    print("[*] Cargando 2 bloques del Maestro F32...")
    model = GenomicLLM(model_path, num_blocks=2, mode="f32")

    prompt = "La inteligencia artificial es"
    print(f"\n📝 Prompt: '{prompt}'")

    output = model.generate(prompt, max_new_tokens=10, temperature=0.7)
    print(f"\n✨ Resultado Maestro: '{output}'")


if __name__ == "__main__":
    run_teacher_test()
