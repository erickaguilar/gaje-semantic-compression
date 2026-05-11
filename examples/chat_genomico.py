import os
import sys
from python.gaje.processing.pipeline import GenomicLLM


def main():
    if len(sys.argv) < 2:
        prompt = "Cual es la capital de mexico?"
    else:
        prompt = sys.argv[1]

    model_path = "./data/models/qwen2-0_5b-instruct-fp16.gguf"

    if not os.path.exists(model_path):
        print(f"❌ Error: El modelo GGUF no se encuentra en {model_path}")
        return

    # Cargamos el modelo completo (24 bloques) para máxima fidelidad
    print("🧬 Cargando Motor Genómico Completo (24 bloques)...")
    print("⚠️ Nota: Esto puede tomar unos segundos extra en inicializar.")
    llm = GenomicLLM(model_path, num_blocks=24)

    # Generar respuesta con mejores parámetros
    # Usamos un prompt más natural y pedimos más tokens.
    response = llm.generate(prompt, max_new_tokens=25)

    print("\n" + "=" * 40)
    print("🤖 GAJE responde:")
    print(response)
    print("=" * 40)


if __name__ == "__main__":
    main()
