import os
import sys

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM


def main():
    model_path = "models/production/qwen2_fixed.gaje"
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        return

    print(f"[*] Cargando {model_path} en Python para diagnóstico...")
    try:
        model = GenomicLLM.load_genomic(model_path)
        tokenizer = model.tokenizer

        prompt = "Hola, ¿quién eres?"
        print(f"👤 Usuario > {prompt}")

        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        print(f"[*] Tokens iniciales: {tokens}")

        print("🧬 Organismo (Python) > ", end="", flush=True)

        generated = tokens
        for _ in range(50):
            logits = model.forward(generated, clear_cache=False)[-1]

            # Muestreo Greedy para diagnóstico (el más determinista)
            next_token = int(np.argmax(logits))

            if next_token == tokenizer.eos_token_id or next_token == 151643:
                break

            generated.append(next_token)
            word = tokenizer.decode([next_token], skip_special_tokens=True)
            print(word, end="", flush=True)

        print("\n\n[+] Prueba completada.")

    except Exception as e:
        print(f"❌ Error durante el diagnóstico: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import numpy as np

    main()
