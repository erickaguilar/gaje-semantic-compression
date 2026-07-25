from llama_cpp import Llama
import os


def test_gguf_integrity():
    model_path = "/data/data/com.termux/files/home/models/gguf/qwen2-0_5b-q8_0.gguf"
    if not os.path.exists(model_path):
        print(f"❌ Error: El archivo {model_path} no existe.")
        return

    print("=" * 60)
    print("🎯 VALIDACIÓN DE MODELO: LLAMA.CPP BASELINE")
    print("=" * 60)

    try:
        # Inicializamos llama.cpp con el modelo Qwen2
        # n_ctx=512 para un contexto ligero, verbose=False para limpiar el output
        llm = Llama(model_path=model_path, n_ctx=512, verbose=False)

        test_phrase = "Paris is the capital of"
        print(f"\n📝 Probando frase: '{test_phrase}'")

        # Generamos una respuesta corta
        output = llm(test_phrase, max_tokens=10, stop=["\n"], echo=True)

        generated_text = output["choices"][0]["text"]
        print(f"\n✨ Resultado Llama.cpp: '{generated_text}'")

        if "France" in generated_text:
            print("\n✅ EL MODELO ES COHERENTE. El archivo GGUF está sano.")
        else:
            print(
                "\n⚠️ EL MODELO NO RESPONDIÓ LO ESPERADO. Podría haber un problema de prompt o de cuantización en el archivo original."
            )

    except Exception as e:
        print(f"\n❌ Error al cargar llama.cpp: {e}")


if __name__ == "__main__":
    test_gguf_integrity()
