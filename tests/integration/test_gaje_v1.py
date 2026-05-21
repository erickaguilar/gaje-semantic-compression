import os
import sys
import time

# Asegurar el uso del paquete local
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM

def test_inference():
    model_path = "models/checkpoints/gajeexpert-v1/model.gaje"
    if not os.path.exists(model_path):
        print(f"[!] Error: No se encuentra el modelo en {model_path}")
        return

    print(f"🧬 Despertando al organismo GajeExpert-v1...")
    model = GenomicLLM.load_genomic(model_path)
    
    test_prompts = [
        "Usuario: Hola, ¿quién eres?\nAsistente:",
        "Usuario: ¿Qué es la compresión genómica?\nAsistente:",
        "Usuario: ¿Cómo crear una función en Rust?\nAsistente:",
        "Usuario: ¿Qué es el protocolo GAJE?\nAsistente:"
    ]

    print("\n" + "="*50)
    print("🚀 INICIANDO SESIÓN DE PRUEBAS")
    print("="*50)

    for prompt in test_prompts:
        print(f"\n[?] Pregunta: {prompt.split('\\n')[0]}")
        print(f"🤖 GAJE: ", end="", flush=True)
        
        # Usamos parámetros conservadores para mayor estabilidad
        try:
            for token in model.generate(prompt, max_new_tokens=40, temperature=0.1, top_p=0.85):
                print(token, end="", flush=True)
        except Exception as e:
            print(f"\n[!] Error durante la generación: {e}")
        
        print("\n" + "-"*30)

if __name__ == "__main__":
    test_inference()
