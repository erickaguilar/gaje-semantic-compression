import os
import sys
import numpy as np
import time

# Ensure we use the local package
sys.path.append(os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM

def main():
    print("🧬 GAJE PROTOCOL: GENOMIC CHAT v0.6.1 (Kernel Fixed)")
    print("=" * 60)

    model_path = "/data/data/com.termux/files/home/models/smollm2-135m-f16.gguf"
    
    if not os.path.exists(model_path):
        # Intentar buscar en rutas comunes
        possible_paths = [
            model_path,
            "models/smollm2-135m-f16.gguf",
            "smollm2-135m-f16.gguf"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                model_path = p
                break
        else:
            print(f"❌ Error: El modelo GGUF no se encuentra.")
            print("Por favor, asegúrate de que el modelo SmolLM2 F16 esté disponible.")
            return

    # Cargamos el motor estabilizado con Carga Completa
    llm = GenomicLLM(model_path, num_blocks=None)
    
    print("\n" + "✨ SISTEMA LISTO. Escribe '/exit' para salir.")
    print("-" * 60)

    while True:
        try:
            prompt = input("\n👤 Usuario: ")
            if prompt.lower() in ['/exit', 'quit', 'exit']:
                break
            
            if not prompt.strip():
                continue

            print("\n🤖 GAJE: ", end="", flush=True)
            
            start_time = time.time()
            token_count = 0
            
            # Inferencia Generativa (V0.6.1 con Kernel de Rust optimizado)
            for token_text in llm.generate(prompt, max_new_tokens=50):
                print(token_text, end="", flush=True)
                token_count += 1
            
            duration = time.time() - start_time
            tps = token_count / duration if duration > 0 else 0
            
            print(f"\n\n   [Métricas: {duration:.2f}s | {tps:.2f} t/s | Precision Mixta: Activa]")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n⚠️ Error durante la inferencia: {e}")
            import traceback
            traceback.print_exc()

    print("\n[!] Organismo Genómico hibernando. Adiós.")

if __name__ == "__main__":
    main()
