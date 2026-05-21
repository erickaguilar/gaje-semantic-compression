import sys
import os
import time

sys.path.append(os.path.abspath("python"))
from gaje.nn.stabilized import GenomicLLM

def main():
    model_path = "models/checkpoints/mc_optimized_qwen/model.gaje"
    print(f"[*] Intentando cargar el modelo desde: {model_path}")
    
    start = time.time()
    try:
        # Cargamos el modelo
        llm = GenomicLLM.load_genomic(model_path)
        print(f"✅ ¡Modelo cargado exitosamente en {time.time() - start:.2f}s!")
        print(f"    - Bloques Ocultos: {len(llm.blocks)}")
        print(f"    - Dimensión de Embeddings: {llm.n_embd}")
        print(f"    - Tamaño del Vocabulario: {len(llm.tokenizer)}")
        
        print("\n[*] Generando texto de prueba...")
        prompt = "Hola, mundo"
        print(f"🤖 User: {prompt}")
        print("🤖 GAJE: ", end="", flush=True)
        for token_text in llm.generate(prompt, max_new_tokens=15, temperature=0.7):
            print(token_text, end="", flush=True)
        print("\n")
        
    except Exception as e:
        print(f"❌ Error al cargar o generar con el modelo: {e}")

if __name__ == "__main__":
    main()
