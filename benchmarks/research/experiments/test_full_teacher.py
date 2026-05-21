import os
import sys
import numpy as np
import time

# Añadir el directorio python al path
sys.path.append(os.path.abspath("python"))

from genomize_llm import GenomicLLM

def test_full_teacher():
    model_path = "/data/data/com.termux/files/home/models/qwen2-0_5b-q8_0.gguf"
    if not os.path.exists(model_path):
        print("❌ Modelo no encontrado.")
        return

    print("="*60)
    print("🧪 PRUEBA DE ARQUITECTURA: MAESTRO F32 (24 BLOQUES) 🧪")
    print("="*60)

    # Cargamos los 24 bloques en modo F32
    print(f"[*] Cargando 24 bloques en modo F32...")
    model = GenomicLLM(model_path, num_blocks=24, mode='f32')
    
    prompt = "El ADN es la"
    print(f"\n📝 Prompt: '{prompt}'")
    
    # Generar 10 tokens
    output = model.generate(prompt, max_new_tokens=10, temperature=0.1)
    print(f"\n\n✨ Resultado Maestro: '{output}'")

if __name__ == "__main__":
    test_full_teacher()
