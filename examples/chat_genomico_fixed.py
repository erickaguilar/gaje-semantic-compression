import os
import sys
import numpy as np
from gaje.nn.stabilized import GenomicLLM

def main():
    if len(sys.argv) < 2:
        prompt = "Cual es la capital de mexico?"
    else:
        prompt = sys.argv[1]

    model_path = "/data/data/com.termux/files/home/models/qwen2-0_5b-q8_0.gguf"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: El modelo GGUF no se encuentra en {model_path}")
        return

    # Cargamos el modelo completo (24 bloques) para máxima fidelidad
    print(f"🧬 Cargando Motor Genómico Completo (24 bloques)...")
    print(f"⚠️ Nota: Esto puede tomar unos segundos extra en inicializar.")
    llm = GenomicLLM(model_path, num_blocks=24)
    
    # Generar respuesta con mejores parámetros
    # Usamos un prompt más natural y pedimos más tokens.
    
    # Simple Greedy Search
    generated = prompt
    tokens = llm.tokenizer.encode(prompt, add_special_tokens=False)
    
    print("[*] Generando", end="", flush=True)
    for _ in range(max_new_tokens):
        logits = llm.forward(tokens)
        next_token = np.argmax(logits[-1])
        tokens.append(int(next_token))
        word = llm.tokenizer.decode([next_token])
        generated += word
        print(".", end="", flush=True)
    print(" [Hecho]")
    response = generated

    
    print("\n" + "="*40)
    print(f"🤖 GAJE responde:")
    print(response)
    print("="*40)

if __name__ == "__main__":
    main()
