import os
import sys
import numpy as np
import time

# Ensure we use the local package
sys.path.append(os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM

def main():
    print("🧬 GAJE PROTOCOL: GENOMIC CHAT v0.6.0 (Phase 12 Stable)")
    print("=" * 60)

    model_path = "/data/data/com.termux/files/home/models/smollm2-135m-f16.gguf"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: El modelo GGUF no se encuentra en {model_path}")
        print("Por favor, descarga el modelo SmolLM2 F16 primero.")
        return

    # Cargamos el motor estabilizado
    # Nota: Usamos 8 bloques para un balance entre velocidad y coherencia en la demo
    print(f"[*] Sincronizando Organismo Genómico (8 bloques)...")
    llm = GenomicLLM(model_path, num_blocks=8)
    
    print("\n" + "✨ SISTEMA LISTO. Escribe '/exit' para salir.")
    print("-" * 60)

    while True:
        try:
            prompt = input("\n👤 Usuario: ")
            if prompt.lower() in ['/exit', 'quit', 'exit']:
                break
            
            if not prompt.strip():
                continue

            print("\n🧬 Pensando (Genomic Inference)...")
            
            # Tokenización
            tokens = llm.tokenizer.encode(prompt, add_special_tokens=False)
            
            start_time = time.time()
            
            # Inferencia Forward (V0.6.0 Nativa con DGI y Sparse Fidelity)
            all_logits = llm.forward(tokens)
            logits = all_logits[-1] # Tomamos el último token
            
            duration = time.time() - start_time
            
            # Decodificación Top-1 (Simple para la demo)
            next_token_id = np.argmax(logits)
            response = llm.tokenizer.decode([next_token_id])
            
            print(f"🤖 GAJE: {response}")
            print(f"   [Métricas: {duration*1000:.2f}ms | Precision Mixta: 2/4/6-bit Activa]")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"⚠️ Error durante la inferencia: {e}")

    print("\n[!] Organismo Genómico hibernando. Adiós.")

if __name__ == "__main__":
    main()
