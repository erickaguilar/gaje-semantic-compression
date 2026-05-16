import os
import sys
import numpy as np
import time
import argparse

# Ensure we use the local package
sys.path.append(os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM

def main():
    parser = argparse.ArgumentParser(description="🧬 GAJE PROTOCOL: GENOMIC CHAT")
    parser.add_argument("--model", type=str, default="/data/data/com.termux/files/home/models/smollm2-135m-f16.gguf", help="Path to the GGUF model")
    parser.add_argument("--blocks", type=int, default=None, help="Number of transformer blocks to load")
    parser.add_argument("--tokens", type=int, default=100, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.8, help="Top-P sampling")
    parser.add_argument("--penalty", type=float, default=1.15, help="Repetition penalty")
    args = parser.parse_args()

    print("🧬 GAJE PROTOCOL: GENOMIC CHAT v0.6.1 (Modernizado)")
    print("=" * 60)

    model_path = args.model
    
    if not os.path.exists(model_path):
        possible_paths = [
            model_path,
            f"models/{os.path.basename(model_path)}",
            os.path.basename(model_path)
        ]
        for p in possible_paths:
            if os.path.exists(p):
                model_path = p
                break
        else:
            print(f"❌ Error: El modelo GGUF no se encuentra en {model_path}.")
            print("Por favor, proporciona una ruta válida usando --model.")
            return

    # Cargamos el motor estabilizado
    print(f"[*] Inicializando GenomicLLM con {model_path}...")
    llm = GenomicLLM(model_path, num_blocks=args.blocks)
    
    print("\n✨ SISTEMA LISTO. Escribe '/exit' para salir.")
    print("-" * 60)

    # Initialize chat history for templating
    chat_history = []

    while True:
        try:
            user_input = input("\n👤 Usuario: ")
            if user_input.lower() in ['/exit', 'quit', 'exit']:
                break
            
            if not user_input.strip():
                continue
                
            chat_history.append({"role": "user", "content": user_input})
            
            # Apply chat template (ChatML format)
            prompt = ""
            for msg in chat_history:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"

            print("\n🤖 GAJE: ", end="", flush=True)
            
            start_time = time.time()
            token_count = 0
            full_response = ""
            
            # Inferencia Generativa
            for token_text in llm.generate(prompt, max_new_tokens=args.tokens, temperature=args.temperature, top_p=args.top_p, repetition_penalty=args.penalty):
                print(token_text, end="", flush=True)
                full_response += token_text
                token_count += 1
            
            chat_history.append({"role": "assistant", "content": full_response.strip()})
            
            duration = time.time() - start_time
            tps = token_count / duration if duration > 0 else 0
            
            # Detect if mixed precision was actually used (experimental)
            pm_status = "Activa" if any(b.q_gen.precision_mask for b in llm.rust_llm.blocks) else "Inactiva"

            print(f"\n\n   [Métricas: {duration:.2f}s | {tps:.2f} t/s | Precision Mixta: {pm_status}]")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n⚠️ Error durante la inferencia: {e}")
            import traceback
            traceback.print_exc()

    print("\n[!] Organismo Genómico hibernando. Adiós.")

if __name__ == "__main__":
    main()
