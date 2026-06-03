import os
import sys
import time
import argparse

# Asegurar que usamos el código local de 'python/'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python")))

from gaje.nn.stabilized import GenomicLLM
from gaje.core import SessionMemory

def main():
    parser = argparse.ArgumentParser(description="🧬 GAJE PROTOCOL: CHAT SOBERANO (Silver Adult)")
    parser.add_argument("--model", type=str, default="models/production/silver_adult_steel.gaje", help="Path to the GAJE model")
    parser.add_argument("--tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P sampling")
    parser.add_argument("--penalty", type=float, default=1.2, help="Repetition penalty")
    parser.add_argument("--spiking", action="store_true", help="Use Neuromorphic Spiking Inference")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt inicial (activa modo no interactivo)")
    args = parser.parse_args()

    print("🧬 GAJE PROTOCOL: CHAT SOBERANO v1.5 (Silver Adult)")
    print("=" * 60)

    model_path = args.model
    if not os.path.exists(model_path):
        # Intentar rutas comunes
        for p in ["models/production/silver_adult_steel.gaje", "models/silver_adult_anchored.gaje"]:
            if os.path.exists(p):
                model_path = p
                break
        else:
            print(f"❌ Error: El modelo no se encuentra en {model_path}.")
            return

    print(f"[*] Cargando Organismo: {model_path}")
    llm = GenomicLLM.load_genomic(model_path)
    session_memory = SessionMemory(capacity=1024, dim=llm.n_embd)
    session_file = "session_data.bin"

    if os.path.exists(session_file):
        try:
            session_memory = SessionMemory.load(session_file)
            print(f"    [+] {len(session_memory)} interacciones recuperadas de la sesión previa.")
        except: pass

    # Modo No-Interactivo
    if args.prompt or not sys.stdin.isatty():
        user_input = args.prompt or "Hola, preséntate brevemente."
        print(f"\n👤 Usuario: {user_input}")
        prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        print("\n🤖 GAJE: ", end="", flush=True)

        for token in llm.generate(prompt, max_new_tokens=args.tokens, temperature=args.temperature, top_p=args.top_p, repetition_penalty=args.penalty, use_spiking=args.spiking):
            print(token, end="", flush=True)
        print("\n\n[!] Proceso finalizado.")
        return

    print("\n✨ SISTEMA LISTO. Escribe '/exit' para salir.")
    if args.spiking: print("🧠 MODO NEUROMÓRFICO (SPIKING) ACTIVADO")
    print("-" * 60)
    chat_history = []

    while True:
        try:
            user_input = input("\n👤 Usuario: ")
            if user_input.lower() in ["/exit", "quit", "exit"]:
                session_memory.save(session_file)
                break
            if not user_input.strip(): continue

            # Recuperación Semántica
            user_tokens = llm.tokenizer.encode(user_input, add_special_tokens=False)
            if hasattr(user_tokens, "ids"): user_tokens = user_tokens.ids
            _, user_phase = llm.rust_llm.forward_with_hidden(user_tokens[-1], True)
            relevant_context = session_memory.retrieve(user_phase, top_k=2)
            
            context_prompt = ""
            if relevant_context:
                context_prompt = "--- MEMORIA DE SESIÓN RELEVANTE ---\n" + "\n".join(relevant_context) + "\n----------------------------------\n"

            chat_history.append({"role": "user", "content": user_input})
            prompt = f"<|im_start|>system\n{context_prompt}<|im_end|>\n" if context_prompt else ""
            for msg in chat_history:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"

            print("\n🤖 GAJE: ", end="", flush=True)
            start_time = time.time()
            token_count = 0
            full_response = ""

            for token in llm.generate(prompt, max_new_tokens=args.tokens, temperature=args.temperature, top_p=args.top_p, repetition_penalty=args.penalty, use_spiking=args.spiking):
                print(token, end="", flush=True)
                full_response += token
                token_count += 1

            # Recirculación
            session_memory.push(f"Pregunta: {user_input}\nRespuesta: {full_response.strip()}", user_phase)
            chat_history.append({"role": "assistant", "content": full_response.strip()})
            duration = time.time() - start_time
            print(f"\n\n   [Métricas: {duration:.2f}s | {token_count/duration:.2f} t/s]")

        except KeyboardInterrupt:
            session_memory.save(session_file)
            break
        except Exception as e:
            print(f"\n⚠️ Error: {e}")

if __name__ == "__main__":
    main()
