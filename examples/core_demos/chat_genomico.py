import os
import sys
import time
import argparse

# Asegurar que usamos el código local de 'python/'
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
)

from gaje.nn.stabilized import GenomicLLM  # noqa: E402
from gaje.core import SessionMemory  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="🧬 GAJE PROTOCOL: GENOMIC CHAT")
    parser.add_argument(
        "--model",
        type=str,
        default="models/silver_adult_anchored.gaje",
        help="Path to the GAJE model",
    )
    parser.add_argument(
        "--blocks", type=int, default=None, help="Number of transformer blocks to load"
    )
    parser.add_argument(
        "--tokens", type=int, default=128, help="Max new tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.4, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P sampling")
    parser.add_argument(
        "--penalty", type=float, default=1.2, help="Repetition penalty"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt inicial (activa modo no interactivo)",
    )
    args = parser.parse_args()

    print("🧬 GAJE PROTOCOL: GENOMIC CHAT v0.7.1 (Estabilizado)")
    print("=" * 60)

    model_path = args.model

    if not os.path.exists(model_path):
        possible_paths = [
            model_path,
            f"models/{os.path.basename(model_path)}",
            os.path.basename(model_path),
            f"./data/models/{os.path.basename(model_path)}",
        ]
        for p in possible_paths:
            if os.path.exists(p):
                model_path = p
                break
        else:
            print(f"❌ Error: El modelo no se encuentra en {model_path}.")
            return

    # Cargamos el motor estabilizado
    print(f"[*] Inicializando GenomicLLM con {model_path}...")
    llm = GenomicLLM.load_genomic(model_path)

    # Inicializamos la Capa de Sesión (Ring Buffer)
    session_memory = SessionMemory(capacity=1024, dim=llm.n_embd)
    session_file = "session_data.bin"

    # Intentar cargar sesión previa
    if os.path.exists(session_file):
        print(f"[*] Recuperando sesión toroidal previa desde {session_file}...")
        try:
            session_memory = SessionMemory.load(session_file)
            print(f"    [+] {len(session_memory)} interacciones recuperadas.")
        except Exception as e:
            print(f"    [!] No se pudo cargar la sesión: {e}")

    # Modo No-Interactivo
    if args.prompt or not sys.stdin.isatty():
        user_input = args.prompt or "Hola, preséntate brevemente."
        print(f"\n👤 Usuario: {user_input}")

        prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        print("\n🤖 GAJE: ", end="", flush=True)

        for token_text in llm.generate(
            prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.penalty,
        ):
            print(token_text, end="", flush=True)
        print("\n\n[!] Proceso finalizado.")
        return

    print("\n✨ SISTEMA LISTO. Escribe '/exit' para salir.")
    print("-" * 60)
    chat_history = []

    while True:
        try:
            user_input = input("\n👤 Usuario: ")
            if user_input.lower() in ["/exit", "quit", "exit"]:
                print(f"[*] Guardando sesión en {session_file}...")
                session_memory.save(session_file)
                break

            if not user_input.strip():
                continue

            # 1. Recuperación de Memoria Semántica (Toroidal Recall)
            user_tokens = llm.tokenizer.encode(user_input, add_special_tokens=False)
            if hasattr(user_tokens, "ids"):
                user_tokens = user_tokens.ids
            
            # Obtenemos el hidden state del último token del input
            # Usamos forward_with_hidden para capturar la fase semántica
            _, user_phase = llm.rust_llm.forward_with_hidden(user_tokens[-1], True)
            
            # Recuperamos contexto relevante del buffer
            relevant_context = session_memory.retrieve(user_phase, top_k=2)
            
            context_prompt = ""
            if relevant_context:
                print(f"    [🔍 Memoria: {len(relevant_context)} fragmentos recuperados]")
                # Formateamos el contexto para inyectarlo al sistema
                context_prompt = "--- MEMORIA DE SESIÓN RELEVANTE ---\n"
                context_prompt += "\n".join(relevant_context)
                context_prompt += "\n----------------------------------\n"

            chat_history.append({"role": "user", "content": user_input})

            # Apply chat template (ChatML format)
            prompt = ""
            if context_prompt:
                prompt += f"<|im_start|>system\n{context_prompt}<|im_end|>\n"
                
            for msg in chat_history:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"

            print("\n🤖 GAJE: ", end="", flush=True)

            start_time = time.time()
            token_count = 0
            full_response = ""

            # Inferencia Generativa
            for token_text in llm.generate(
                prompt,
                max_new_tokens=args.tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.penalty,
            ):
                print(token_text, end="", flush=True)
                full_response += token_text
                token_count += 1

            # 2. Recirculación: Guardamos la interacción en la memoria
            # Usamos el par pregunta/respuesta como bloque semántico
            interaction_text = f"Pregunta: {user_input}\nRespuesta: {full_response.strip()}"
            session_memory.push(interaction_text, user_phase)

            chat_history.append({"role": "assistant", "content": full_response.strip()})

            duration = time.time() - start_time
            tps = token_count / duration if duration > 0 else 0

            print(f"\n\n   [Métricas: {duration:.2f}s | {tps:.2f} t/s]")

        except KeyboardInterrupt:
            print(f"\n[*] Guardando sesión en {session_file}...")
            session_memory.save(session_file)
            break
        except Exception as e:
            print(f"\n⚠️ Error durante la inferencia: {e}")

    print("\n[!] Organismo Genómico hibernando. Adiós.")


if __name__ == "__main__":
    main()
