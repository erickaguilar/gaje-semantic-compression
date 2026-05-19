import os
import sys
import time
import argparse

# Asegurar el uso del paquete local
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM
from gaje.nn.configs import get_config
from gaje.nn.trainer import GenomicTrainer

def main():
    parser = argparse.ArgumentParser(description="🧬 GAJE PROTOCOL: SMOL-DISTILLATION")
    parser.add_argument("--source", type=str, default="models/SmolLM2-135M-Instruct-Q8_0.gguf", help="Source GGUF model")
    parser.add_argument("--name", type=str, default="GajeSmol-v1", help="Name of the distilled organism")
    parser.add_argument("--epochs", type=int, default=10, help="Refinement epochs (default: 10)")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"[!] Error: No se encuentra el modelo fuente en {args.source}")
        return

    print(f"🧬 PROTOCOLO DE DESTILACIÓN GAJE: {args.name}")
    print("=" * 60)
    print(f"[*] Fuente: {args.source}")
    
    # 1. Cargar y Genomizar (Destilación de Pesos)
    # Al pasar un path al constructor, GenomicLLM realiza la genomización de los pesos originales.
    start_distill = time.time()
    llm = GenomicLLM(model_path=args.source)
    distill_time = time.time() - start_distill
    print(f"[*] Destilación completada en {distill_time:.2f}s")

    # 2. Carga de Dataset para Refinamiento de Identidad
    dataset_path = "dataset_entrenamiento.txt"
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        dataset = [line.strip() for line in lines if len(line.strip()) > 10]
        print(f"[*] Dataset de refinamiento cargado: {len(dataset)} interacciones.")
    else:
        print("[!] Dataset no encontrado, saltando refinamiento.")
        dataset = []

    # 3. Refinamiento Genómico (Fine-tuning post-destilación)
    if dataset and args.epochs > 0:
        print(f"\n[*] Iniciando refinamiento genómico ({args.epochs} épocas)...")
        trainer = GenomicTrainer(llm, lr=0.001) # LR bajo para no romper la sabiduría heredada
        trainer.fit(dataset, epochs=args.epochs)

    # 4. Guardar el nuevo organismo
    out_dir = f"models/{args.name.lower()}"
    os.makedirs(out_dir, exist_ok=True)
    llm.save(out_dir)
    print(f"\n[+] Organismo destilado guardado en {out_dir}")

    # 5. Prueba de Coherencia
    prompt = "Usuario: Hola, ¿quién eres?\nAsistente:"
    print(f"\n[*] Prueba de Coherencia Post-Destilación:")
    print(f"🤖 GAJE: ", end="", flush=True)
    for token in llm.generate(prompt, max_new_tokens=30, temperature=0.3):
        print(token, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    main()
