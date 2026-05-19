import os
import sys
import argparse
import time

# Ensure we use the local package first
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM
from gaje.nn.configs import get_config
from gaje.nn.trainer import GenomicTrainer

def main():
    parser = argparse.ArgumentParser(description="🧬 GAJE PROTOCOL: BORN-GENOMIC TRAINING")
    parser.add_argument("--arch", type=str, default="qwen2", help="Architecture config to use (default: qwen2)")
    parser.add_argument("--blocks", type=int, default=2, help="Number of transformer blocks (default: 2)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    args = parser.parse_args()

    print(f"🧬 GAJE PROTOCOL: BORN-GENOMIC TRAINING v0.7.0")
    print("=" * 60)

    # 1. Configuración de Arquitectura
    config = get_config(args.arch)
    print(f"[*] Configuración cargada: {config.name}")

    # 2. Inicialización Born-Genomic
    # No pasamos model_path, así que el motor inicializa tensores genómicos con centroides aleatorios.
    llm = GenomicLLM(model_path=None, num_blocks=args.blocks, config=config)
    print(f"[*] Modelo inicializado con {args.blocks} bloques ocultos y vocabulario de {len(llm.tokenizer)}.")

    # 3. Micro-Dataset de prueba (Para probar Overfitting / Memorización)
    dataset = [
        "El protocolo GAJE es nativo.",
        "El protocolo GAJE comprime semántica.",
        "GAJE utiliza ADN en lugar de pesos.",
        "Qwen es la arquitectura base."
    ]
    print("\n[*] Dataset de prueba:")
    for text in dataset:
        print(f"    - '{text}'")

    # 4. Iniciar Entrenamiento
    trainer = GenomicTrainer(llm, lr=args.lr)
    
    start_time = time.time()
    trainer.fit(dataset, epochs=args.epochs)
    duration = time.time() - start_time
    
    print(f"\n[*] Entrenamiento finalizado en {duration:.2f} segundos.")
    
    # 5. Generación de prueba (Inferencia Zero-Shot tras el entrenamiento)
    prompt = "El protocolo GAJE"
    print(f"\n[*] Generando texto a partir de: '{prompt}'")
    print("🤖 GAJE: ", end="", flush=True)
    
    for token_text in llm.generate(prompt, max_new_tokens=10, temperature=0.1):
        print(token_text, end="", flush=True)
    
    print("\n")
    
    # 6. Guardar el organismo
    out_dir = "models/born_genomic_qwen"
    os.makedirs(out_dir, exist_ok=True)
    llm.save(out_dir)

if __name__ == "__main__":
    main()
