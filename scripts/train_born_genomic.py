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
    parser.add_argument("--evolve", action="store_true", help="Enable evolutionary refinement phase")
    parser.add_argument("--gen", type=int, default=20, help="Number of evolutionary generations (default: 20)")
    parser.add_argument("--dataset", type=str, default="data/datasets/dataset_entrenamiento.txt", help="Path to dataset file")
    args = parser.parse_args()

    print(f"🧬 GAJE PROTOCOL: BORN-GENOMIC TRAINING v0.7.1")
    print("=" * 60)

    # 1. Configuración de Arquitectura
    config = get_config(args.arch)
    print(f"[*] Configuración cargada: {config.name}")

    # 2. Inicialización Born-Genomic
    # No pasamos model_path, así que el motor inicializa tensores genómicos con centroides aleatorios.
    llm = GenomicLLM(model_path=None, num_blocks=args.blocks, config=config)
    print(f"[*] Modelo inicializado con {args.blocks} bloques ocultos y vocabulario de {len(llm.tokenizer)}.")

    # 3. Carga de Dataset
    if os.path.exists(args.dataset):
        with open(args.dataset, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Filtramos líneas vacías o muy cortas
        dataset = [line.strip() for line in lines if len(line.strip()) > 10]
        print(f"[*] Dataset cargado: {len(dataset)} interacciones desde {args.dataset}")
    else:
        print(f"[!] Aviso: Dataset no encontrado en {args.dataset}. Usando dataset de respaldo.")
        dataset = [
            "El protocolo GAJE es nativo.",
            "El protocolo GAJE comprime semántica.",
            "GAJE utiliza ADN en lugar de pesos.",
            "Qwen es la arquitectura base."
        ]

    # 4. Iniciar Entrenamiento
    trainer = GenomicTrainer(llm, lr=args.lr)
    
    start_time = time.time()
    print(f"[*] Fase 1: Entrenamiento por Gradientes ({args.epochs} épocas)...")
    trainer.fit(dataset, epochs=args.epochs)
    
    # 5. Fase Evolutiva (Opcional)
    if args.evolve:
        print(f"\n[*] Fase 2: Refinamiento Evolutivo ({args.gen} generaciones)...")
        trainer.evolve(dataset, generations=args.gen, mutation_scale=0.02)

    duration = time.time() - start_time
    
    print(f"\n[*] Protocolo completo finalizado en {duration:.2f} segundos.")
    
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
