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
    parser = argparse.ArgumentParser(description="🧬 GAJE PROTOCOL: SPANISH ENRICHMENT TRAINING")
    parser.add_argument("--arch", type=str, default="qwen2", help="Architecture config to use")
    parser.add_argument("--blocks", type=int, default=4, help="Number of blocks")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()

    print(f"🧬 GAJE PROTOCOL: SPANISH ENRICHMENT v0.7.0")
    print("=" * 60)

    # 1. Cargar Dataset (Usando el dataset extendido de 150 frases)
    dataset_path = "data/datasets/dataset_es_ext.txt"
    if not os.path.exists(dataset_path):
        print(f"❌ Error: {dataset_path} no encontrado.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = [line.strip() for line in f if line.strip()]

    print(f"[*] Dataset cargado: {len(dataset)} frases en español.")

    # 2. Inicialización
    config = get_config(args.arch)
    llm = GenomicLLM(model_path=None, num_blocks=args.blocks, config=config)
    
    # 3. Entrenamiento
    trainer = GenomicTrainer(llm, lr=args.lr)
    print(f"[*] Iniciando entrenamiento intensivo ({args.epochs} épocas)...")
    
    start_time = time.time()
    trainer.fit(dataset, epochs=args.epochs)
    duration = time.time() - start_time
    
    print(f"\n[*] Entrenamiento finalizado en {duration:.2f} segundos.")

    # 4. Validación Directa
    prompts = [
        "La capital de España es",
        "El ADN es",
        "Rust es un",
        "Yo soy un",
        "¿Cómo puedo"
    ]
    
    print("\n" + "=" * 40)
    print("🧪 RESULTADOS DE LA CRIANZA (DATASET EXTENDIDO)")
    print("=" * 40)
    for p in prompts:
        print(f"\nPrompt: '{p}'")
        print("🤖 GAJE: ", end="", flush=True)
        for token in llm.generate(p, max_new_tokens=15, temperature=0.1):
            print(token, end="", flush=True)
        print("\n" + "-" * 20)

    # 5. Guardar Organismo Evolucionado
    out_dir = "models/qwen2_es_v2_ext"
    os.makedirs(out_dir, exist_ok=True)
    llm.save(out_dir)
    print(f"\n✅ Organismo 'Políglota ES v2' guardado en {out_dir}")

if __name__ == "__main__":
    main()
