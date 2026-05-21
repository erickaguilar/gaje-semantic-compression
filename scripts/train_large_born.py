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
    parser = argparse.ArgumentParser(description="🧬 GAJE PROTOCOL: BORN-GENOMIC TRAINING (LARGE)")
    parser.add_argument("--name", type=str, default="GajeExpert-v1", help="Name of the model")
    parser.add_argument("--blocks", type=int, default=8, help="Number of transformer blocks (default: 8)")
    parser.add_argument("--embd", type=int, default=512, help="Embedding dimension (default: 512)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 0.005)")
    parser.add_argument("--dataset", type=str, default="data/datasets/dataset_entrenamiento.txt", help="Path to dataset")
    args = parser.parse_args()

    print(f"🧬 GAJE PROTOCOL: BORN-GENOMIC LARGE TRAINING")
    print("=" * 60)

    # 1. Configuración de Arquitectura Personalizada
    config = get_config("qwen2") # Usamos qwen2 como base
    config.name = args.name
    
    # Nota: El motor actual usa los parámetros pasados a GenomicLLM
    print(f"[*] Configurando organismo: {args.name}")
    print(f"    - Bloques: {args.blocks}")
    print(f"    - Dimensión: {args.embd}")

    # 2. Inicialización Born-Genomic con dimensiones personalizadas
    # Modificamos el config temporalmente para la inicialización
    llm = GenomicLLM(model_path=None, num_blocks=args.blocks, config=config, n_embd=args.embd)
    
    # Ajustamos las dimensiones internas si es necesario (el constructor de GenomicLLM suele inferir del config o defaults)
    # Para esta prueba, confiaremos en que el constructor maneja las dimensiones base.
    
    print(f"[*] Modelo inicializado. Vocabulario: {len(llm.tokenizer)}")

    # 3. Carga de Dataset Real
    if os.path.exists(args.dataset):
        with open(args.dataset, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Filtramos líneas vacías y limpiamos
        dataset = [line.strip() for line in lines if len(line.strip()) > 10]
        print(f"[*] Dataset cargado: {len(dataset)} interacciones encontradas.")
    else:
        print(f"[!] Error: Dataset {args.dataset} no encontrado.")
        return

    # 4. Iniciar Entrenamiento
    trainer = GenomicTrainer(llm, lr=args.lr)
    
    print(f"\n[*] Iniciando fase de nacimiento (Born)...")
    start_time = time.time()
    trainer.fit(dataset, epochs=args.epochs)
    duration = time.time() - start_time
    
    print(f"\n[*] Nacimiento finalizado en {duration:.2f} segundos.")
    
    # 5. Validación de Identidad
    test_prompts = ["Usuario: Hola, ¿quién eres?", "Asistente: Hola. Soy"]
    print("\n[*] Validación de Coherencia:")
    for prompt in test_prompts:
        print(f"    - Prompt: '{prompt}'")
        print(f"    - Respuesta: ", end="", flush=True)
        for token_text in llm.generate(prompt, max_new_tokens=15, temperature=0.2):
            print(token_text, end="", flush=True)
        print("\n")
    
    # 6. Guardar el organismo experto
    out_dir = f"models/checkpoints/{args.name.lower()}"
    os.makedirs(out_dir, exist_ok=True)
    llm.save(out_dir)
    print(f"[+] Organismo guardado en: {out_dir}")

if __name__ == "__main__":
    main()
