import os
import sys
import argparse
import time
import numpy as np

# Asegurar que usamos el código local de 'python/'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from gaje.core._impl import ArchConfig, ModelConfig, init_born_genomic_model, save_genomic_model, RustGenomicLLM
from gaje.nn.stabilized import GenomicLLM
from gaje.nn.trainer import GenomicTrainer
from gaje.utils.version import get_project_version
from tokenizers import Tokenizer

def main():
    parser = argparse.ArgumentParser(description="🧬 GAJE Born-Genomic Training Phase 1")
    parser.add_argument("--name", type=str, default="GajeSmall-v1", help="Nombre del organismo")
    parser.add_argument("--blocks", type=int, default=4, help="Número de bloques")
    parser.add_argument("--embd", type=int, default=512, help="Dimensión de embedding")
    parser.add_argument("--epochs", type=int, default=50, help="Épocas de entrenamiento")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning Rate")
    parser.add_argument("--dataset", type=str, default="data/datasets/dataset_born_2000.txt", help="Ruta al dataset")
    parser.add_argument("--tokenizer", type=str, default="models/core/tokenizer.json", help="Ruta al tokenizador")
    args = parser.parse_args()

    version = get_project_version()
    print(f"🧬 Iniciando Nacimiento Genómico (v{version}): {args.name}")
    print("-" * 50)

    # 1. Configuración del Organismo
    arch = ArchConfig(
        name=args.name,
        version=version,
        tokenizer_id=args.tokenizer, # Use the local path
        rope_base=1000000.0,
        ffn_act="swiglu",
        use_genomic_norm=True,
        rope_style="split"
    )
    
    config = ModelConfig(
        config=arch,
        n_embd=args.embd,
        n_head=8,
        n_head_kv=8,
        n_blocks=args.blocks,
        vocab_size=151936, # Qwen2 Vocab Size
        eps=1e-6
    )

    model_dir = f"models/checkpoints/{args.name.lower()}"
    model_path = f"{model_dir}/model.gaje"
    os.makedirs(model_dir, exist_ok=True)

    # 2. Inicialización Nativa (Pesos aleatorios)
    print(f"[*] Inicializando organismo en {model_path}...")
    rust_llm = init_born_genomic_model(model_path, config, 151936)
    
    # Cargar tokenizador
    if os.path.exists(args.tokenizer):
        tokenizer = Tokenizer.from_file(args.tokenizer)
        print(f"[*] Tokenizador cargado desde {args.tokenizer}")
    else:
        print(f"❌ Error: Tokenizador no encontrado en {args.tokenizer}")
        return

    # Envolver en la clase estabilizada de Python
    llm = GenomicLLM(None, config=arch, n_embd=args.embd, num_blocks=args.blocks)
    llm.rust_llm = rust_llm
    llm.tokenizer = tokenizer
    llm.config = config # This is the ModelConfig

    # 3. Preparación del Dataset
    print(f"[*] Cargando dataset desde {args.dataset}...")
    if not os.path.exists(args.dataset):
        print(f"❌ Error: Dataset no encontrado.")
        return
        
    with open(args.dataset, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"📊 Dataset cargado: {len(lines)} líneas.")

    # 4. Bucle de Entrenamiento Híbrido (Phase 1)
    trainer = GenomicTrainer(llm, lr=args.lr, use_torch=True)
    
    print(f"[*] Iniciando entrenamiento Born-Genomic...")
    start_time = time.time()
    
    trainer.fit(lines, epochs=args.epochs)
    
    total_duration = time.time() - start_time
    print(f"\n✅ Entrenamiento completado en {total_duration/60:.2f} minutos.")

    # 5. Guardar Organismo Final
    print(f"[*] Guardando organismo evolucionado...")
    save_genomic_model(model_path, rust_llm, config, args.tokenizer)
    print(f"✨ Organismo '{args.name}' nacido y guardado exitosamente.")

if __name__ == "__main__":
    main()
