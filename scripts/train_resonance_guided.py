import os
import sys
import argparse
import time
import numpy as np

# Asegurar que usamos el código local de 'python/'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from gaje.core._impl import NativeLoader, save_genomic_model
from gaje.nn.stabilized import GenomicLLM
from gaje.nn.trainer import GenomicTrainer
from gaje.utils.version import get_project_version
from tokenizers import Tokenizer

def main():
    parser = argparse.ArgumentParser(description="🧬 GAJE Graph-Guided Resonance Training (Phase 4.1)")
    parser.add_argument("--model", type=str, default="models/checkpoints/gold_embryo.gaje", help="Ruta al modelo base")
    parser.add_argument("--topology", type=str, default="models/core/topology_es.json", help="Ruta al mapa de topología")
    parser.add_argument("--dataset", type=str, default="data/datasets/dataset_es.txt", help="Ruta al dataset (ES)")
    parser.add_argument("--tokenizer", type=str, default="models/core/tokenizer.json", help="Ruta al tokenizador")
    parser.add_argument("--epochs", type=int, default=10, help="Épocas de entrenamiento")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--output", type=str, default="models/checkpoints/gold_embryo_guided.gaje", help="Ruta de guardado")
    args = parser.parse_args()

    version = get_project_version()
    print(f"🧬 Iniciando Entrenamiento por Resonancia Guiado (v{version})")
    print(f"[*] Modelo Base: {args.model}")
    print(f"[*] Topología: {args.topology}")
    print("-" * 50)

    # 1. Carga del Modelo y Configuración
    if not os.path.exists(args.model):
        print(f"❌ Error: Modelo base no encontrado en {args.model}")
        return

    loader = NativeLoader(args.model)
    config = loader.py_load_config()
    rust_llm = loader.py_load_llm()
    
    # Inyectar Topología Nativa
    if os.path.exists(args.topology):
        print(f"[*] Inyectando topología relacional desde {args.topology}...")
        rust_llm.load_topology(args.topology)
    else:
        print(f"⚠️ Advertencia: Topología no encontrada en {args.topology}. Continuando sin guía.")

    # Cargar tokenizador
    if os.path.exists(args.tokenizer):
        tokenizer = Tokenizer.from_file(args.tokenizer)
        print(f"[*] Tokenizador cargado desde {args.tokenizer}")
    else:
        print(f"❌ Error: Tokenizador no encontrado.")
        return

    # Envolver para Python
    llm = GenomicLLM(None, config=config.config, n_embd=config.n_embd, num_blocks=config.n_blocks)
    llm.rust_llm = rust_llm
    llm.tokenizer = tokenizer
    llm.config = config

    # 2. Preparación del Dataset
    print(f"[*] Cargando dataset desde {args.dataset}...")
    if not os.path.exists(args.dataset):
        print(f"❌ Error: Dataset no encontrado.")
        return
        
    with open(args.dataset, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"📊 Dataset cargado: {len(lines)} líneas.")

    # 3. Bucle de Entrenamiento por Resonancia
    # Usamos el entrenador estabilizado que interactúa con el núcleo de Rust
    trainer = GenomicTrainer(llm, lr=args.lr, use_torch=False) # Pure Native Training
    
    print(f"[*] Iniciando fase de resonancia guidada...")
    start_time = time.time()
    
    # Entrenamiento por resonancia: pocas épocas, LR bajo para estabilizar
    trainer.fit(lines, epochs=args.epochs)
    
    total_duration = time.time() - start_time
    print(f"\n✅ Fase de resonancia completada en {total_duration/60:.2f} minutos.")

    # 4. Guardar Organismo Final
    print(f"[*] Guardando organismo refinado en {args.output}...")
    save_genomic_model(args.output, rust_llm, config, args.tokenizer)
    print(f"✨ Entrenamiento completado exitosamente.")

if __name__ == "__main__":
    main()
