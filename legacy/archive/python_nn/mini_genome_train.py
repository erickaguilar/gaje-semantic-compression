import os
import sys
import time
import numpy as np
import torch

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from python.gaje.nn.stabilized import GenomicLLM
from python.gaje.nn.configs import get_config
from python.gaje.native.trainer import GenomicTrainer

def main():
    print("🧬 GAJE PHASE 3: TRAINING THE MINI-GENOME")
    print("="*60)
    
    # 1. Configuración del Micro-Modelo (10M-30M aprox)
    config = get_config("gaje_native")
    # Arquitectura ultra-ligera para Termux
    model = GenomicLLM(num_blocks=4, config=config)
    model.n_embd = 256 # Forzamos dimensión pequeña
    # Nota: El re-dimensionamiento real ocurre en el __init__, 
    # pero para el demo forzamos los parámetros de entrenamiento.
    
    trainer = GenomicTrainer(model, lr=0.1)
    
    # 2. Cargar Dataset de Memorización
    dataset_path = "data/training/datasets/mini_story.txt"
    with open(dataset_path, "r") as f:
        story = f.read()
    
    print(f"[*] Dataset cargado: {len(story)} caracteres.")
    tokens = model.tokenizer.encode(story)
    print(f"[*] Tokens: {len(tokens)}")
    
    input_ids = tokens[:-1]
    target_ids = tokens[1:]
    
    # 3. Bucle de Entrenamiento (Sobreadiestramiento para memorización)
    epochs = 100
    print(f"[*] Iniciando entrenamiento (Epochs: {epochs}, LR: {trainer.lr})...")
    
    history = []
    start_train = time.time()
    for epoch in range(epochs):
        loss = trainer.train_step(input_ids, target_ids)
        history.append(loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss:.6f}")
        if loss < 0.001:
            print(f"[*] Convergencia alcanzada en la época {epoch+1}")
            break
            
    end_train = time.time()
    print(f"\n[*] Entrenamiento finalizado en {end_train - start_train:.2f}s")
    
    # 4. Validación de Memorización
    print("\n[*] Validando Memorización (Generación):")
    prompt = "Once upon a time"
    print(f"    Prompt: '{prompt}'")
    print("    Salida: ", end="", flush=True)
    
    for token in model.generate(prompt, max_new_tokens=50, temperature=0.1):
        print(token, end="", flush=True)
    print("\n")
    
    # 5. Persistencia
    checkpoint_dir = "data/training/checkpoints/mini_genome_v1"
    model.save(checkpoint_dir)
    print("="*60)
    print("🚀 FASE 3 COMPLETADA: El Mini-Genome ha nacido y ha sido guardado.")
    print("="*60)

if __name__ == "__main__":
    main()
