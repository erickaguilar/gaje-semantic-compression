import os
import sys
import time
import json
import numpy as np
import argparse

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from gaje.nn.stabilized import GenomicLLM
from gaje.utils.version import get_project_version
import gaje.core._impl as engine

def save_smg1_model(output_path, layers, vocab_size, dim_latent, dim_logic, centroides):
    """Guarda el modelo SMG-1 destilado en un formato compatible con GajeDatabase."""
    version = get_project_version()
    print(f"[*] Guardando modelo SMG-1 (v{version}) en: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    writer = engine.GajeDatabaseWriter(output_path)
    
    # 1. Guardar Metadatos
    metadata = {
        "type": "smg1_spiking",
        "version": version,
        "vocab_size": vocab_size,
        "dim_latent": dim_latent,
        "dim_logic": dim_logic,
        "centroides": centroides,
        "layers": [
            {"name": "l0", "in": vocab_size, "out": dim_latent},
            {"name": "l1", "in": dim_latent, "out": dim_logic},
            {"name": "l2", "in": dim_logic, "out": vocab_size}
        ]
    }
    writer.write_metadata("config", json.dumps(metadata))
    
    # 2. Guardar Pesos Empaquetados (2-bits)
    # Accedemos al buffer de bytes crudos expuesto por la clase de Rust
    for i, layer in enumerate(layers):
        writer.write_tensor_compressed(f"layer.{i}.packed_weights", layer.packed_weights)
        
    print(f"✨ Modelo SMG-1 guardado exitosamente.")

def main():
    parser = argparse.ArgumentParser(description="🧬 GAJE: SMG-1 Distillation Bridge")
    parser.add_argument("--teacher", type=str, default="models/gguf/smollm2-135m-f16.gguf")
    parser.add_argument("--dataset", type=str, default="data/datasets/dataset_born_2000.txt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="models/checkpoints/smg1_student.gaje")
    args = parser.parse_args()

    print(f"🧬 Iniciando PUENTE DE DESTILACIÓN (Teacher -> SMG-1)")
    print(f"[*] Maestro (Teacher): {args.teacher}")

    # 1. Cargar Maestro (SmolLM2)
    teacher = GenomicLLM(args.teacher)
    vocab_size = teacher.tokenizer.get_vocab_size() if hasattr(teacher.tokenizer, "get_vocab_size") else len(teacher.tokenizer)
    
    # 2. Inicializar Estudiante SMG-1 (Nativo en Rust)
    # Usamos las dimensiones del SMG-1 estandarizado
    dim_latent = 256
    dim_logic = 128
    centroides = [-1.5, -0.5, 0.5, 1.5]
    
    print(f"[*] Inicializando Estudiante SMG-1 (3 Capas, Vocab: {vocab_size})...")
    l0 = engine.GajeNeuromorphicLayer(dim_latent, vocab_size, 0.4, 0.8)
    l1 = engine.GajeNeuromorphicLayer(dim_logic, dim_latent, 0.4, 0.8)
    l2 = engine.GajeNeuromorphicLayer(vocab_size, dim_logic, 0.4, 0.8)
    student_layers = [l0, l1, l2]

    # 3. Cargar Dataset
    with open(args.dataset, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if len(line.strip()) > 10]

    print(f"[*] Dataset cargado: {len(lines)} líneas.")
    start_time = time.time()

    # 4. Bucle de Destilación
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        hits = 0
        tokens_processed = 0
        
        # Mezclar dataset
        np.random.shuffle(lines)
        
        # Procesar todo el dataset (Overnight Mode)
        for text_idx, text in enumerate(lines):
            tokens = teacher.tokenizer.encode(text, add_special_tokens=False)
            if hasattr(tokens, "ids"): tokens = tokens.ids
            
            if len(tokens) < 2: continue
            
            for t in range(len(tokens) - 1):
                input_id = tokens[t]
                target_id = tokens[t+1]
                
                # 1. Reforzar asociación en el Estudiante
                # Capa 0
                l0_deltas = [0.0] * dim_latent
                offset = (input_id * 16) % dim_latent
                for i in range(16):
                    l0_deltas[(offset + i) % dim_latent] = 1.0
                student_layers[0].refine_step(input_id, l0_deltas, 1.0)
                
                # Simular Spikes L0
                student_layers[0].integrate_batch(input_id, centroides, 1.0)
                s0 = student_layers[0].check_spikes()
                
                if not s0: continue
                
                # Capa 1
                l1_deltas = [0.0] * dim_logic
                offset_l1 = (input_id * 8) % dim_logic
                for i in range(8):
                    l1_deltas[(offset_l1 + i) % dim_logic] = 1.0
                for idx, _, _ in s0:
                    student_layers[1].refine_step(idx, l1_deltas, 1.0)
                
                # Simular Spikes L1
                for idx, intensity, _ in s0:
                    student_layers[1].integrate_batch(idx, centroides, intensity)
                s1 = student_layers[1].check_spikes()
                
                if not s1: continue

                # Capa 2 (Output)
                l2_deltas = [0.0] * vocab_size
                l2_deltas[target_id] = 1.0
                
                for idx, _, _ in s1:
                    student_layers[2].refine_step(idx, l2_deltas, args.lr)
                
                # Verificación de acierto
                for idx, intensity, _ in s1:
                    student_layers[2].integrate_batch(idx, centroides, intensity)
                s2 = student_layers[2].check_spikes()
                
                if any(spike[0] == target_id for spike in s2):
                    hits += 1
                tokens_processed += 1
                
                # Homeostasis
                for layer in student_layers:
                    layer.apply_homeostasis(2.0)
            
            if (text_idx + 1) % 1000 == 0:
                print(f"    [~] Progreso Época {epoch}: {text_idx + 1}/{len(lines)} líneas...", flush=True)

        if tokens_processed > 0:
            accuracy = (hits / tokens_processed) * 100
            print(f"🔥 Época {epoch}/{args.epochs} | Precisión: {accuracy:.2f}% | Tiempo Transcurrido: {time.time()-start_time:.1f}s", flush=True)

    print("\n✅ DESTILACIÓN COMPLETADA")
    
    # Guardar modelo final
    save_smg1_model(
        args.output, 
        student_layers, 
        vocab_size, 
        dim_latent, 
        dim_logic, 
        centroides
    )

if __name__ == "__main__":
    main()
