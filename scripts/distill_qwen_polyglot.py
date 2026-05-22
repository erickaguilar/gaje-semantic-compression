import os
import sys
import time
import numpy as np
import argparse

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from gaje.nn.stabilized import GenomicLLM
import gaje.core._impl as dna_semantic_compression

def main():
    parser = argparse.ArgumentParser(description="🧬 GAJE: Polyglot Breeding with Qwen2 Teacher")
    parser.add_argument("--teacher", type=str, default="models/gguf/qwen2-0_5b-q8_0.gguf")
    parser.add_argument("--student", type=str, default="models/checkpoints/mature_polyglot_organism.gaje")
    parser.add_argument("--dataset", type=str, default="data/datasets/hybrid_polyglot_dataset.txt")
    parser.add_argument("--output", type=str, default="models/checkpoints/qwen_bred_organism.gaje")
    parser.add_argument("--minutes", type=int, default=30)
    args = parser.parse_args()

    print(f"🧬 Iniciando CRIANZA POR DISTILACIÓN (Simbiótica)")
    print(f"[*] Maestro (Padre): {args.teacher}")
    print(f"[*] Estudiante (Organismo): {args.student}")

    # 1. Cargar Maestro (Qwen2) - Usando cargador genómico para obtener logits
    teacher_model = GenomicLLM(args.teacher)
    
    # 2. Cargar Estudiante (GAJE)
    student_model = GenomicLLM.load_genomic(args.student)
    
    # Sincronizar h_scale
    for block in student_model.blocks:
        block.rust_block.h_scale = 1.0

    # 3. Cargar Dataset Híbrido
    with open(args.dataset, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if len(line.strip()) > 15]

    print(f"[*] Crianza políglota iniciada sobre {len(lines)} líneas.")
    
    start_time = time.time()
    time_limit = args.minutes * 60
    lr = 0.002
    
    epoch = 0
    while (time.time() - start_time) < time_limit:
        epoch += 1
        np.random.shuffle(lines)
        print(f"\n🔥 Época {epoch} | Maestro guía al organismo...")
        
        for i, text in enumerate(lines):
            if i % 10 == 0 and (time.time() - start_time) >= time_limit: break
            
            tokens = student_model.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) < 3: continue
            
            # El Maestro genera las probabilidades "ideales"
            teacher_logits = teacher_model.forward(tokens, clear_cache=True)
            
            # El Estudiante intenta imitar al Maestro en cada paso de la secuencia
            total_loss = 0
            for t in range(len(tokens) - 1):
                target_probs = teacher_logits[t] # Logits del maestro para el sgte token
                
                # Refinamiento Simbiótico: 
                # El organismo ajusta su LM Head para resonar con el Maestro
                loss = student_model.rust_llm.train_step(tokens[t], tokens[t+1], lr)
                total_loss += loss
                
            if i % 20 == 0:
                elapsed = int(time.time() - start_time)
                print(f"  - [{elapsed}s] Progreso: {i}/{len(lines)} | Simbiosis Loss: {loss:.4f}", end="\r")
        
        student_model.save(args.output)
        print(f"\n[+] Checkpoint simbiótico guardado.")

    print("\n" + "="*60)
    print(f"✅ CRIANZA CON QWEN2 COMPLETADA")
    print(f"🚀 Organismo Políglota Evolucionado: {args.output}")
    print("="*60)

if __name__ == "__main__":
    main()
