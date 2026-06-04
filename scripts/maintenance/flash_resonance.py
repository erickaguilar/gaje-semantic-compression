import os
import sys
import time
import numpy as np
import argparse

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM

def flash_resonance(model_path, data_path, output_path, lr=0.001, steps=50):
    print(f"🧬 RESONANCIA FLASH: Optimizando '{model_path}'")
    print(f"[*] Datos de sintonía: {data_path}")
    
    llm = GenomicLLM.load_genomic(model_path)
    tokenizer = llm.tokenizer
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Archivo de datos no encontrado.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read(5000) # Solo una pequeña muestra para sintonía rápida
        
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(tokens, "ids"): tokens = tokens.ids
    
    print(f"[*] Sintonizando {len(tokens)} tokens en {steps} pasos...")
    
    t0 = time.time()
    for step in range(steps):
        # Tomamos un segmento aleatorio
        start = np.random.randint(0, len(tokens) - 32)
        seq = tokens[start : start + 32]
        
        loss = llm.rust_llm.train_on_sequence(seq, lr)
        
        if (step + 1) % 10 == 0:
            print(f"    [Step {step+1}/{steps}] Loss: {loss:.4f}")
            
    print(f"[*] Guardando modelo afinado en: {output_path}")
    llm.save(output_path)
    print(f"✅ Sintonía completada en {time.time() - t0:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/production/silver_adult_calibrated.gaje")
    parser.add_argument("--data", type=str, default="data/datasets/specialized/tiny_shakespeare.txt")
    parser.add_argument("--output", type=str, default="models/production/silver_adult_tuned.gaje")
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    
    flash_resonance(args.model, args.data, args.output, args.lr, args.steps)
