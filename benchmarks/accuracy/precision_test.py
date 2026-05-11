import os
import sys
import numpy as np
import time

# Añadir el directorio python al path
sys.path.append(os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM

def calculate_ppl_and_show_predictions(model, phrases):
    print(f"\n{'='*20} Evaluando Coherencia Base {'='*20}")
    
    for text in phrases:
        tokens = model.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 1: continue
        
        # Forward pass
        logits = model.forward(tokens)[-1]
        
        # Softmax y Top-k
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        
        top_indices = np.argsort(probs)[::-1][:5]
        top_probs = probs[top_indices]
        top_tokens = [model.tokenizer.decode([idx]) for idx in top_indices]
        
        print(f"\n📝 Contexto: '{text}'")
        print(f"   🔍 Top 5 predicciones:")
        for i in range(5):
            print(f"      {i+1}. '{top_tokens[i]}' ({top_probs[i]:.4f})")

def run_precision_test():
    model_path = "/data/data/com.termux/files/home/models/qwen2-0_5b-q8_0.gguf"
    
    # Frases de baja entropía
    low_entropy_phrases = [
        "2 + 2 =",
        "Paris is the capital of",
        "The sun rises in the",
        "El ADN es la base de la"
    ]

    print("="*60)
    print("🎯 TEST DE PRECISIÓN: BAJA ENTROPÍA (GAJE 2-BIT)")
    print("="*60)

    # Cargamos el modelo genómico (limitamos bloques para velocidad si es necesario)
    print("[*] Cargando Organismo Genómico (4 bloques)...")
    model = GenomicLLM(model_path, num_blocks=4)
    
    calculate_ppl_and_show_predictions(model, low_entropy_phrases)

if __name__ == "__main__":
    run_precision_test()
