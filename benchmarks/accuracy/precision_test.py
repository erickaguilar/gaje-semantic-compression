import os
import sys
import numpy as np
import time

# Añadir el directorio python al path
sys.path.append(os.path.abspath("python"))

from gaje.nn.genomize import GenomicLLM

def calculate_ppl_and_show_predictions(model, phrases):
    print(f"\n{'='*20} Evaluando Coherencia Base {'='*20}")
    
    for text in phrases:
        tokens = model.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 1: continue
        
        # Tomamos el último token para ver qué predice a continuación
        last_id = tokens[-1]
        
        # Forward pass (Identity-like for single token to see next-token prediction)
        x = model.embedding_matrix[last_id].copy()
        for i, block in enumerate(model.blocks):
            #Residual Connection
            x_in = x.copy()
            x_norm = block.rms_norm(x_in, block.layers.get('attn_norm'))
            attn_out = block.attn.forward(x_norm.tolist(), len(tokens) - 1)
            x = x_in + attn_out
            
            # FFN DESACTIVADO PARA ESTE TEST
            # x_norm = block.rms_norm(x, block.layers.get('ffn_norm'))
            # ...
            
        x = model.rms_norm(x, model.output_norm_weight)
        logits = np.dot(model.embedding_matrix, x)
        
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
    print("🎯 TEST DE PRECISIÓN: BAJA ENTROPÍA (F32 TEACHER)")
    print("="*60)

    # Probamos con el Maestro F32 para asegurar que la arquitectura base funciona
    print("[*] Cargando Maestro F32 (24 bloques)...")
    model = GenomicLLM(model_path, num_blocks=24, mode='f32')
    
    calculate_ppl_and_show_predictions(model, low_entropy_phrases)

if __name__ == "__main__":
    run_precision_test()
