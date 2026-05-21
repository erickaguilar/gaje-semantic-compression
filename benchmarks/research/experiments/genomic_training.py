import os
import sys
import numpy as np
import time

# Añadir el directorio python al path
sys.path.append(os.path.abspath("python"))

from transformers import AutoTokenizer
from stabilized_genomic_llm import GenomicLLM

def train_metabolism():
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model_path = "/data/data/com.termux/files/home/models/smollm2-135m-q8_0.gguf"
    
    print("🧬 INICIANDO MICRO-ENTRENAMIENTO GENÓMICO")
    print("="*50)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    student = GenomicLLM(model_path, num_blocks=30)
    
    # Dataset de calibración (Hechos críticos para el modelo)
    calibration_data = [
        "The capital of France is Paris.",
        "The sun rises in the east.",
        "Python is a programming language.",
        "Water boils at one hundred degrees."
    ]
    
    epochs = 3
    learning_rate = 0.05
    
    for epoch in range(epochs):
        print(f"\n🌱 Época {epoch+1}/{epochs}")
        total_loss = 0
        
        for text in calibration_data:
            tokens = tokenizer.encode(text)
            target_token = tokens[-1]
            input_tokens = tokens[:-1]
            
            # Forward pass
            logits_all = student.forward(input_tokens)
            logits = logits_all[-1]
            
            # Softmax
            probs = np.exp(logits - np.max(logits))
            probs /= probs.sum()
            
            prob_target = probs[target_token]
            loss = -np.log(max(prob_target, 1e-10))
            total_loss += loss
            
            # --- AJUSTE METABÓLICO (Backprop Genómico Simulado) ---
            # Si el modelo falla, "estiramos" los centroides epigenéticos
            # de la última capa (lm_head) para capturar mejor el outlier.
            if prob_target < 0.1:
                # Ajustamos los centroides de la capa de salida
                current_centroids = np.array(student.lm_head.linear.epigenetic_centroids)
                # Gradiente simulado: movemos los centroides externos hacia afuera
                # para aumentar el rango dinámico de la corrección.
                adjustment = learning_rate * (1.0 - prob_target)
                current_centroids[0] -= adjustment # Más negativo
                current_centroids[3] += adjustment # Más positivo
                
                student.lm_head.linear.epigenetic_centroids = current_centroids.tolist()
        
        avg_loss = total_loss / len(calibration_data)
        print(f"   [~] Loss promedio: {avg_loss:.4f}")

    print("\n" + "="*50)
    print("🔥 EVOLUCIÓN COMPLETADA")
    print("="*50)
    
    # Test final con el hecho clave
    test_text = "The capital of France is"
    test_tokens = tokenizer.encode(test_text)
    final_logits = student.forward(test_tokens)[-1]
    final_probs = np.exp(final_logits - np.max(final_logits))
    final_probs /= final_probs.sum()
    
    paris_token = tokenizer.encode(" Paris")[0]
    print(f"🎯 Probabilidad final de ' Paris': {final_probs[paris_token]:.6f}")
    
    target_prob_initial = 0.000036
    if final_probs[paris_token] > target_prob_initial:
        improvement = final_probs[paris_token] / target_prob_initial
        print(f"✅ Mejora semántica: {improvement:.2f}x")
    
if __name__ == "__main__":
    train_metabolism()
