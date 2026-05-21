import os
import sys
import numpy as np
import time

# Añadir el directorio python al path
sys.path.append(os.path.abspath("python"))

from genomize_llm import GenomicLLM, GenomicLLMLayer, GenomicTransformerBlock
from gaje.core import _impl as dna_semantic_compression

def calculate_ppl_custom(model, text):
    tokens = model.tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < 2: return 0.0
    
    log_likelihoods = []
    print(f"\n[*] Debugging Predictions for: '{text}'")
    for i in range(1, len(tokens)):
        last_id = tokens[i-1]
        target_id = tokens[i]
        
        # Forward pass
        x = model.embedding_matrix[last_id].copy()
        for block in model.blocks:
            x = block.forward(x.tolist(), i-1)
            x = np.array(x)
            
        x = model.rms_norm(x, model.output_norm_weight)
        logits = np.dot(model.embedding_matrix, x)
        
        # Top 5
        top_ids = np.argsort(logits)[::-1][:5]
        top_tokens = [model.tokenizer.decode([tid]) for tid in top_ids]
        target_token = model.tokenizer.decode([target_id])
        
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        prob_target = probs[target_id]
        
        if i == 1:
            print(f"    Token {i} ('{model.tokenizer.decode([last_id])}'):")
            print(f"      Target: '{target_token}' (Prob: {prob_target:.8f})")
            print(f"      Top 5 predicted: {top_tokens}")
            print(f"      Logit Range: {np.min(logits):.2f} to {np.max(logits):.2f}")

        log_likelihoods.append(np.log(max(prob_target, 1e-10)))
        
    return np.exp(-np.mean(log_likelihoods))

def run_isolation_test():
    model_path = "/data/data/com.termux/files/home/models/gguf/qwen2-0_5b-q8_0.gguf"
    test_text = "El protocolo GAJE es eficiente."

    print("="*60)
    print("🔍 TEST DE AISLAMIENTO DE PERPLEXITY (PPL)")
    print("="*60)

    # Solo probamos Maestro F32 con 24 bloques para ver si la arquitectura es correcta
    print("\n[1] Probando Maestro F32 (24 bloques)...")
    teacher = GenomicLLM(model_path, num_blocks=24, mode='f32')
    ppl_f32 = calculate_ppl_custom(teacher, test_text)
    print(f"✅ PPL F32: {ppl_f32:.4f}")

if __name__ == "__main__":
    run_isolation_test()
