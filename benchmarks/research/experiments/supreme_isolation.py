import os
import sys
import numpy as np
import time

# Añadir el directorio python al path
sys.path.append(os.path.abspath("python"))

from genomize_llm import GenomicLLM

def calculate_top_prediction(model, text, disable_attn=False, disable_ffn=False):
    tokens = model.tokenizer.encode(text, add_special_tokens=False)
    last_id = tokens[-1]
    
    x = model.embedding_matrix[last_id].copy()
    for block in model.blocks:
        # Residual connections with optional component disabling
        x_in = x.copy()
        
        # 1. Atención
        if not disable_attn:
            x_norm = block.rms_norm(x_in, block.layers.get('attn_norm'))
            attn_out = block.attn.forward(x_norm.tolist(), len(tokens)-1)
            x = x_in + attn_out
        
        # 2. FFN
        x_mid = x.copy()
        if not disable_ffn:
            x_norm = block.rms_norm(x_mid, block.layers.get('ffn_norm'))
            gate = block.layers['ffn_gate'].forward(x_norm)
            up = block.layers['ffn_up'].forward(x_norm)
            ffn_hidden = block.silu(gate) * up
            ffn_out = block.layers['ffn_down'].forward(ffn_hidden)
            
            # LOGGING
            print(f"      DEBUG FFN: x_norm={np.linalg.norm(x_norm):.4f}, gate_max={np.max(gate):.4f}, up_max={np.max(up):.4f}, out_norm={np.linalg.norm(ffn_out):.4f}")
            
            x = x_mid + ffn_out

    x = model.rms_norm(x, model.output_norm_weight)
    logits = np.dot(model.embedding_matrix, x)
    top_id = np.argmax(logits)
    return model.tokenizer.decode([top_id])

def run_supreme_isolation():
    model_path = "/data/data/com.termux/files/home/models/gguf/qwen2-0_5b-q8_0.gguf"
    test_phrase = "Paris is the capital of"
    
    print("[*] Cargando Maestro F32 (1 bloque para aislar)...")
    model = GenomicLLM(model_path, num_blocks=1, mode='f32')
    
    print(f"\n📝 Frase: '{test_phrase}'")
    
    # Caso 1: Identidad Total
    res = calculate_top_prediction(model, test_phrase, disable_attn=True, disable_ffn=True)
    print(f"   [1] Identidad (Sin Attn, Sin FFN): '{res}'")
    
    # Caso 2: Sólo Atención
    res = calculate_top_prediction(model, test_phrase, disable_attn=False, disable_ffn=True)
    print(f"   [2] Sólo Atención:                 '{res}'")
    
    # Caso 3: Sólo FFN
    res = calculate_top_prediction(model, test_phrase, disable_attn=True, disable_ffn=False)
    print(f"   [3] Sólo FFN:                      '{res}'")
    
    # Caso 4: Bloque Completo
    res = calculate_top_prediction(model, test_phrase, disable_attn=False, disable_ffn=False)
    print(f"   [4] Bloque Completo:               '{res}'")

if __name__ == "__main__":
    run_supreme_isolation()
