import gguf
import numpy as np
from gaje.core import _impl as dna_semantic_compression
import os
from transformers import AutoTokenizer

def dequantize_variant_a(tensor):
    # Asumimos que el primer eje es el bloqueado (in_features)
    in_f = tensor.shape[0]
    out_f = tensor.shape[1]
    flat = dna_semantic_compression.dequantize_q8_0_native(tensor.data.tobytes(), out_f, in_f)
    return np.array(flat).reshape(out_f, in_f)

def dequantize_variant_b(tensor):
    # Asumimos que el segundo eje es el bloqueado (in_features)
    out_f = tensor.shape[0]
    in_f = tensor.shape[1]
    flat = dna_semantic_compression.dequantize_q8_0_native(tensor.data.tobytes(), out_f, in_f)
    return np.array(flat).reshape(out_f, in_f).T

def diagnostic_test():
    model_path = "/data/data/com.termux/files/home/models/qwen2-0_5b-q8_0.gguf"
    reader = gguf.GGUFReader(model_path)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    
    emb_tensor = next(t for t in reader.tensors if "token_embd.weight" in t.name)
    norm_w = next(t for t in reader.tensors if "output_norm.weight" in t.name).data.astype(np.float32)
    
    prompt = "Paris is the capital of"
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    last_id = tokens[-1]
    
    for name, variant_fn in [("Variant A (In=shape[0])", dequantize_variant_a), 
                             ("Variant B (In=shape[1])", dequantize_variant_b)]:
        print(f"\n--- Probando {name} ---")
        try:
            emb_matrix = variant_fn(emb_tensor)
            x = emb_matrix[last_id].copy()
            
            # RMSNorm final
            rms = np.sqrt(np.mean(x**2) + 1e-6)
            x = (x / rms) * norm_w
            
            # Logits
            logits = np.dot(emb_matrix, x)
            top_id = np.argmax(logits)
            print(f"   Predicción: '{tokenizer.decode([top_id])}' (ID: {top_id})")
        except Exception as e:
            print(f"   Error: {e}")

if __name__ == "__main__":
    diagnostic_test()
