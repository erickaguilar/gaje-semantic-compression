import os
import sys
import numpy as np
import gguf

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

import gaje.core._impl as d

def verify_quantization(gguf_path, gaje_path):
    print(f"🔬 [VERIFICACIÓN] GGUF vs GAJE")
    
    reader_gguf = gguf.GGUFReader(gguf_path)
    reader_gaje = d.GajeDatabaseReader(gaje_path)
    
    layer_name = "blk.0.attn_q.weight"
    gaje_prefix = "blk.0.attn_q"
    
    # GGUF
    tensor_gguf = next(t for t in reader_gguf.tensors if t.name == layer_name)
    w_orig = tensor_gguf.data.astype(np.float32)
    
    # GAJE
    dna = reader_gaje.read_tensor(f"{gaje_prefix}.dna")
    centroids = np.frombuffer(reader_gaje.read_tensor(f"{gaje_prefix}.centroids"), dtype=np.float32)
    
    # Dequantize GAJE (manual 4-bit)
    n_elements = w_orig.size
    block_size = 32
    n_blocks = n_elements // block_size
    
    w_recon = np.zeros(n_elements, dtype=np.float32)
    for b in range(n_blocks):
        block_dna = dna[b * 16 : (b+1) * 16] # 4-bit -> 16 bytes per block of 32
        block_centroids = centroids[b * 16 : (b+1) * 16]
        
        for k in range(16):
            byte = block_dna[k]
            w_recon[b * 32 + k * 2] = block_centroids[byte >> 4]
            w_recon[b * 32 + k * 2 + 1] = block_centroids[byte & 0x0F]
            
    # MSE
    mse = np.mean((w_orig.flatten() - w_recon)**2)
    cos_sim = np.dot(w_orig.flatten(), w_recon) / (np.linalg.norm(w_orig) * np.linalg.norm(w_recon))
    
    print(f"Layer: {layer_name}")
    print(f"  MSE: {mse:.8f}")
    print(f"  Cos Sim: {cos_sim:.8f}")
    
    # Primeros 10 valores
    print("\nOriginal vs Reconstructed (first 10):")
    for i in range(10):
        print(f"  [{i}] {w_orig.flatten()[i]:>8.4f} | {w_recon[i]:>8.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=str, default="models/gguf/smollm2-135m-f16.gguf")
    parser.add_argument("--gaje", type=str, default="models/production/smollm2_mixed_v1.gaje")
    args = parser.parse_args()
    verify_quantization(args.gguf, args.gaje)
