import os
import sys
import numpy as np
import gguf

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

import gaje.core._impl as d

def verify_embeddings(gguf_path, gaje_path):
    print(f"🔬 [VERIFICACIÓN] Embeddings")
    
    reader_gguf = gguf.GGUFReader(gguf_path)
    reader_gaje = d.GajeDatabaseReader(gaje_path)
    
    layer_name = "token_embd.weight"
    gaje_prefix = "token_embd"
    
    # GGUF
    tensor_gguf = next(t for t in reader_gguf.tensors if t.name == layer_name)
    w_orig_full = tensor_gguf.data.astype(np.float32)
    
    # ROW 1 (BOS)
    row_idx = 1
    w_orig_row = w_orig_full[row_idx]
    
    # GAJE read row via reader
    # We can use our manual dequantizer or the NativeLoader
    dna_full = reader_gaje.read_tensor(f"{gaje_prefix}.dna")
    centroids_full = np.frombuffer(reader_gaje.read_tensor(f"{gaje_prefix}.centroids"), dtype=np.float32)
    
    # Dequantize row 1
    n_embd = 576
    block_size = 32
    n_blocks_row = n_embd // block_size # 18
    
    # row_start in DNA bytes: row_idx * 18 * 16
    dna_row = dna_full[row_idx * 18 * 16 : (row_idx+1) * 18 * 16]
    # centroids_start: row_idx * 18 * 16
    centroids_row = centroids_full[row_idx * 18 * 16 : (row_idx+1) * 18 * 16]
    
    w_recon_row = np.zeros(n_embd, dtype=np.float32)
    for b in range(18):
        block_dna = dna_row[b * 16 : (b+1) * 16]
        block_centroids = centroids_row[b * 16 : (b+1) * 16]
        for k in range(16):
            byte = block_dna[k]
            w_recon_row[b * 32 + k * 2] = block_centroids[byte >> 4]
            w_recon_row[b * 32 + k * 2 + 1] = block_centroids[byte & 0x0F]
            
    # Cos Sim
    cos_sim = np.dot(w_orig_row, w_recon_row) / (np.linalg.norm(w_orig_row) * np.linalg.norm(w_recon_row))
    print(f"Row {row_idx} (BOS) Cos Sim: {cos_sim:.8f}")
    print(f"Original Row 1 (first 5): {w_orig_row[:5]}")
    print(f"Reconstructed Row 1 (first 5): {w_recon_row[:5]}")

verify_embeddings("models/gguf/smollm2-135m-f16.gguf", "models/production/smollm2_mixed_v1.gaje")
