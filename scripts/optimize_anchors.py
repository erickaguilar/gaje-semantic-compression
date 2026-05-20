import os
import sys
import numpy as np
import json

# Ensure we use the local package
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM
from gaje.core import _impl as core

def prune_anchors(model_path, output_path, threshold_sigma=1.0):
    print(f"🧬 GAJE OPTIMIZER: Anchor Pruning (Target Sigma: {threshold_sigma})")
    print("-" * 60)
    
    # Abrir DB directamente para mayor control
    reader = core.GajeDatabaseReader(model_path)
    writer = core.GajeDatabaseWriter(output_path)
    
    # Copiar metadata
    config_str = reader.read_metadata("config")
    writer.write_metadata("config", config_str)
    if reader.has_metadata("tokenizer"):
        writer.write_metadata("tokenizer", reader.read_metadata("tokenizer"))
    
    # Listar todos los tensores
    # Nota: No tenemos una función 'list_tensors' expuesta directamente de forma fácil,
    # pero conocemos la estructura por el script de guardado.
    
    # Obtener info del modelo para saber qué capas procesar
    meta = json.loads(config_str)
    n_blocks = meta["n_blocks"]
    
    layers = ["token_embd", "lm_head"]
    for i in range(n_blocks):
        p = f"blk.{i}."
        layers.extend([
            p + "attn_q", p + "attn_k", p + "attn_v", p + "attn_output",
            p + "ffn_gate", p + "ffn_up", p + "ffn_down"
        ])
    
    total_saved_bytes = 0
    total_original_bytes = 0

    for layer in layers:
        print(f"[*] Processing {layer}...")
        
        # Copiar DNA y Centroids (sin cambios)
        for suffix in [".dna", ".centroids", ".bias", ".ffn_norm", ".attn_norm"]:
            key = f"{layer}{suffix}" if suffix.startswith(".") else suffix
            if reader.has_tensor(key):
                writer.write_tensor(key, reader.read_tensor(key))
        
        # Pruning de Anchors
        anchor_key = f"{layer}.anchors"
        if reader.has_tensor(anchor_key):
            anchor_data = reader.read_tensor(anchor_key)
            anchors_f16 = np.frombuffer(anchor_data, dtype=np.float16)
            total_original_bytes += len(anchor_data)
            
            # Calcular umbral basado en la desviación estándar de los residuos
            # Si el residuo es pequeño, lo ponemos a cero
            abs_anchors = np.abs(anchors_f16.astype(np.float32))
            std = np.std(abs_anchors)
            threshold = threshold_sigma * std
            
            pruned_anchors = anchors_f16.copy()
            mask = abs_anchors < threshold
            pruned_anchors[mask] = 0
            
            # Estadísticas
            n_pruned = np.sum(mask)
            pct = (n_pruned / len(anchors_f16)) * 100
            print(f"    - Threshold: {threshold:.6f} | Pruned: {pct:.2f}%")
            
            new_data = pruned_anchors.tobytes()
            writer.write_tensor(anchor_key, new_data)
            total_saved_bytes += (n_pruned * 2) # f16 = 2 bytes

    print("-" * 60)
    print(f"✅ Pruning Finished!")
    print(f"   - Total Anchor Bytes Removed: {total_saved_bytes / (1024*1024):.2f} MB")
    print(f"   - Final Estimated Size Reduction: {(total_saved_bytes / os.path.getsize(model_path)) * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prune_gaje.py <input.gaje> <output.gaje> [sigma]")
    else:
        sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
        prune_anchors(sys.argv[1], sys.argv[2], sigma)
