import json
import sys
import os

# Asegurar que el path del proyecto este en PYTHONPATH
sys.path.insert(0, os.path.abspath("python"))

import gaje.core._impl as dna_semantic_compression
from gaje.nn import constants as C

def inspect_model(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    db_reader = dna_semantic_compression.GajeDatabaseReader(path)
    meta_str = db_reader.read_metadata(C.META_KEY_CONFIG)
    meta = json.loads(meta_str)
    
    n_embd = meta.get(C.META_KEY_N_EMBD, 576)
    n_head = meta.get(C.META_KEY_N_HEAD, 9)
    n_blocks = meta.get(C.META_KEY_N_BLOCKS, 30)
    
    # Suponiendo que ffn_hidden se puede deducir del tamaño de un tensor
    # blk.0.ffn_gate.centroids -> out_features * (in_features / 32) * 4 * 4 bytes
    # Pero es más fácil ver los out_features de la capa linear
    
    print(f"Architecture: {meta['config']['name']}")
    print(f"n_embd: {n_embd}")
    print(f"n_blocks: {n_blocks}")
    
    # Cálculo de parámetros por bloque
    # Atención: Q, K, V, O -> (embd * embd) * 4
    attn_params = (n_embd * n_embd) * 4
    
    # FFN: Gate, Up, Down -> 3 capas
    # Necesitamos ffn_hidden. Para SmolLM es 1536.
    # Vamos a intentar detectarlo de la BD.
    
    ffn_hidden = 1536 # Default fallback
    if db_reader.has_tensor("blk.0.ffn_gate.centroids"):
        c_bytes = db_reader.read_tensor("blk.0.ffn_gate.centroids")
        c_count = len(c_bytes) // 4
        # count = out_features * (in_features // 32) * 4
        ffn_hidden = c_count // ((n_embd // 32) * 4)
        
    print(f"ffn_hidden: {ffn_hidden}")
    
    ffn_params = (n_embd * ffn_hidden) * 3
    
    total_attn = attn_params * n_blocks
    total_ffn = ffn_params * n_blocks
    
    print("-" * 30)
    print(f"DISTRIBUCIÓN DE PARÁMETROS (por bloque):")
    print(f"  - Atención: {attn_params:,}")
    print(f"  - FFN:      {ffn_params:,}")
    print("-" * 30)
    print(f"TOTAL MODELO ({n_blocks} bloques):")
    print(f"  - Capas de Atención: {total_attn:,} params")
    print(f"  - Capas FFN:         {total_ffn:,} params")
    print(f"  - Ratio Attn/FFN:    {total_attn / total_ffn:.2f}")

if __name__ == "__main__":
    inspect_model("models/production/silver_adult_sovereign.gaje")
