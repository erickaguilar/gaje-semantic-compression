import os
import sys
import numpy as np

# Ensure we use the local package
sys.path.insert(0, os.path.abspath("python"))

from gaje.core import _impl as core

def inspect_db(path):
    print(f"🔍 Inspecting GAJE DB: {path}")
    reader = core.GajeDatabaseReader(path)
    
    # Check config
    config_str = reader.read_metadata("config")
    print(f"[*] Metadata Config: {config_str}")
    
    # Check layers
    layers = ["token_embd", "lm_head", "blk.0.attn_q", "blk.1.ffn_up"]
    for layer in layers:
        for suffix in [".dna", ".centroids", ".anchors"]:
            key = f"{layer}{suffix}"
            if reader.has_tensor(key):
                data = reader.read_tensor(key)
                print(f"  - {key:20} | Size: {len(data):10} bytes")
                if suffix == ".centroids":
                    arr = np.frombuffer(data, dtype=np.float32)
                    print(f"    - First 5 centroids: {arr[:5].tolist()}")
            else:
                print(f"  - {key:20} | NOT FOUND")

if __name__ == "__main__":
    inspect_db("models/born_genomic_qwen/model.gaje")
