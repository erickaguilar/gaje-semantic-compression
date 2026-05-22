import os
import sys
import json

# Asegurar que usamos el código local de 'python/'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from gaje.core import _impl as dna_semantic_compression

def check_tensor_sizes(path):
    if not os.path.exists(path):
        print(f"File {path} not found")
        return
    
    db_reader = dna_semantic_compression.GajeDatabaseReader(path)
    meta_str = db_reader.read_metadata("config")
    meta = json.loads(meta_str)
    n_embd = meta["n_embd"]
    
    for tensor_name in ["token_embd.dna", "token_embd.anchors", "lm_head.dna", "lm_head.anchors"]:
        if db_reader.has_tensor(tensor_name):
            data = db_reader.read_tensor(tensor_name)
            print(f"{tensor_name}: {len(data)} bytes")
            if "anchors" in tensor_name:
                num_elements = len(data) // 2
                print(f"  {num_elements} elements")
                if n_embd > 0:
                    print(f"  {num_elements / n_embd} rows")

if __name__ == "__main__":
    check_tensor_sizes("models/checkpoints/smollm2_f16_distilled.gaje")
