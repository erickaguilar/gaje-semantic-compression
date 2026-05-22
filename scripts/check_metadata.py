import os
import sys
import json

# Asegurar que usamos el código local de 'python/'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from gaje.core import _impl as dna_semantic_compression

def check_metadata(path):
    if not os.path.exists(path):
        print(f"File {path} not found")
        return
    
    db_reader = dna_semantic_compression.GajeDatabaseReader(path)
    meta_str = db_reader.read_metadata("config")
    meta = json.loads(meta_str)
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    check_metadata("models/checkpoints/smollm2_f16_distilled.gaje")
