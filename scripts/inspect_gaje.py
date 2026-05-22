import json
import gaje.core._impl as dna_semantic_compression
import sys

def inspect_gaje(path):
    try:
        reader = dna_semantic_compression.GajeDatabaseReader(path)
        config_str = reader.read_metadata("config")
        config = json.loads(config_str)
        print(f"--- Metadata for {path} ---")
        print(json.dumps(config, indent=2))
        
        # Check some tensor shapes
        # For GenomicLinear, we can't directly get shapes, but we can check if tensors exist
        # and maybe infer something from their sizes.
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_gaje(sys.argv[1])
    else:
        inspect_gaje("models/checkpoints/test_organism.gaje")
