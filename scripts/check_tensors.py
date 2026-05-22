import gaje.core._impl as dna_semantic_compression
import sys

def check_tensors(path):
    try:
        reader = dna_semantic_compression.GajeDatabaseReader(path)
        for key in ["token_embd.dna", "token_embd.anchors", "token_embd.centroids"]:
            if reader.has_tensor(key):
                data = reader.read_tensor(key)
                print(f"{key}: {len(data)} bytes")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_tensors(sys.argv[1] if len(sys.argv) > 1 else "models/checkpoints/test_organism.gaje")
