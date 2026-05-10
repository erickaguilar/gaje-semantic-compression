import numpy as np
import sys
import os
sys.path.append(os.path.abspath("python"))
from stabilized_genomic_llm import GenomicLayer
from gaje.core import _impl as dna_semantic_compression

def test_layer_accuracy():
    print("🧪 Testing GenomicLayer Reconstruction Accuracy")
    # Small random weight matrix (out=64, in=128)
    in_features = 128
    out_features = 64
    w_orig = np.random.normal(0, 0.02, (out_features, in_features)).astype(np.float32)
    
    # Add some outliers (Anchors)
    w_orig[0, 0] = 0.5
    w_orig[10, 10] = -0.5
    
    layer = GenomicLayer("test", w_orig, block_size=32)
    
    # Reconstruct row 0
    w_rec_row0 = layer.get_row(0)
    
    # Calculate Cosine Similarity
    def cos_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    
    sim = cos_sim(w_orig[0], w_rec_row0)
    print(f"[*] Row 0 Cosine Similarity: {sim:.4f}")
    
    # Compare with a row that has no outliers
    w_rec_row1 = layer.get_row(1)
    sim1 = cos_sim(w_orig[1], w_rec_row1)
    print(f"[*] Row 1 Cosine Similarity: {sim1:.4f}")
    
    if sim > 0.90:
        print("✅ SUCCESS: Reconstruction is accurate.")
    else:
        print("❌ FAILURE: Reconstruction is poor.")

if __name__ == "__main__":
    test_layer_accuracy()
