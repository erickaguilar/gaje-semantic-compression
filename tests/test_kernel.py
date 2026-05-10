import numpy as np
import sys
import os
sys.path.append(os.path.abspath("python"))
from stabilized_genomic_llm import GenomicLayer
from gaje.core import _impl as dna_semantic_compression

def test_kernel_consistency():
    print("🧪 Testing Rust Kernel Consistency vs np.dot")
    in_features = 128
    out_features = 64
    w_orig = np.random.normal(0, 0.02, (out_features, in_features)).astype(np.float32)
    
    layer = GenomicLayer("test", w_orig, block_size=32)
    x = np.random.normal(0, 1.0, (in_features,)).astype(np.float32)
    
    # 1. Forward using Rust Kernel
    y_rust = layer.forward(x)
    
    # 2. Forward using np.dot with dequantized weights
    w_deq = np.zeros_like(w_orig)
    for i in range(out_features):
        w_deq[i] = layer.get_row(i)
    
    y_dot = np.dot(w_deq, x)
    
    # Compare
    diff = np.abs(y_rust - y_dot).max()
    print(f"[*] Max Difference: {diff:.6e}")
    
    if diff < 1e-4:
        print("✅ SUCCESS: Rust Kernel is consistent with np.dot.")
    else:
        print("❌ FAILURE: Rust Kernel is inconsistent.")

if __name__ == "__main__":
    test_kernel_consistency()
