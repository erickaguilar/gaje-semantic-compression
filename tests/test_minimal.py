import numpy as np
import sys
import os
sys.path.append(os.path.abspath("python"))
from gaje.core import _impl as dna_semantic_compression

def test_minimal_consistency():
    print("🧪 Testing Minimal Rust Kernel Consistency (1 neuron, 32 dims)")
    in_features = 32
    out_features = 1
    block_size = 32
    
    # 1. Weights: Only first dimension active with value 1.0
    w_orig = np.zeros((out_features, in_features), dtype=np.float32)
    w_orig[0, 0] = 1.0
    x = np.zeros(in_features, dtype=np.float32)
    x[0] = 1.0
    
    # 2. Quantize: 1.0 should result in bits 0b10 (match else in quantize_embedding)
    # thresholds = [0.1, 0.2, 0.3] -> 1.0 > 0.3 -> bits 10
    t = [0.1, 0.2, 0.3]
    dna = dna_semantic_compression.quantize_embedding(w_orig[0].tolist(), t)
    
    # Verify DNA bits
    first_byte = dna[0]
    bits = (first_byte >> 6) & 0b11
    print(f"[*] First token bits: {bits:02b}")
    
    # 3. Centroids: Bits 0b10 should map to c[3] (per lib.rs match)
    # We set c[3] = 42.0
    c = [1.0, 2.0, 3.0, 42.0]
    
    # 4. Setup GenomicLinear
    linear = dna_semantic_compression.GenomicLinear(
        dna,           # database
        [],            # anchors (F32)
        c,             # centroids
        out_features, 
        in_features, 
        block_size
    )
    
    # 5. Rust Forward
    y_rust = linear.forward(x)[0]
    
    # 6. Python Reference (Manual Dequantize)
    w_deq = np.array(dna_semantic_compression.dequantize_embedding(dna, in_features, c))
    y_py = np.dot(w_deq, x)
    
    print(f"[*] Rust output: {y_rust:.6f}")
    print(f"[*] Py output:   {y_py:.6f}")
    
    # Check if Rust maps 10 to c[3]
    if y_rust == 42.0:
        print("✅ Rust maps bits 10 to c[3]")
    elif y_rust == 3.0:
         print("❌ Rust maps bits 10 to c[2]")
    else:
         print(f"❌ Rust maps bits 10 to something else: {y_rust}")

    if y_py == 42.0:
        print("✅ Python maps bits 10 to c[3]")
    elif y_py == 3.0:
         print("❌ Python maps bits 10 to c[2]")
    else:
         print(f"❌ Python maps bits 10 to something else: {y_py}")

if __name__ == "__main__":
    test_minimal_consistency()
