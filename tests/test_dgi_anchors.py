import numpy as np
from gaje.core import _impl as dna_core
import struct

def test_dgi_native_f32_with_anchors():
    print("🔬 Validando DGI Native con Escudo (F32 + Anchors)...")
    dim = 128
    # Create random F32 data with some huge outliers
    data = np.random.normal(0, 0.1, dim).astype(np.float32)
    data[5] = 2.5 # Outlier
    data[50] = -3.0 # Outlier
    data_bytes = data.tobytes()
    
    # Process with Rust (Threshold 0.15)
    dna, centroids, anchors = dna_core.genomize_f32_native(data_bytes, 32, 0.15)
    
    print(f"[*] DNA strands: {len(dna)} bytes")
    print(f"[*] Anchors found: {np.count_nonzero(anchors)}")
    
    assert len(dna) == dim // 4
    assert np.count_nonzero(anchors) > 0
    assert abs(anchors[5]) > 0
    print("✅ DGI F32 + Anchors Validated.")

def test_dgi_native_f16_with_anchors():
    print("\n🔬 Validando DGI Native con Escudo (F16 + Anchors)...")
    dim = 128
    data = np.random.normal(0, 0.1, dim).astype(np.float16)
    data[10] = 5.0 # Outlier
    data_bytes = data.tobytes()
    
    # Process with Rust
    dna, centroids, anchors = dna_core.genomize_f16_native(data_bytes, 32, 0.15)
    
    assert len(dna) == dim // 4
    assert np.count_nonzero(anchors) > 0
    print("✅ DGI F16 + Anchors Validated.")

if __name__ == "__main__":
    try:
        test_dgi_native_f32_with_anchors()
        test_dgi_native_f16_with_anchors()
        print("\n🚀 DGI CON ESCUDO GENÓMICO (ANCHORS) OPERATIVO.")
    except Exception as e:
        print(f"❌ Error en DGI+Anchors: {e}")
        import traceback
        traceback.print_exc()
