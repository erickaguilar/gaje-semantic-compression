import numpy as np
from gaje.core import _impl as dna_core
import struct

def test_dgi_native_f32():
    print("🔬 Validando DGI Native (Float32)...")
    dim = 128
    # Create random F32 data
    data = np.random.randn(dim).astype(np.float32)
    data_bytes = data.tobytes()
    
    # Process with Rust
    dna, centroids = dna_core.genomize_f32_native(data_bytes, 32)
    
    print(f"[*] DNA strands: {len(dna)} bytes")
    print(f"[*] Centroids: {len(centroids) // 4} blocks")
    
    assert len(dna) == dim // 4
    assert len(centroids) == (dim // 32) * 4
    print("✅ DGI F32 Validated.")

def test_dgi_native_f16():
    print("\n🔬 Validando DGI Native (Float16)...")
    dim = 128
    # Create random F16 data
    data = np.random.randn(dim).astype(np.float16)
    data_bytes = data.tobytes()
    
    # Process with Rust
    dna, centroids = dna_core.genomize_f16_native(data_bytes, 32)
    
    assert len(dna) == dim // 4
    assert len(centroids) == (dim // 32) * 4
    print("✅ DGI F16 Validated.")

if __name__ == "__main__":
    try:
        test_dgi_native_f32()
        test_dgi_native_f16()
        print("\n🚀 PUENTE DE ALTA FIDELIDAD (DGI) OPERATIVO.")
    except Exception as e:
        print(f"❌ Error en DGI: {e}")
        import traceback
        traceback.print_exc()
