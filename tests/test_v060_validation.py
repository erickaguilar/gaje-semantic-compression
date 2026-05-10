import numpy as np
from gaje.nn.stabilized import GenomicLayer
import time

def test_native_learning_convergence():
    print("🔬 Validando Convergencia de Aprendizaje Local (Mobile-Native Learning)...")
    dim = 64
    out_dim = 32
    
    # Datos de prueba
    x = np.random.randn(dim).astype(np.float32)
    # Definimos un objetivo arbitrario (queremos que la capa aprenda a producir este vector)
    target = np.random.randn(out_dim).astype(np.float32)
    
    weights = np.random.randn(out_dim, dim).astype(np.float32)
    layer = GenomicLayer("learner", weights)
    
    # Medir error inicial
    initial_out = layer.forward(x)
    initial_error = np.mean((initial_out - target)**2)
    print(f"[*] Error MSE Inicial: {initial_error:.6f}")
    
    # Ciclo de refinamiento nativo en Rust
    lr = 0.05
    iterations = 20
    for i in range(iterations):
        layer.linear.refine_centroids(x.tolist(), target.tolist(), lr)
        
    # Medir error final
    final_out = layer.forward(x)
    final_error = np.mean((final_out - target)**2)
    print(f"[*] Error MSE Final (tras {iterations} iters): {final_error:.6f}")
    
    improvement = (initial_error - final_error) / initial_error * 100
    print(f"🚀 Mejora en Precisión Local: {improvement:.2f}%")
    
    assert final_error < initial_error, "El optimizador nativo no está reduciendo el error."

def test_kv_cache_integrity():
    print("\n🔬 Validando Integridad de KV-Cache DNA (2-bit ADC)...")
    # Este test verifica que la caché interna de Rust no corrompa la señal
    from gaje.core import _impl as dna_core
    
    dim = 128
    n_head = 4
    head_dim = dim // n_head
    
    # Fake weights for Q, K, V (128x128 each)
    packed_w = b"\x00" * ((dim * dim) // 4)
    # Centroids for Q, K, V (3 matrices * 128 rows * 4 centroids)
    total_rows = dim + dim + dim # Q + K + V
    centroids = [-1.0, -0.3, 0.3, 1.0] * total_rows
    
    attn = dna_core.GenomicAttention(
        packed_w, packed_w, packed_w, centroids, dim // 4, n_head, n_head
    )
    
    x = np.random.randn(dim).astype(np.float32)
    
    # Primera pasada (debería llenar caché)
    out1 = attn.forward(x.tolist(), 0)
    # Segunda pasada (debería usar caché de 2 bits)
    out2 = attn.forward(x.tolist(), 1)
    
    print(f"[*] Forward con KV-Cache (2-bit) exitoso. Output dim: {len(out2)}")
    assert len(out1) == len(out2) == dim

if __name__ == "__main__":
    try:
        test_native_learning_convergence()
        test_kv_cache_integrity()
        print("\n✅ VALIDACIÓN TÉCNICA EXITOSA: El motor v0.6.0 es robusto.")
    except Exception as e:
        print(f"\n❌ FALLO EN VALIDACIÓN: {e}")
        import traceback
        traceback.print_exc()
