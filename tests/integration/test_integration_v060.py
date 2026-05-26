import numpy as np
from gaje.core import _impl as core

def test_full_integration():
    print("🔬 Probando integración básica: GajeIndex (ADC) -> Batch Load -> Search")
    n, dim = 500, 64
    base_c = [-1.0, -0.3, 0.3, 1.0]
    
    # 1. Instanciación (Sincronizada con src/core/index.rs: new(dims, centroids))
    idx = core.GajeIndex(dim, base_c)
    
    # 2. Preparación de datos (Empaquetado 2-bit dummy: 4 dims por byte)
    # n registros, cada uno con dim/4 bytes
    packed_db = [bytes([0xAA] * (dim // 4)) for _ in range(n)] # 0xAA = 10101010 (binario)
    
    # 3. Carga de datos
    idx.add_batch(packed_db)
    
    # 4. Búsqueda semántica (ADC)
    query = np.random.randn(dim).astype(np.float32).tolist()
    k = 5
    results = idx.flat_search(query, k)
    
    print(f"\n📊 Resultados Integración:")
    print(f"   - Registros cargados: {n}")
    print(f"   - Resultados obtenidos: {len(results)}")
    if results:
        print(f"   - Top 1 IDX: {results[0][0]}, Distancia: {results[0][1]:.4f}")
    
    assert len(results) == k
    assert results[0][1] >= 0
    print("\n✅ INTEGRACIÓN BÁSICA EXITOSA")

if __name__ == "__main__":
    test_full_integration()
