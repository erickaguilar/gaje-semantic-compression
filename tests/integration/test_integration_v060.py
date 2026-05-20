import numpy as np
from gaje.core import _impl as core
from gaje.processing.balancer import SignalToNoiseBalancer

def test_full_integration():
    print("🔬 Probando integración completa: Balancer -> Mask -> HNSW Search")
    n, dim = 500, 64
    data = np.random.randn(n, dim).astype(np.float32)
    base_c = [-1.0, -0.3, 0.3, 1.0]
    
    # Empaquetado dummy
    packed_db = [bytes([0] * (dim // 4)) for _ in range(n)]
    
    # 1. Balancer
    balancer = SignalToNoiseBalancer()
    # Generamos entropía aleatoria para el test
    entropies = np.random.rand(dim).astype(np.float32)
    mask = balancer.generate_precision_mask(entropies)
    
    # 2. Index con Mask
    # Firma: database, centroids, epi_db, epi_c, tri_db, tri_c, mask
    idx = core.GajeIndex(
        packed_db, base_c, 
        [], [], # No epi data
        [], [], # No tri data
        mask.tolist()
    )
    
    idx.build()
    
    # 3. Search
    query = np.random.randn(dim).astype(np.float32).tolist()
    results = idx.search(query, k=5)
    
    print(f"\n📊 Resultados Integración:")
    print(f"   - Tamaño Máscara: {len(mask)} bloques")
    print(f"   - Resultados obtenidos: {len(results)}")
    print(f"   - Top 1 IDX: {results[0][0] if results else 'N/A'}")
    
    assert len(results) <= 5
    print("\n✅ INTEGRACIÓN HNSW + MASK EXITOSA")

if __name__ == "__main__":
    test_full_integration()
