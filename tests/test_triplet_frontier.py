import numpy as np
from gaje.core import GajeIndex
import time
from gaje.utils.codebook import fast_kmeans_1d

def train_centroids(data, n_clusters=4):
    return fast_kmeans_1d(data.flatten(), k=n_clusters).tolist()

def quantize(data, centroids):
    packed = []
    stride = data.shape[1] // 4
    for row in data:
        row_bytes = []
        for i in range(stride):
            byte = 0
            for j in range(4):
                val = row[i*4 + j]
                idx = np.argmin([abs(val - c) for c in centroids])
                bits = [0b00, 0b01, 0b11, 0b10][idx]
                byte = (byte << 2) | bits
            row_bytes.append(byte)
        packed.append(bytes(row_bytes))
    return packed

def dequantize(packed, centroids, dim):
    rec = np.zeros(dim)
    dims = 0
    for b in packed:
        for j in range(4):
            if dims >= dim: break
            shift = (3 - j) * 2
            bits = (b >> shift) & 0b11
            idx = {0b00:0, 0b01:1, 0b11:2, 0b10:3}[bits]
            rec[dims] = centroids[idx]
            dims += 1
    return rec

def test_triplet_frontier():
    print("🔬 Validando la Triplet Frontier (6-bit Navigation)...")
    
    n_vectors = 2000
    dim = 128
    data = np.random.randn(n_vectors, dim).astype(np.float32)
    query = np.random.randn(dim).astype(np.float32)
    
    # Ground Truth
    distances = np.linalg.norm(data - query, axis=1)
    gt_indices = np.argsort(distances)[:10]
    
    # 1. Base (2-bit)
    base_centroids = train_centroids(data)
    base_packed = quantize(data, base_centroids)
    
    # 2. Epigenetic (Residual 2-bit -> 4-bit total)
    dequant_base = np.array([dequantize(p, base_centroids, dim) for p in base_packed])
    residuals_1 = data - dequant_base
    epi_centroids = train_centroids(residuals_1)
    epi_packed = quantize(residuals_1, epi_centroids)
    
    # 3. Triplet (Residual of Residual 2-bit -> 6-bit total)
    dequant_epi = np.array([dequantize(p, epi_centroids, dim) for p in epi_packed])
    residuals_2 = residuals_1 - dequant_epi
    tri_centroids = train_centroids(residuals_2)
    tri_packed = quantize(residuals_2, tri_centroids)
    
    # --- Experiment A: Base Only (2-bit) ---
    index_2b = GajeIndex(base_packed, base_centroids)
    index_2b.build()
    res_2b = index_2b.search(query.tolist(), 10, 100)
    recall_2b = len(set([r[0] for r in res_2b]) & set(gt_indices)) / 10.0
    
    # --- Experiment B: Epigenetic (4-bit) ---
    index_4b = GajeIndex(base_packed, base_centroids, epi_packed, epi_centroids)
    index_4b.build()
    res_4b = index_4b.search(query.tolist(), 10, 100)
    recall_4b = len(set([r[0] for r in res_4b]) & set(gt_indices)) / 10.0
    
    # --- Experiment C: Triplet Frontier (6-bit) ---
    index_6b = GajeIndex(base_packed, base_centroids, epi_packed, epi_centroids, tri_packed, tri_centroids)
    index_6b.build()
    
    # Benchmark Search Speed
    start = time.time()
    res_6b = index_6b.search(query.tolist(), 10, 100)
    end = time.time()
    
    recall_6b = len(set([r[0] for r in res_6b]) & set(gt_indices)) / 10.0
    
    print(f"\n📊 RESULTADOS (N={n_vectors}, Dim={dim})")
    print(f"   [DNA 2-bit]   Recall@10: {recall_2b:.2%}")
    print(f"   [Epi 4-bit]   Recall@10: {recall_4b:.2%}")
    print(f"   [Triplet 6-b] Recall@10: {recall_6b:.2%}")
    print(f"   Latencia 6-bit (NEON): {(end-start)*1000:.4f}ms")
    
    assert recall_6b >= recall_4b, "La Triplet Frontier debería mejorar o mantener el Recall."
    print("\n✅ VALIDACIÓN EXITOSA: La precisión de 6 bits ha sido integrada en el motor.")

if __name__ == "__main__":
    test_triplet_frontier()
