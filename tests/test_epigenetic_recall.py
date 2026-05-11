import numpy as np
from gaje.core import GajeIndex
import time

from gaje.utils.codebook import fast_kmeans_1d

def train_centroids(data, n_clusters=4):
    return fast_kmeans_1d(data.flatten(), k=n_clusters).tolist()

def quantize(data, centroids):
    # simple nearest centroid quantization
    packed = []
    stride = data.shape[1] // 4
    for row in data:
        row_bytes = []
        for i in range(stride):
            byte = 0
            for j in range(4):
                val = row[i*4 + j]
                # find closest centroid index
                idx = np.argmin([abs(val - c) for c in centroids])
                # map to GAJE bits: 00, 01, 11, 10
                bits = [0b00, 0b01, 0b11, 0b10][idx]
                byte = (byte << 2) | bits
            row_bytes.append(byte)
        packed.append(bytes(row_bytes))
    return packed

def test_epigenetic_recall():
    print("🔬 Investigando Impacto de la Capa Epigenética en el Recall...")
    
    n_vectors = 1000
    dim = 128
    data = np.random.randn(n_vectors, dim).astype(np.float32)
    query = np.random.randn(dim).astype(np.float32)
    
    # Ground Truth
    distances = np.linalg.norm(data - query, axis=1)
    gt_indices = np.argsort(distances)[:10]
    
    # 1. Base Quantization (2-bit)
    base_centroids = train_centroids(data)
    base_packed = quantize(data, base_centroids)
    
    # Calculate Residuals
    dequantized_base = np.zeros_like(data)
    for i, p in enumerate(base_packed):
        dims = 0
        for b in p:
            for j in range(4):
                shift = (3 - j) * 2
                bits = (b >> shift) & 0b11
                idx = {0b00:0, 0b01:1, 0b11:2, 0b10:3}[bits]
                dequantized_base[i, dims] = base_centroids[idx]
                dims += 1
    
    residuals = data - dequantized_base
    
    # 2. Epigenetic Quantization (Residual 2-bit)
    epi_centroids = train_centroids(residuals)
    epi_packed = quantize(residuals, epi_centroids)
    
    # --- Experiment A: Base Only ---
    index_base = GajeIndex(base_packed, base_centroids)
    index_base.build()
    results_base = index_base.search(query.tolist(), 10, 100)
    idx_base = [r[0] for r in results_base]
    recall_base = len(set(idx_base) & set(gt_indices)) / 10.0
    
    # --- Experiment B: Epigenetic (Base + Residual) ---
    index_epi = GajeIndex(base_packed, base_centroids, epi_packed, epi_centroids)
    index_epi.build()
    results_epi = index_epi.search(query.tolist(), 10, 100)
    idx_epi = [r[0] for r in results_epi]
    recall_epi = len(set(idx_epi) & set(gt_indices)) / 10.0
    
    print(f"\n📊 RESULTADOS (N={n_vectors}, Dim={dim})")
    print(f"   [Base 2-bit] Recall@10: {recall_base:.2%}")
    print(f"   [Epi  4-bit] Recall@10: {recall_epi:.2%}")
    
    improvement = (recall_epi - recall_base) * 100
    print(f"\n🚀 Incremento de Recall: +{improvement:.2f} puntos")
    
    assert recall_epi >= recall_base

if __name__ == "__main__":
    try:
        test_epigenetic_recall()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
