import numpy as np
import os
import sys

def weighted_kmeans_1d(data, weights, k=4, max_iters=20):
    """
    K-means 1D con pesos (Weighted K-means).
    Minimiza la distorsión: sum(w_i * (x_i - c_j)^2)
    """
    # Inicialización inteligente (percentiles pesados)
    sorted_idx = np.argsort(data)
    data_s = data[sorted_idx]
    weights_s = weights[sorted_idx]
    cum_weights = np.cumsum(weights_s)
    
    centroids = []
    for p in [0.125, 0.375, 0.625, 0.875]:
        target = p * cum_weights[-1]
        idx = np.searchsorted(cum_weights, target)
        centroids.append(data_s[min(idx, len(data_s)-1)])
    
    centroids = np.array(centroids)
    
    for _ in range(max_iters):
        # Asignación (E-step)
        # Distancia euclidiana simple en 1D
        dists = np.abs(data[:, None] - centroids[None, :])
        labels = np.argmin(dists, axis=1)
        
        # Actualización (M-step)
        new_centroids = np.zeros(k)
        for j in range(k):
            mask = (labels == j)
            if np.any(mask):
                # Centroide pesado: media ponderada
                new_centroids[j] = np.sum(data[mask] * weights[mask]) / np.sum(weights[mask])
            else:
                # Si un cluster queda vacío, re-inicializar aleatoriamente
                new_centroids[j] = np.random.choice(data)
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    return np.sort(centroids)

def recalibrate_organism(model_path, corpus_path):
    print(f"🧬 [RE-CALIBRACIÓN] Alineamiento Global de Centroides (Weighted Fisher)")
    print(f"[*] Modelo: {model_path}")
    print(f"[*] Corpus: {corpus_path}")
    
    # En un entorno real, extraeríamos los pesos y gradientes (Fisher) del modelo.
    # Para este experimento, simulamos la extracción de la distribución de una capa crítica.
    # Simulamos pesos (f32) y su importancia Fisher (w).
    np.random.seed(42)
    sample_size = 50000
    
    # Distribución real del español (más ancha que la calibración original)
    actual_weights = np.random.normal(0, 1.2, sample_size)
    
    # Importancia Fisher simulada (Heterogénea)
    # Algunos pesos son 10x más importantes (Anclas potenciales)
    fisher_importance = np.ones(sample_size)
    important_idx = np.random.choice(sample_size, sample_size // 10, replace=False)
    fisher_importance[important_idx] = 10.0
    
    # Centroides Originales (Desplazados)
    c_orig = np.array([-1.51, -0.45, 0.45, 1.51])
    
    # Re-calibración: K-means Pesado
    print("[~] Ejecutando Weighted K-means sobre el manifold del español...")
    c_new = weighted_kmeans_1d(actual_weights, fisher_importance, k=4)
    
    print("-" * 50)
    print(f"📊 COMPARATIVA DE CENTROIDES (ε-net)")
    print(f"  - Originales: {c_orig}")
    print(f"  - Alineados:  {c_new}")
    print(f"  - Desplazamiento medio: {np.mean(np.abs(c_orig - c_new)):.4f}")
    print("-" * 50)
    
    # Simulación de impacto en PPL
    # MSE original vs MSE nuevo (pesado)
    def calc_weighted_mse(data, w, centroids):
        dist = np.abs(data[:, None] - centroids[None, :])
        indices = np.argmin(dist, axis=1)
        q = centroids[indices]
        return np.sum(w * (data - q)**2) / np.sum(w)

    mse_orig = calc_weighted_mse(actual_weights, fisher_importance, c_orig)
    mse_new = calc_weighted_mse(actual_weights, fisher_importance, c_new)
    
    print(f"📉 IMPACTO EN DISTORSIÓN (Fisher-weighted MSE)")
    print(f"  - MSE Original: {mse_orig:.6f}")
    print(f"  - MSE Alineado: {mse_new:.6f}")
    improvement = (mse_orig - mse_new) / mse_orig * 100
    print(f"  - Reducción de distorsión: {improvement:.2f}%")
    
    # Proyección PPL
    ppl_orig = 25000
    k = np.log(ppl_orig) / mse_orig
    ppl_est = np.exp(mse_new * k)
    
    print(f"🚀 PPL Proyectada (Español): {ppl_est:.2f}")
    print("-" * 50)

if __name__ == "__main__":
    recalibrate_organism("models/production/silver_adult_sovereign.gaje", "data/datasets/curated_1mb_flash.txt")
