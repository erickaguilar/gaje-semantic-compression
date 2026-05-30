from gaje.core import _impl as dna_semantic_compression
import numpy as np


def kmeans_1d(data, k, iterations=10):
    # Inicializar centroides con percentiles
    centroids = np.percentile(data, np.linspace(0, 100, k + 2)[1:-1])
    for _ in range(iterations):
        # Asignar puntos al centroide más cercano
        distances = np.abs(data[:, None] - centroids)
        labels = np.argmin(distances, axis=1)
        # Actualizar centroides
        new_centroids = np.array(
            [
                data[labels == i].mean() if np.any(labels == i) else centroids[i]
                for i in range(k)
            ]
        )
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return sorted(centroids)


def debug_signal_fidelity():
    print("🔬 DEBUG: SIGNAL FIDELITY ANALYSIS (K-Means 1D Manual)")
    dims = 768

    np.random.seed(42)
    w_original = np.random.normal(0, 0.02, dims).astype(np.float32)

    # K-Means 1D manual para 4 centroides
    c = kmeans_1d(w_original, 4)

    # Los umbrales son el punto medio entre centroides
    t = [(c[i] + c[i + 1]) / 2 for i in range(len(c) - 1)]

    # Cuantizar y De-cuantizar
    dna = dna_semantic_compression.quantize_embedding(w_original.tolist(), t)
    w_rec = np.array(dna_semantic_compression.dequantize_embedding(dna, dims, c))

    # Comparar estadísticas
    mse = np.mean((w_original - w_rec) ** 2)
    cos_sim = np.dot(w_original, w_rec) / (
        np.linalg.norm(w_original) * np.linalg.norm(w_rec)
    )

    print("--- Estadísticas (K-Means 1D) ---")
    print(f"MSE:      {mse:.8f}")
    print(f"Cos Sim:  {cos_sim:.6f}")
    print(
        f"Rel Error: {np.sum(np.abs(w_original - w_rec)) / np.sum(np.abs(w_original)):.4%}"
    )


if __name__ == "__main__":
    debug_signal_fidelity()
