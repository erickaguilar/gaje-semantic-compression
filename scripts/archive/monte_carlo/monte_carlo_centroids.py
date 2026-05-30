import numpy as np
import time


def simulate_gaje_quantization(weights_f32, centroids):
    """
    Simula la cuantización a 2 bits utilizando los centroides dados
    y devuelve el error de reconstrucción (MSE).
    """
    # 1. Asignar cada peso al centroide más cercano
    # En un entorno real, esto se hace en bloque y bitwise (Rust).
    # Aquí usamos broadcasting de NumPy para simulación rápida.
    distances = np.abs(weights_f32[:, np.newaxis] - centroids)
    nearest_idx = np.argmin(distances, axis=1)

    # 2. Reconstruir los pesos usando los centroides elegidos
    reconstructed = centroids[nearest_idx]

    # 3. Calcular el Error Cuadrático Medio (MSE)
    mse = np.mean((weights_f32 - reconstructed) ** 2)
    return mse


def monte_carlo_centroid_search(weights_f32, iterations=5000, noise_scale=0.1):
    """
    Utiliza Simulación Monte Carlo para encontrar los 4 centroides óptimos.
    En lugar de descenso de gradiente, aplicamos mutaciones aleatorias masivas.
    """
    print(f"🎲 Iniciando Simulación Monte Carlo ({iterations} iteraciones)...")

    # Base estadística clásica (nuestro punto de partida heurístico)
    mean = np.mean(weights_f32)
    std = np.std(weights_f32)
    base_centroids = np.array(
        [
            mean - 1.510 * std,
            mean - 0.4528 * std,
            mean + 0.4528 * std,
            mean + 1.510 * std,
        ]
    )

    best_centroids = base_centroids
    best_mse = simulate_gaje_quantization(weights_f32, base_centroids)

    print(f"   [Base] MSE Inicial: {best_mse:.6f}")

    start_time = time.time()

    # Bucle Principal de Monte Carlo
    for i in range(iterations):
        # Generar mutaciones aleatorias (ruido Gaussiano)
        mutation = np.random.normal(0, noise_scale * std, 4)
        candidate_centroids = base_centroids + mutation

        # Mantener el orden para preservar la lógica de cuantización
        candidate_centroids.sort()

        # Evaluar aptitud (fitness)
        mse = simulate_gaje_quantization(weights_f32, candidate_centroids)

        # Selección: Si la mutación mejora la aptitud, la aceptamos
        if mse < best_mse:
            best_mse = mse
            best_centroids = candidate_centroids

        if (i + 1) % 1000 == 0:
            print(f"   [Iter {i+1}] Mejor MSE actual: {best_mse:.6f}")

    duration = time.time() - start_time

    print(f"\n✅ Simulación completada en {duration:.2f}s")
    print(
        f"   Mejora Relativa: {((simulate_gaje_quantization(weights_f32, base_centroids) - best_mse) / simulate_gaje_quantization(weights_f32, base_centroids)) * 100:.2f}%"
    )
    print(f"   Centroides Clásicos: {base_centroids}")
    print(f"   Centroides Óptimos (MC): {best_centroids}")


if __name__ == "__main__":
    # Generar un tensor de ejemplo simulando una capa densa pequeña (ej. 1024 pesos)
    # Usamos una distribución ligeramente asimétrica para que MC tenga ventaja
    # sobre la asunción gaussiana perfecta de la estadística clásica.
    np.random.seed(42)
    dummy_weights = np.random.normal(0, 0.02, 1024) + np.random.uniform(
        -0.01, 0.01, 1024
    )

    monte_carlo_centroid_search(dummy_weights, iterations=10000, noise_scale=0.15)
