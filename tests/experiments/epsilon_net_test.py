import numpy as np


def quantize_2bit(data, centroids):
    """Mapeo a 2 bits (4 centroides)."""
    # Encontramos el centroide más cercano para cada valor
    dist = np.abs(data[:, None] - centroids[None, :])
    indices = np.argmin(dist, axis=1)
    return centroids[indices], indices


def quantize_4bit(data, centroids):
    """Mapeo a 4 bits (16 centroides)."""
    dist = np.abs(data[:, None] - centroids[None, :])
    indices = np.argmin(dist, axis=1)
    return centroids[indices], indices


def run_experiment(size=100000):
    print(f"🧪 Iniciando Experimento de Escalado de Bits (N={size})")

    # Simular una distribución de pesos (Gaussiana con cola larga)
    np.random.seed(42)
    weights = np.random.normal(0, 1, size)

    # 2 Bits (4 Centroides)
    c2 = np.array([-1.5, -0.5, 0.5, 1.5])
    q2, _ = quantize_2bit(weights, c2)
    mse2 = np.mean((weights - q2) ** 2)

    # 4 Bits (16 Centroides)
    # Generamos 16 centroides uniformes en el rango [-2, 2]
    c4 = np.linspace(-2, 2, 16)
    q4, _ = quantize_4bit(weights, c4)
    mse4 = np.mean((weights - q4) ** 2)

    print("-" * 30)
    print("📊 RESULTADOS (MSE: Mean Squared Error)")
    print(f"  - 2 bits (ε-net rala):   {mse2:.6f}")
    print(f"  - 4 bits (ε-net densa):  {mse4:.6f}")
    print(f"  - Mejora teórica:        {((mse2 - mse4) / mse2) * 100:.2f}%")
    print("-" * 30)

    # Implicación semántica (PPL Estimada)
    # Si asumimos que PPL ~ exp(MSE * k)
    ppl_baseline = 25000
    k = np.log(ppl_baseline) / mse2
    ppl_est_4bit = np.exp(mse4 * k)

    print("📉 PROYECCIÓN SEMÁNTICA")
    print(f"  - PPL Proyectada 4-bit:  {ppl_est_4bit:.2f}")
    print(
        f"  - Meta Nivel 2 (<15):    {'POSIBLE' if ppl_est_4bit < 15 else 'INSUFICIENTE'}"
    )
    print("-" * 30)


if __name__ == "__main__":
    run_experiment()
