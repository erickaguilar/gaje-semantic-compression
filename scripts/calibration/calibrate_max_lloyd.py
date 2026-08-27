import numpy as np
import gguf
import argparse
import os


def max_lloyd(weights, n_clusters=4, iterations=20, tol=1e-5):
    """
    Algoritmo de Max-Lloyd para encontrar centroides óptimos.
    """
    # Inicialización K-means++ simple
    centroids = np.sort(np.random.choice(weights, n_clusters))

    for i in range(iterations):
        # 1. Partición de Voronoi (E-step)
        # Calculamos los límites entre centroides
        boundaries = (centroids[:-1] + centroids[1:]) / 2

        # 2. Actualización de centroides (M-step)
        # Calculamos la media de cada intervalo
        new_centroids = np.zeros_like(centroids)

        # Caso 0: weights < boundaries[0]
        mask = weights < boundaries[0]
        if np.any(mask):
            new_centroids[0] = np.mean(weights[mask])
        else:
            new_centroids[0] = centroids[0]

        # Casos intermedios
        for j in range(1, n_clusters - 1):
            mask = (weights >= boundaries[j - 1]) & (weights < boundaries[j])
            if np.any(mask):
                new_centroids[j] = np.mean(weights[mask])
            else:
                new_centroids[j] = centroids[j]

        # Último caso: weights >= boundaries[-1]
        mask = weights >= boundaries[-1]
        if np.any(mask):
            new_centroids[-1] = np.mean(weights[mask])
        else:
            new_centroids[-1] = centroids[-1]

        # Comprobar convergencia
        diff = np.abs(new_centroids - centroids).sum()
        centroids = np.sort(new_centroids)

        if diff < tol:
            break

    return centroids


def main():
    parser = argparse.ArgumentParser(
        description="🧬 GAJE: Calibración Max-Lloyd por Bloque"
    )
    parser.add_argument("--model", type=str, required=True, help="Ruta al modelo GGUF")
    parser.add_argument(
        "--limit", type=int, default=10, help="Límite de bloques a procesar"
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Error: Modelo no encontrado en {args.model}")
        return

    print(f"[*] Cargando modelo GGUF: {args.model}")
    reader = gguf.GGUFReader(args.model)

    # Buscamos tensores de las capas FFN (donde más duele el ruido)
    print("[*] Iniciando calibración Max-Lloyd (2-bit, 4 centroides)...")
    print("-" * 60)
    print(f"{'Capa':<40} | {'Centroides Óptimos':<30}")
    print("-" * 60)

    blocks_processed = 0
    for tensor in reader.tensors:
        # Filtrar solo capas de pesos (weights) de los bloques principales
        if any(
            x in tensor.name
            for x in [
                "ffn_down.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "attn_q.weight",
            ]
        ):
            weights = tensor.data.astype(np.float32).flatten()

            # Sub-muestreo para velocidad si el tensor es muy grande
            if len(weights) > 1_000_000:
                weights_sample = np.random.choice(weights, 1_000_000, replace=False)
            else:
                weights_sample = weights

            centroids = max_lloyd(weights_sample)
            print(
                f"{tensor.name:<40} | {np.array2string(centroids, precision=4, separator=', ')}"
            )

            blocks_processed += 1
            if blocks_processed >= args.limit:
                break

    print("-" * 60)
    print(
        "[*] Calibración finalizada. Estos centroides deben inyectarse en el cargador GAJE."
    )


if __name__ == "__main__":
    main()
