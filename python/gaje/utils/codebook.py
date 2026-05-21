import numpy as np
import json


def fast_kmeans_1d(data, k=4):
    if len(data) < k:
        return np.linspace(np.min(data), np.max(data), k)

    # Initialize with percentiles for stability
    centroids = np.percentile(data, np.linspace(0, 100, k + 2)[1:-1])

    for _ in range(10):
        # Assign
        diffs = np.abs(data[:, np.newaxis] - centroids)
        labels = np.argmin(diffs, axis=1)

        # Update
        new_centroids = []
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                new_centroids.append(data[mask].mean())
            else:
                new_centroids.append(centroids[i])

        new_centroids = np.sort(np.array(new_centroids))
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids


def train_genomic_codebook(vectors, output_path="models/core/codebook.json", mode="per_dim"):
    """
    mode: 'global' or 'per_dim'
    """
    print(f"🧬 TRAINING GENOMIC CODEBOOK ({mode.upper()}) 🧬")
    print("-" * 50)

    num_records, dims = vectors.shape

    all_centroids = []
    all_thresholds = []

    if mode == "global":
        flat_data = vectors.flatten()
        centroids = fast_kmeans_1d(flat_data, k=4)
        thresholds = [(centroids[i] + centroids[i + 1]) / 2 for i in range(3)]
        all_centroids = [float(c) for c in centroids]
        all_thresholds = [float(t) for t in thresholds]
    else:
        # Per dimension
        for d in range(dims):
            dim_data = vectors[:, d]
            centroids = fast_kmeans_1d(dim_data, k=4)
            thresholds = [(centroids[i] + centroids[i + 1]) / 2 for i in range(3)]
            all_centroids.extend([float(c) for c in centroids])
            all_thresholds.extend([float(t) for t in thresholds])

    codebook = {
        "mode": mode,
        "dims": dims,
        "centroids": all_centroids,
        "thresholds": all_thresholds,
    }

    with open(output_path, "w") as f:
        json.dump(codebook, f, indent=4)

    print(f"✅ Codebook saved to: {output_path}")
    print(f"  Total Centroids: {len(all_centroids)}")
    print("-" * 50)
    return codebook


if __name__ == "__main__":
    biased_data = np.random.normal(0.2, 0.4, (1000, 768)).astype(np.float32)
    train_genomic_codebook(biased_data, mode="per_dim")
