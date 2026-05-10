import os
import sys
import numpy as np

# Ensure we can import our modules
sys.path.append(os.path.abspath("python"))
sys.path.append(os.path.abspath("benchmarks"))

try:
    from gaje.core import _impl as dna_semantic_compression
except ImportError:
    print("Error: Library not built. Run 'maturin develop' first.")
    sys.exit(1)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def calculate_top_k(query_vec, db_vecs, k=10):
    # Normalized dot product
    dots = np.dot(db_vecs, query_vec)
    top_indices = np.argsort(dots)[::-1][:k]
    return top_indices.tolist()


def run_accuracy_test():
    print("🧪 GAJE PROTOCOL: HIGH ACCURACY TEST (Phase 2+) 🧪")
    print("-" * 50)

    num_records = 2000
    dims = 768
    top_k = 10

    # 1. Generate Real-looking Data (Normalized)
    # Using a slightly structured distribution to simulate real manifolds
    print(f"[*] Generating {num_records} structured vectors...")
    latent = np.random.normal(0, 1, (num_records, 64))
    projection = np.random.normal(0, 1, (64, dims))
    db_vecs = np.dot(latent, projection).astype(np.float32)
    db_vecs = np.array([normalize(v) for v in db_vecs])

    # 2. Train Codebook (Per Dimension)
    from gaje.utils.codebook import train_genomic_codebook

    codebook = train_genomic_codebook(db_vecs, mode="per_dim")
    thresholds = codebook["thresholds"]
    centroids = codebook["centroids"]

    # 3. Pack into DNA strands
    print("[*] Quantizing database...")
    db_dna = [
        dna_semantic_compression.quantize_embedding(v.tolist(), thresholds)
        for v in db_vecs
    ]

    # 4. Select random queries
    num_queries = 100
    overlaps_adc = []

    print(f"[*] Validating {num_queries} queries (Top-10 Overlap)...")

    for _ in range(num_queries):
        q_idx = np.random.randint(0, num_records)
        # Add noise (simulating semantic variation)
        query_vec = normalize(db_vecs[q_idx] + np.random.normal(0, 0.02, dims))

        # Ground Truth
        truth_indices = calculate_top_k(query_vec, db_vecs, k=top_k)

        # DNA Space Results (ADC)
        adc_results = dna_semantic_compression.dna_similarity_search_adc(
            query_vec.tolist(), db_dna, centroids
        )
        adc_indices = [idx for idx, dist in adc_results[:top_k]]

        # Calculate Overlap
        intersection_adc = set(truth_indices).intersection(set(adc_indices))
        overlaps_adc.append(len(intersection_adc) / top_k)

    avg_accuracy_adc = np.mean(overlaps_adc) * 100

    print("-" * 50)
    print("RESULTADO DE PRECISIÓN (Recall@10):")
    print(f"🚀 ADC (Asimétrico + Per-Dim K-Means): {avg_accuracy_adc:.2f}%")

    if avg_accuracy_adc >= 85:
        print("🏆 OBJETIVO ALCANZADO: El protocolo GAJE es ahora grado industrial.")
    elif avg_accuracy_adc >= 70:
        print("✅ FASE 2 COMPLETADA: Superamos el 70% significativamente.")

    print("-" * 50)

    # Save results
    with open("benchmarks/ACCURACY_LOG.txt", "w") as f:
        f.write(f"Recall@10 ADC (Per-Dim Optimized): {avg_accuracy_adc:.2f}%\n")


if __name__ == "__main__":
    run_accuracy_test()
