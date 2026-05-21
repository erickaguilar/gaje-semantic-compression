import numpy as np
import os
import sys

# Add benchmarks path to sys.path to import train_codebook
sys.path.append("benchmarks")
from gaje.utils.codebook import train_genomic_codebook  # noqa: E402

try:
    from gaje.core import _impl as dna_semantic_compression  # noqa: E402
except ImportError:
    print("Error: Library not built. Run 'maturin develop' first.")
    sys.exit(1)


def load_glove_subset(file_path, num_vectors=10000):
    print(f"[*] Loading {num_vectors} GloVe vectors from {file_path}...")
    vectors = []
    words = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_vectors:
                break
            parts = line.split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            vectors.append(vector)
            words.append(word)
    return np.array(vectors), words


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def calculate_top_k_cosine(query_vec, db_vecs, k=10):
    # Normalize for cosine similarity
    norms = np.linalg.norm(db_vecs, axis=1)
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        return []

    similarities = np.dot(db_vecs, query_vec) / (norms * q_norm)
    top_indices = np.argsort(similarities)[::-1][:k]
    return top_indices.tolist()


def run_phase3_validation():
    print("🌍 GAJE PROTOCOL: PHASE 3 - REAL DATA VALIDATION (GloVe) 🌍")
    print("-" * 60)

    glove_path = "data/glove.6B.100d.txt"
    if not os.path.exists(glove_path):
        print(f"Error: {glove_path} not found.")
        return

    # 1. Load Real Data
    raw_db_vecs, words = load_glove_subset(glove_path, num_vectors=5000)

    # Normalize vectors so L2 distance in ADC matches Cosine similarity!
    print("[*] Normalizing vectors for Cosine-equivalent ADC search...")
    db_vecs = np.array([normalize(v) for v in raw_db_vecs], dtype=np.float32)

    db_vecs.shape[1]
    top_k = 10

    # 2. Train Codebook on Real Distribution
    codebook = train_genomic_codebook(
        db_vecs, output_path="benchmarks/logs/glove_codebook.json"
    )
    thresholds = codebook["thresholds"]
    centroids = codebook["centroids"]

    # 3. Compress DB into DNA
    print("[*] Compressing GloVe database into genomic strands...")
    db_dna = [
        dna_semantic_compression.quantize_pq(v.tolist(), thresholds) for v in db_vecs
    ]

    # 4. Perform Comparative Tests
    num_queries = 100
    overlaps_adc = []

    print(f"[*] Validating {num_queries} real semantic queries...")
    for _ in range(num_queries):
        q_idx = np.random.randint(0, len(db_vecs))

        # We must use the normalized query for ADC search
        query_vec = db_vecs[q_idx]

        # Ground Truth can use either, but let's use the normalized ones since they represent the same
        truth_indices = calculate_top_k_cosine(query_vec, db_vecs, k=top_k)

        # GAJE ADC Results
        adc_results = dna_semantic_compression.dna_similarity_search_adc(
            query_vec.tolist(), db_dna, centroids
        )
        adc_indices = [idx for idx, dist in adc_results[:top_k]]

        intersection = set(truth_indices).intersection(set(adc_indices))
        overlaps_adc.append(len(intersection) / top_k)

    avg_accuracy = np.mean(overlaps_adc) * 100

    print("-" * 60)
    print("RESULTADOS FINALES (GloVe-100d):")
    print(f"🎯 Precisión (Recall@10): {avg_accuracy:.2f}%")
    print("-" * 60)

    if avg_accuracy >= 80:
        print(
            "🏆 OBJETIVO ALCANZADO: El protocolo es altamente confiable en datos reales."
        )
    elif avg_accuracy >= 60:
        print(
            "📈 EXCELENTE PROGRESO: Superior a cualquier método de compresión binaria estándar."
        )
    else:
        print(
            "💡 NOTA: GloVe tiene una estructura densa. Se recomienda probar con sub-espacios más pequeños (PQ Fine-tuning)."
        )


if __name__ == "__main__":
    run_phase3_validation()
