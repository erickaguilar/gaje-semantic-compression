import sys
import urllib.request
import numpy as np

sys.path.append("benchmarks")
from train_codebook import train_genomic_codebook  # noqa: E402

try:
    import dna_semantic_compression  # noqa: E402
except ImportError:
    print("Error: Library not built. Run 'maturin develop' first.")
    sys.exit(1)

from sentence_transformers import SentenceTransformer  # noqa: E402


def get_sentences(num_sentences=5000):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"[*] Downloading text data from {url}...")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        text = response.read().decode("utf-8")

    # Split by lines and filter empty/short ones
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 20]

    sentences = lines[:num_sentences]
    print(f"[*] Extracted {len(sentences)} sentences.")
    return sentences


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def calculate_top_k_cosine(query_vec, db_vecs, k=10):
    norms = np.linalg.norm(db_vecs, axis=1)
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        return []
    similarities = np.dot(db_vecs, query_vec) / (norms * q_norm)
    top_indices = np.argsort(similarities)[::-1][:k]
    return top_indices.tolist()


def run_sbert_validation():
    print("🌍 GAJE PROTOCOL: SBERT REAL DATA VALIDATION (768 dims) 🌍")
    print("-" * 60)

    sentences = get_sentences(2000)

    print("[*] Loading SentenceTransformer model (all-mpnet-base-v2 - 768 dims)...")
    # This model outputs 768-dimensional embeddings
    model = SentenceTransformer("all-mpnet-base-v2")

    print("[*] Encoding sentences into vectors...")
    raw_db_vecs = model.encode(sentences, show_progress_bar=True)

    print("[*] Normalizing vectors for Cosine-equivalent ADC search...")
    db_vecs = np.array([normalize(v) for v in raw_db_vecs], dtype=np.float32)

    dims = db_vecs.shape[1]
    top_k = 10

    print(f"[*] Generated DB Shape: {db_vecs.shape}")

    # Train Codebook
    codebook = train_genomic_codebook(
        db_vecs, output_path="benchmarks/sbert_codebook.json"
    )
    thresholds = codebook["thresholds"]
    centroids = codebook["centroids"]

    print("[*] Compressing SBERT database into genomic strands...")
    db_dna = [
        dna_semantic_compression.quantize_pq(v.tolist(), thresholds) for v in db_vecs
    ]

    num_queries = 200
    overlaps_adc = []

    print(f"[*] Validating {num_queries} real SBERT semantic queries...")
    for _ in range(num_queries):
        q_idx = np.random.randint(0, len(db_vecs))
        query_vec = db_vecs[q_idx]

        # Ground Truth
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
    print(f"RESULTADOS FINALES (SBERT-{dims}d):")
    print(f"🎯 Precisión (Recall@10): {avg_accuracy:.2f}%")
    print("-" * 60)

    if avg_accuracy >= 85:
        print("🏆 OBJETIVO DEL ROADMAP ALCANZADO (>85% con vectores densos reales).")
    else:
        print("📈 BUEN RESULTADO, pero requiere más ajuste.")


if __name__ == "__main__":
    run_sbert_validation()
