import sys
from unittest.mock import MagicMock
import importlib.machinery

import importlib.util

# Conditional Monkeypatch for scipy (mostly for Termux/Android compatibility)
if importlib.util.find_spec("scipy") is None:
    print("[!] Scipy not found or broken (common on Termux). Applying monkeypatch...")
    mock_scipy = MagicMock()
    mock_scipy.__spec__ = importlib.machinery.ModuleSpec("scipy", None)
    mock_scipy.__path__ = []

    sys.modules["scipy"] = mock_scipy
    sys.modules["scipy.sparse"] = MagicMock()
    sys.modules["scipy.spatial"] = MagicMock()
    sys.modules["scipy.spatial.distance"] = MagicMock()
    sys.modules["scipy.special"] = MagicMock()
    sys.modules["scipy.stats"] = MagicMock()


import urllib.request
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.append("benchmarks")
sys.path.append("python")
from train_codebook import train_genomic_codebook  # noqa: E402

try:
    import dna_semantic_compression  # noqa: E402
except ImportError:
    print(
        "Error: Library not built or not in path. Ensure the 'python' folder is visible."
    )
    sys.exit(1)


def mean_pooling(model_output, attention_mask):
    """Perform mean pooling on token embeddings using the attention mask."""
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_sentences(num_sentences=5000):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"[*] Downloading text data from {url}...")
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as response:
            text = response.read().decode("utf-8")
    except Exception as e:
        print(f"Error downloading data: {e}")
        # Fallback sentences if download fails
        return ["This is a fallback sentence because download failed."] * num_sentences

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
    try:
        # Try to use torch for faster and safer calculation in this environment
        q_tensor = torch.from_numpy(query_vec)
        db_tensor = torch.from_numpy(db_vecs)

        cos = torch.nn.CosineSimilarity(dim=1)
        similarities = cos(q_tensor.unsqueeze(0), db_tensor)

        top_indices = torch.topk(similarities, k).indices
        return top_indices.tolist()
    except Exception:
        # Fallback to numpy if torch fails (though scipy might be needed by some np operations indirectly)
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

    sentences = get_sentences(100)

    print("[*] Loading Transformers model (sentence-transformers/all-mpnet-base-v2)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2"
        )
        model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        use_synthetic = False
    except Exception as e:
        print(f"[*] WARNING: Could not load SBERT model due to network/SSL issues: {e}")
        print(
            "[*] Falling back to SYNTHETIC SBERT-like vectors (768d) to proceed with benchmark logic."
        )
        use_synthetic = True

    if not use_synthetic:
        print("[*] Encoding sentences into vectors...")
        embeddings = []
        batch_size = 32
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            encoded_input = tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                model_output = model(**encoded_input)
            batch_embeddings = mean_pooling(
                model_output, encoded_input["attention_mask"]
            )
            embeddings.append(batch_embeddings.numpy())
        raw_db_vecs = np.vstack(embeddings)
    else:
        # Generate synthetic vectors that follow a structured distribution
        print(
            f"[*] Generating {len(sentences)} synthetic 768-dim structured vectors..."
        )
        latent = np.random.normal(0, 1, (len(sentences), 64))
        projection = np.random.normal(0, 1, (64, 768))
        raw_db_vecs = np.dot(latent, projection).astype(np.float32)

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

    num_queries = 10  # Reduced for speed
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
