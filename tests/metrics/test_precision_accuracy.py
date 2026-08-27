import os
import sys
import numpy as np
import pytest

# Asegurar uso de código local
sys.path.append(os.path.abspath("python"))
from gaje.core import _impl as dna_core
from gaje.nn.stabilized import GenomicLLM
from gaje.utils.codebook import train_genomic_codebook


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def calculate_top_k(query_vec, db_vecs, k=10):
    dots = np.dot(db_vecs, query_vec)
    return np.argsort(dots)[::-1][:k].tolist()


def test_recall_at_10_adc():
    """Valida la precisión de búsqueda (Recall@10) usando ADC (Asymmetric Distance Computation)."""
    num_records, dims, top_k = 500, 128, 10

    # Generar datos estructurados
    latent = np.random.normal(0, 1, (num_records, 32))
    projection = np.random.normal(0, 1, (32, dims))
    db_vecs = np.array(
        [normalize(v) for v in np.dot(latent, projection).astype(np.float32)]
    )

    # Entrenar codebook y cuantizar
    codebook = train_genomic_codebook(db_vecs, mode="per_dim")
    db_dna = [
        dna_core.quantize_embedding(v.tolist(), codebook["thresholds"]) for v in db_vecs
    ]

    # Validar queries
    overlaps = []
    for _ in range(20):
        q_idx = np.random.randint(0, num_records)
        query_vec = normalize(db_vecs[q_idx] + np.random.normal(0, 0.05, dims))

        truth_indices = calculate_top_k(query_vec, db_vecs, k=top_k)
        adc_results = dna_core.dna_similarity_search_adc(
            query_vec.tolist(), db_dna, codebook["centroids"]
        )
        adc_indices = [idx for idx, dist in adc_results[:top_k]]

        intersection = set(truth_indices).intersection(set(adc_indices))
        overlaps.append(len(intersection) / top_k)

    avg_recall = np.mean(overlaps)
    print(f"Recall@10 ADC: {avg_recall * 100:.2f}%")
    assert avg_recall > 0.7, "El Recall@10 es demasiado bajo para el protocolo GAJE"


def test_next_token_prediction_precision():
    """Valida que el modelo prediga tokens coherentes en frases de baja entropía."""
    model_path = "models/gguf/qwen2-0_5b-q8_0.gguf"
    if not os.path.exists(model_path):
        pytest.skip(f"Modelo no encontrado en {model_path}")

    llm = GenomicLLM(model_path, num_blocks=4)
    test_phrase = "Paris is the capital of"

    tokens = llm.tokenizer.encode(test_phrase, add_special_tokens=False)
    logits = llm.forward(tokens)[-1]

    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()

    top_id = np.argmax(probs)
    predicted_token = llm.tokenizer.decode([top_id]).strip()

    print(f"Contexto: '{test_phrase}' -> Predicción: '{predicted_token}'")
    # En Qwen2/SmolLM2, la respuesta debería ser 'France'
    assert "France" in predicted_token or "France" in llm.tokenizer.decode(
        [top_id]
    ), f"Predicción inesperada: {predicted_token}"


if __name__ == "__main__":
    pytest.main([__file__])
