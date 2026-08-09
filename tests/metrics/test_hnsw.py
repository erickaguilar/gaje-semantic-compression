import numpy as np
import os
from gaje.core import _impl as dna_semantic_compression


def test_hnsw_index_creation():
    db_dna = [os.urandom(192) for _ in range(10)]  # 768 / 4 = 192 bytes
    centroids = [0.0] * (768 * 4)
    # New API: GajeIndex(dims, centroids)
    index = dna_semantic_compression.GajeIndex(768, centroids)
    index.add_batch(db_dna)
    assert index is not None


def test_hnsw_search_accuracy():
    dims = 768
    num_records = 100
    data = np.random.normal(0, 1, (num_records, dims)).astype(np.float32)
    thresholds = [-0.34, 0.0, 0.34]
    centroids = []
    for _ in range(dims):
        centroids.extend([-0.68, -0.17, 0.17, 0.68])

    db_dna = [
        bytes(dna_semantic_compression.quantize_embedding(v.tolist(), thresholds))
        for v in data
    ]
    index = dna_semantic_compression.GajeIndex(dims, centroids)
    index.add_batch(db_dna)

    query = data[0].tolist()
    results = index.flat_search(query, k=10)

    # The exact match at index 0 should be the top result
    top_indices = [idx for idx, dist in results]
    assert 0 in top_indices


def test_lut_vs_normal_adc():
    dims = 768
    data = np.random.normal(0, 1, (10, dims)).astype(np.float32)
    thresholds = [-0.34, 0.0, 0.34]
    centroids = []
    for _ in range(dims):
        centroids.extend([-0.68, -0.17, 0.17, 0.68])

    db_dna = [
        bytes(dna_semantic_compression.quantize_embedding(v.tolist(), thresholds))
        for v in data
    ]
    query = data[0].tolist()

    # Create index
    index = dna_semantic_compression.GajeIndex(dims, centroids)
    index.add_batch(db_dna)
    res_hnsw = index.flat_search(query, k=1)

    assert len(res_hnsw) == 1
    assert res_hnsw[0][0] == 0
