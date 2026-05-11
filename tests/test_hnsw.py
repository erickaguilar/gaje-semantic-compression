import pytest
import numpy as np
from gaje.core import _impl as dna_semantic_compression

def test_hnsw_index_creation():
    db_dna = [os.urandom(192) for _ in range(10)] # 768 / 4 = 192 bytes
    centroids = [0.0] * (768 * 4)
    index = dna_semantic_compression.GajeIndex(db_dna, centroids)
    # Almacenamiento plano: len es el total de bytes
    assert len(index.database) == 10 * 192
    assert index.stride == 192

def test_hnsw_search_accuracy():
    dims = 768
    num_records = 100
    data = np.random.normal(0, 1, (num_records, dims)).astype(np.float32)
    thresholds = [-0.34, 0.0, 0.34]
    centroids = []
    for _ in range(dims):
        centroids.extend([-0.68, -0.17, 0.17, 0.68])
    
    db_dna = [dna_semantic_compression.quantize_embedding(v.tolist(), thresholds) for v in data]
    index = dna_semantic_compression.GajeIndex(db_dna, centroids)
    index.build()
    
    query = data[0].tolist()
    results = index.search(query, ef=50) # Increased ef for test stability
    
    # In HNSW (approximate), the exact match might not be at index 0 
    # but should definitely be in the top results
    top_indices = [idx for idx, dist in results[:10]]
    assert 0 in top_indices
    # In 768d, L2 distance of ~15 is normal for quantized vectors
    match_dist = [dist for idx, dist in results if idx == 0][0]
    assert match_dist < 20.0 

def test_lut_vs_normal_adc():
    # Verify that HNSW search (using LUT) is consistent with direct ADC
    dims = 768
    data = np.random.normal(0, 1, (10, dims)).astype(np.float32)
    thresholds = [-0.34, 0.0, 0.34]
    centroids = []
    for _ in range(dims):
        centroids.extend([-0.68, -0.17, 0.17, 0.68])
    
    db_dna = [dna_semantic_compression.quantize_embedding(v.tolist(), thresholds) for v in data]
    query = data[0].tolist()
    
    # 1. Direct ADC
    res_adc = dna_semantic_compression.dna_similarity_search_adc(query, db_dna, centroids)
    
    # 2. HNSW Search
    index = dna_semantic_compression.GajeIndex(db_dna, centroids)
    index.build()
    res_hnsw = index.search(query, ef=10)
    
    # They should find the same best match
    assert res_adc[0][0] == res_hnsw[0][0]

import os
