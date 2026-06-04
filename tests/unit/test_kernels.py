import numpy as np
import pytest
import sys
import os

# Asegurar el uso del paquete local
sys.path.append(os.path.abspath("python"))
from gaje.core import _impl as dna_core
from gaje.nn.stabilized import GenomicLayer

def test_minimal_mapping_consistency():
    """Verifica que el mapeo de bits a centroides sea consistente entre Rust y Python."""
    in_features = 32
    out_features = 1
    block_size = 32

    # 1. Pesos: Solo primera dimensión activa
    w_orig = np.zeros((out_features, in_features), dtype=np.float32)
    w_orig[0, 0] = 1.0
    x = np.zeros(in_features, dtype=np.float32)
    x[0] = 1.0

    # 2. Cuantización
    t = [0.1, 0.2, 0.3]
    dna = dna_core.quantize_embedding(w_orig[0].tolist(), t)

    # 3. Centroides: Bits 0b10 deben mapear a c[3] (según Gray code en lib.rs)
    c = [1.0, 2.0, 3.0, 42.0]

    linear = dna_core.GenomicLinear(dna, b"", c, out_features, in_features, block_size)
    y_rust = linear.forward(x, False)[0] # Rust espera (input, activate_rna)
    
    w_deq = np.array(dna_core.dequantize_embedding(dna, in_features, c))
    y_py = np.dot(w_deq, x)

    assert y_rust == y_py, f"Desajuste en mapeo: Rust={y_rust}, Python={y_py}"
    assert y_rust == 42.0, f"Mapeo inesperado para bits 10: {y_rust}"

def test_basic_kernel_vs_numpy():
    """Prueba la consistencia del kernel básico frente a np.dot."""
    in_features = 128
    out_features = 64
    w_orig = np.random.normal(0, 0.02, (out_features, in_features)).astype(np.float32)

    layer = GenomicLayer("test", w_orig, block_size=32)
    x = np.random.normal(0, 1.0, in_features).astype(np.float32)
    
    y_rust = layer.forward(x)
    
    # Referencia Python: Usamos get_row() del objeto nativo (.linear)
    w_deq = []
    for i in range(out_features):
        row = layer.linear.get_row(i)
        w_deq.append(row)
    
    y_py = np.dot(np.array(w_deq), x)
    assert np.allclose(y_rust, y_py, atol=1e-5)

def test_mixed_precision_kernel_phase12():
    """Verifica el kernel de precisión mixta (Base + Epi + Tri)."""
    in_features = 16
    out_features = 1
    block_size = 16
    
    weights_base = np.random.normal(0, 0.1, (out_features, in_features)).astype(np.float32)
    thresholds = [-0.1, 0.0, 0.1]
    centroids = [-0.15, -0.05, 0.05, 0.15]
    dna_base = b"".join([dna_core.quantize_embedding(row.tolist(), thresholds) for row in weights_base])
    
    # Epigenético (4-bit) y Triplete (6-bit)
    mask = bytes([2, 1, 0, 0]) # Tri, Epi, Base, Base (bloques de 4 dims)
    
    # Generar bases de datos dummy para epi y tri
    epi_centroids = [-0.07, -0.02, 0.02, 0.07]
    tri_centroids = [-0.03, -0.01, 0.01, 0.03]
    
    epi_strands = b"".join([dna_core.quantize_embedding(np.random.normal(0,0.05,8).tolist(), thresholds) for _ in range(out_features)])
    tri_strands = b"".join([dna_core.quantize_embedding(np.random.normal(0,0.02,4).tolist(), thresholds) for _ in range(out_features)])

    model = dna_core.GenomicLinear(
        dna_base, b"", centroids * out_features, out_features, in_features, block_size,
        [], 1e-6, mask, epi_strands, epi_centroids * out_features, tri_strands, tri_centroids * out_features
    )
    
    input_vec = np.random.normal(0, 1.0, in_features).astype(np.float32)
    output = model.forward(input_vec.tolist(), False)
    assert len(output) == out_features
    print(f"Mixed Precision Kernel Output: {output[0]:.6f}")

if __name__ == "__main__":
    pytest.main([__file__])
