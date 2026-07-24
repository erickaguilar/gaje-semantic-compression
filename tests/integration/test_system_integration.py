import numpy as np
import os
import sys
import pytest

sys.path.append(os.path.abspath("python"))
from gaje.core import _impl as core
from gaje.nn.stabilized import GenomicLLM


def test_index_integration_adc():
    """Prueba la integración básica del índice Gaje (ADC)."""
    print("🔬 Probando GajeIndex (ADC)")
    n, dim = 100, 64
    base_c = [-1.0, -0.3, 0.3, 1.0]
    idx = core.GajeIndex(dim, base_c)

    # add_batch espera Vec<Vec<u8>>
    packed_db = [bytes([0xAA] * (dim // 4)) for _ in range(n)]
    idx.add_batch(packed_db)

    query = np.random.normal(0, 1.0, dim).astype(np.float32)
    # El método nativo es flat_search(query, k)
    results = idx.flat_search(query.tolist(), 5)
    assert len(results) == 5
    print(f"✅ Search results: {results}")


def test_llm_inference_flow():
    """Prueba el flujo de inferencia completo si el modelo existe."""
    model_path = "models/production/silver_adult_steel.gaje"
    if not os.path.exists(model_path):
        pytest.skip(f"Modelo no encontrado en {model_path}")

    llm = GenomicLLM.load_genomic(model_path)
    prompt = "Hola, ¿quién eres?"
    tokens = []
    for token in llm.generate(prompt, max_new_tokens=10):
        tokens.append(token)

    assert len(tokens) > 0
    print(f"✅ Inferencia generada: {''.join(tokens)}")


if __name__ == "__main__":
    pytest.main([__file__])
