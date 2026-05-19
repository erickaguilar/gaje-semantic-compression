import numpy as np
import os
import sys

# Ajustar paths para importar el core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from gaje.core._impl import (
    calculate_mse_native, 
    calculate_cosine_similarity_native, 
    calculate_distribution_entropy_native
)

def test_semantic_metrics():
    print("🧪 GAJE TEST: Native Semantic Metrics 🧪")
    print("-" * 50)
    
    # 1. Preparar datos
    size = 1000
    vec_a = np.random.normal(0, 1, size).astype(np.float32).tolist()
    # vec_b es vec_a con un poco de ruido
    vec_b = (np.array(vec_a) + np.random.normal(0, 0.1, size)).astype(np.float32).tolist()
    
    # 2. Test MSE
    mse = calculate_mse_native(vec_a, vec_b)
    print(f"[*] MSE Nativo: {mse:.8f}")
    
    # 3. Test Cosine Similarity
    cosine = calculate_cosine_similarity_native(vec_a, vec_b)
    print(f"[*] Similitud Coseno Nativa: {cosine:.8f}")
    
    # 4. Test Entropy
    probs = np.exp(vec_a) / np.sum(np.exp(vec_a))
    entropy = calculate_distribution_entropy_native(probs.tolist())
    print(f"[*] Entropía de Distribución Nativa: {entropy:.8f}")
    
    # Verificaciones básicas
    assert mse > 0, "MSE should be positive"
    assert cosine > 0.9, "Cosine similarity should be high for small noise"
    assert entropy > 0, "Entropy should be positive"
    
    print("-" * 50)
    print("✅ MÉTRICAS NATIVAS VERIFICADAS EXITOSAMENTE")

if __name__ == "__main__":
    test_semantic_metrics()
