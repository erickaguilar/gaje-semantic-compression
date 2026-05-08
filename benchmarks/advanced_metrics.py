import numpy as np
from typing import List, Set

def calculate_top_k_overlap(logits_a: np.ndarray, logits_b: np.ndarray, k: int = 10) -> float:
    """
    Calcula la proporción de tokens compartidos en el top-k entre dos conjuntos de logits.
    """
    top_k_a = set(np.argsort(logits_a)[-k:])
    top_k_b = set(np.argsort(logits_b)[-k:])
    overlap = len(top_k_a.intersection(top_k_b))
    return overlap / k

def calculate_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon Divergence entre dos distribuciones de probabilidad.
    """
    # jensenshannon devuelve la raíz cuadrada (JS distance) por defecto en versiones recientes
    # Queremos la divergencia (cuadrado de la distancia) o simplemente la distancia.
    # Usaremos la implementación manual para control total.
    m = 0.5 * (p + q)
    def kl_div(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))
    
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)

def calculate_attention_entropy(attention_weights: np.ndarray) -> float:
    """
    Calcula la entropía de los pesos de atención para detectar colapso (over-focusing).
    attention_weights: [num_heads, seq_len, seq_len]
    """
    # Evitar log(0)
    att = np.clip(attention_weights, 1e-12, 1.0)
    entropy = -np.sum(att * np.log(att), axis=-1)
    return np.mean(entropy)

def calculate_activation_drift(act_orig: np.ndarray, act_gen: np.ndarray) -> float:
    """
    Mide la desviación (drift) de las activaciones entre la capa original y la genómica.
    """
    mse = np.mean((act_orig - act_gen)**2)
    norm_orig = np.linalg.norm(act_orig)
    return mse / (norm_orig + 1e-12)

def calculate_token_repetition_score(tokens: List[int], n: int = 4) -> float:
    """
    Detecta bucles autoregresivos contando n-gramas repetidos.
    """
    if len(tokens) < n:
        return 0.0
    
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    unique_ngrams = set(ngrams)
    
    if not ngrams:
        return 0.0
        
    return 1.0 - (len(unique_ngrams) / len(ngrams))

def calculate_semantic_consistency(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    Similitud coseno entre embeddings de oraciones para validar coherencia narrativa.
    """
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(embedding_a, embedding_b) / (norm_a * norm_b)

if __name__ == "__main__":
    # Test simple de las métricas
    l1 = np.random.randn(100)
    l2 = l1 + np.random.normal(0, 0.1, 100)
    
    p1 = np.exp(l1) / np.sum(np.exp(l1))
    p2 = np.exp(l2) / np.sum(np.exp(l2))
    
    print(f"Top-10 Overlap: {calculate_top_k_overlap(l1, l2, 10)}")
    print(f"JSD: {calculate_jsd(p1, p2):.6f}")
    print(f"Activation Drift: {calculate_activation_drift(l1, l2):.6f}")
    print(f"Repetition Score (test): {calculate_token_repetition_score([1, 2, 3, 1, 2, 3, 1, 2, 3], 3):.4f}")
