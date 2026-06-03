import os
import sys
import numpy as np
import pytest
import time

sys.path.append(os.path.abspath("python"))
from gaje.core import _impl as dna_core
from gaje.nn.stabilized import GenomicLLM

def calculate_ppl_from_logits(logits, target_id):
    """Calcula el log-likelihood de un token específico."""
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    return np.log(max(probs[target_id], 1e-10))

def test_perplexity_real_text():
    """Calcula la perplejidad sobre un texto técnico real."""
    model_path = "models/gguf/qwen2-0_5b-q8_0.gguf"
    if not os.path.exists(model_path):
        pytest.skip("Modelo no encontrado para test de PPL")
        
    llm = GenomicLLM(model_path, num_blocks=4)
    test_text = "The DNA semantic compression protocol optimizes neural networks for edge devices."
    tokens = llm.tokenizer.encode(test_text, add_special_tokens=False)
    
    if len(tokens) < 5:
        pytest.skip("Texto demasiado corto para PPL")
        
    log_likelihoods = []
    # Usamos el forward pass del LLM (que debería ser eficiente)
    for i in range(1, len(tokens)):
        context = tokens[:i]
        target = tokens[i]
        logits = llm.forward(context)[-1]
        log_likelihoods.append(calculate_ppl_from_logits(logits, target))
        
    ppl = np.exp(-np.mean(log_likelihoods))
    print(f"Perplexity (Real Text): {ppl:.4f}")
    assert ppl < 200, f"Perplejidad demasiado alta: {ppl}"

def test_simulated_cloning_impact():
    """Simula el impacto de la 'Clonación de Anclas' en la reducción de PPL."""
    vocab_size = 1000
    logits_base = np.random.normal(0, 1, vocab_size)
    logits_base[10] = 10.0 # Token objetivo con alta confianza
    
    # Escenario A: Ruido alto (Sin anclas)
    logits_noisy = logits_base + np.random.normal(0, 3.0, vocab_size)
    ppl_noisy = np.exp(-calculate_ppl_from_logits(logits_noisy, 10))
    
    # Escenario B: Ruido reducido (Con anclas clonadas)
    logits_cloned = logits_base + np.random.normal(0, 1.0, vocab_size)
    ppl_cloned = np.exp(-calculate_ppl_from_logits(logits_cloned, 10))
    
    print(f"PPL Noisy: {ppl_noisy:.2f} | PPL Cloned: {ppl_cloned:.2f}")
    assert ppl_cloned < ppl_noisy, "La clonación de anclas debería reducir la perplejidad"

if __name__ == "__main__":
    pytest.main([__file__])
