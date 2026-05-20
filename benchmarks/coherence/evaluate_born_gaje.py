import os
import sys
import numpy as np
import time

# Ensure we use the local package
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM

def calculate_ppl(llm, text):
    """Calculates Perplexity: exp(-1/N * sum(log P(token_i | tokens_<i)))"""
    tokens = llm.tokenizer.encode(text, add_special_tokens=False)
    n_tokens = len(tokens)
    if n_tokens < 2: return 0.0

    log_likelihoods = []
    
    # Prefix processing
    logits_all = llm.forward(tokens[:1], clear_cache=True)
    
    for i in range(1, n_tokens):
        logits = logits_all[-1]
        target_token_id = tokens[i]
        
        # Softmax
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        
        prob_target = probs[target_token_id]
        log_likelihoods.append(np.log(max(prob_target, 1e-10)))
        
        # Next token
        logits_all = llm.forward([tokens[i]], clear_cache=False)

    avg_log_likelihood = np.mean(log_likelihoods)
    return np.exp(-avg_log_likelihood)

def main():
    model_path = "models/born_genomic_qwen/model_pruned.gaje"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    print(f"🧬 Loading Born-Genomic Model: {model_path}")
    llm = GenomicLLM.load_genomic(model_path)
    
    # Test dataset (same as training)
    dataset = [
        "El protocolo GAJE es nativo.",
        "El protocolo GAJE comprime semántica.",
        "GAJE utiliza ADN en lugar de pesos.",
        "Qwen es la arquitectura base."
    ]
    
    print("\n📊 Coherence Report (Perplexity):")
    print("-" * 40)
    total_ppl = 0
    for i, text in enumerate(dataset):
        ppl = calculate_ppl(llm, text)
        print(f"  {i+1}. '{text[:30]}...' -> PPL: {ppl:.4f}")
        total_ppl += ppl
    
    avg_ppl = total_ppl / len(dataset)
    print("-" * 40)
    print(f"  AVERAGE PPL: {avg_ppl:.4f}")
    
    if avg_ppl < 5:
        print("\n✅ EXCELLENT: The model has successfully memorized the semantic DNA.")
    elif avg_ppl < 20:
        print("\n⚠️ GOOD: The model captures the patterns but with some noise.")
    else:
        print("\n❌ POOR: High perplexity, the model is not coherent.")

    # Entropy check for a new prompt
    prompt = "El protocolo GAJE"
    tokens = llm.tokenizer.encode(prompt, add_special_tokens=False)
    logits = llm.forward(tokens, clear_cache=True)[-1]
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    print(f"\n📈 Entropy Analysis for '{prompt}':")
    print(f"  - Shannon Entropy: {entropy:.4f} bits")
    print(f"  - Top predicted: '{llm.tokenizer.decode([np.argmax(logits)])}'")

if __name__ == "__main__":
    main()
