import os
import sys
import numpy as np
import time

# Asegurar que usamos el código local de 'python/'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from gaje.nn.stabilized import GenomicLLM

def calculate_ppl(llm, text):
    """
    Calcula la Perplexity: exp(-1/N * sum(log P(token_i | tokens_<i)))
    """
    tokens = llm.tokenizer.encode(text, add_special_tokens=False)
    if hasattr(tokens, 'ids'): # En caso de que sea un tokenizador de la lib 'tokenizers'
        tokens = tokens.ids
        
    n_tokens = len(tokens)

    if n_tokens < 2:
        return 0.0

    print(f"[*] Analizando {n_tokens} tokens para PPL...")
    log_likelihoods = []

    # Reset cache
    llm.rust_llm.clear_cache()

    for i in range(n_tokens - 1):
        # Forward pass del token actual
        logits = llm.rust_llm.forward(tokens[i], False)
        
        # El target es el SIGUIENTE token
        target_token_id = tokens[i+1]

        # Softmax para obtener probabilidades
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()

        # Verosimilitud del token real
        prob_target = probs[target_token_id]
        log_likelihoods.append(np.log(max(prob_target, 1e-10)))

        if (i + 1) % 10 == 0:
            print(f"    [~] Progreso: {i+1}/{n_tokens-1} tokens evaluados...", flush=True)

    avg_log_likelihood = np.mean(log_likelihoods)
    ppl = np.exp(-avg_log_likelihood)
    return ppl

def main():
    model_path = "models/checkpoints/smollm2_f16_distilled.gaje"
    # model_path = "models/checkpoints/mature_polyglot_organism.gaje"
        
    if not os.path.exists(model_path):
        print(f"❌ No se encontró ningún modelo en {model_path}")
        return

    print(f"🧬 Cargando modelo para prueba de coherencia: {model_path}")
    llm = GenomicLLM.load_genomic(model_path)
    
    # Probar con diferentes RoPE base si la perplejidad es muy alta
    original_rope = llm.rope_base
    for forced_rope in [original_rope, 10000.0, 100000.0, 1000000.0]:
        print(f"\n--- Probando con RoPE Base: {forced_rope} ---")
        llm.rope_base = forced_rope
        # Actualizar en el lado de Rust también si es posible
        for block in llm.blocks:
            if hasattr(block.rust_block, "attn"):
                block.rust_block.attn.rope_base = forced_rope

        # Solo probar el primer texto para ahorrar tiempo
        test_text = "El sistema GAJE permite la compresión semántica de modelos de lenguaje."
        print(f"📝 Evaluando: '{test_text}'")
        start_time = time.time()
        try:
            ppl_score = calculate_ppl(llm, test_text)
            print(f"✅ Perplexity (PPL): {ppl_score:.4f}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
