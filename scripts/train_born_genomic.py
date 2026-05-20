import os
import sys
import argparse
import time

# Ensure we use the local package first
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM
from gaje.nn.configs import get_config
from gaje.nn.trainer import GenomicTrainer

def main():
    parser = argparse.ArgumentParser(description="🧬 GAJE PROTOCOL: BORN-GENOMIC TRAINING")
    parser.add_argument("--arch", type=str, default="qwen2", help="Architecture config to use (default: qwen2)")
    parser.add_argument("--blocks", type=int, default=2, help="Number of transformer blocks (default: 2)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    args = parser.parse_args()

    print(f"🧬 GAJE PROTOCOL: BORN-GENOMIC TRAINING v0.7.0")
    print("=" * 60)

    # 1. Configuración de Arquitectura
    config = get_config(args.arch)
    print(f"[*] Configuración cargada: {config.name}")

    # 2. Inicialización Born-Genomic
    # No pasamos model_path, así que el motor inicializa tensores genómicos con centroides aleatorios.
    llm = GenomicLLM(model_path=None, num_blocks=args.blocks, config=config)
    print(f"[*] Modelo inicializado con {args.blocks} bloques ocultos y vocabulario de {len(llm.tokenizer)}.")

    # 3. Micro-Dataset de prueba (Para probar Overfitting / Memorización)
    dataset = [
        "El protocolo GAJE es nativo.",
        "El protocolo GAJE comprime semántica.",
        "GAJE utiliza ADN en lugar de pesos.",
        "Qwen es la arquitectura base."
    ]
    print("\n[*] Dataset de prueba:")
    for text in dataset:
        print(f"    - '{text}'")

    # 4. Iniciar Entrenamiento
    trainer = GenomicTrainer(llm, lr=args.lr)
    
    start_time = time.time()
    trainer.fit(dataset, epochs=args.epochs)
    duration = time.time() - start_time
    
    print(f"\n[*] Entrenamiento finalizado en {duration:.2f} segundos.")
    
    # 5. Generación de prueba (Inferencia Zero-Shot tras el entrenamiento)
    prompt = "El protocolo GAJE"
    print(f"\n[*] Generando texto a partir de: '{prompt}'")
    print("🤖 GAJE: ", end="", flush=True)
    
    for token_text in llm.generate(prompt, max_new_tokens=10, temperature=0.1):
        print(token_text, end="", flush=True)
    
    print("\n")

    # 5.1 Cálculo de Perplexity post-entrenamiento
    import numpy as np
    print("[*] Calculando Perplexity sobre el dataset de entrenamiento...")
    total_ppl = 0
    for text in dataset:
        tokens = llm.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 2: continue
        llm.rust_llm.clear_cache()
        log_likelihoods = []
        # Inferencia secuencial
        logits_all = llm.forward(tokens[:-1], clear_cache=True)
        for i, target_id in enumerate(tokens[1:]):
            logits = logits_all[i]
            probs = np.exp(logits - np.max(logits))
            probs /= probs.sum()
            prob_target = probs[target_id]
            log_likelihoods.append(np.log(max(prob_target, 1e-10)))
        
        ppl = np.exp(-np.mean(log_likelihoods))
        total_ppl += ppl
        print(f"    - PPL para '{text[:20]}...': {ppl:.4f}")
    
    print(f"[*] Perplexity Media: {total_ppl / len(dataset):.4f}")
    
    # Pre-save logits for consistency check
    prompt_test = "El protocolo"
    tokens_test = llm.tokenizer.encode(prompt_test, add_special_tokens=False)
    logits_before = llm.forward(tokens_test, clear_cache=True)[-1]

    # 6. Guardar el organismo
    out_dir = "models/born_genomic_qwen"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "model.gaje")
    llm.save(out_dir)

    # 7. Recargar y verificar consistencia
    print("\n[*] Verificando consistencia tras guardado...")
    llm_reloaded = GenomicLLM.load_genomic(out_file)
    logits_after = llm_reloaded.forward(tokens_test, clear_cache=True)[-1]
    
    from gaje.core import _impl as core
    mse = core.calculate_mse_native(logits_before.tolist(), logits_after.tolist())
    cos_sim = core.calculate_cosine_similarity_native(logits_before.tolist(), logits_after.tolist())
    
    print(f"    - MSE Logits: {mse:.8f}")
    print(f"    - Similitud Coseno: {cos_sim:.8f}")
    
    if cos_sim > 0.999:
        print("✅ CONSISTENCIA OK: El modelo se guardó y cargó perfectamente.")
    else:
        print("❌ ERROR DE CONSISTENCIA: El modelo cargado difiere del original.")
        # Analizar diferencias en parámetros clave
        print(f"    - Original vocab size: {len(llm.tokenizer)}")
        print(f"    - Reloaded vocab size: {len(llm_reloaded.tokenizer)}")
        print(f"    - Original blocks: {len(llm.blocks)}")
        print(f"    - Reloaded blocks: {len(llm_reloaded.blocks)}")

if __name__ == "__main__":
    main()
