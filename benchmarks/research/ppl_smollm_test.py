import os
import sys
import numpy as np

# Añadir el directorio python al path
sys.path.append(os.path.abspath("python"))

from genomize_llm import GenomicLLM


def calculate_ppl(model, text):
    # Text only
    tokens = model.tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < 2:
        return 0.0

    print(f"[*] Evaluando PPL para {len(tokens)} tokens...")
    log_likelihoods = []

    # Procesar secuencialmente (simulando prefill + decoding)
    # Importante: No usamos clear_cache dentro del loop para mantener el contexto
    model.clear_cache()

    for i in range(len(tokens) - 1):
        target_id = tokens[i + 1]

        # Inferencia de todos los tokens hasta i para predecir i+1
        # (Aquí usamos la implementación optimizada de forward que maneja el historial)
        current_tokens = tokens[: i + 1]
        logits = model.forward(current_tokens)

        # Softmax local
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        prob_target = probs[target_id]

        log_likelihoods.append(np.log(max(prob_target, 1e-10)))

        if (i + 1) % 5 == 0:
            print(
                f"    [~] Token {i + 1}/{len(tokens) - 1} evaluado. Prob del target: {prob_target:.4f}"
            )

    avg_log_likelihood = np.mean(log_likelihoods)
    ppl = np.exp(-avg_log_likelihood)
    return ppl


def run_smollm_test():
    model_path = "/data/data/com.termux/files/home/models/gguf/smollm2-135m-q8_0.gguf"
    if not os.path.exists(model_path):
        print("❌ Modelo no encontrado.")
        return

    test_text = "The capital of France is Paris."

    print("=" * 60)
    print("🔍 TEST DE FASE SINCRONIZADA SMOL-LM2 (30 BLOQUES)")
    print("=" * 60)

    try:
        model = GenomicLLM(model_path)
        ppl = calculate_ppl(model, test_text)
        print("\n" + "=" * 45)
        print("📊 REPORTE DE PERPLEXITY (F32 REFERENCE)")
        print("=" * 45)
        print(f"✅ Perplexity (PPL): {ppl:.4f}")
        print("=" * 45)

        if ppl < 50:
            print("🔥 ESTADO: EXCELENTE. Columna vertebral sincronizada.")
        else:
            print("❌ ESTADO: INESTABLE. Aún hay ruido en la fase semántica.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_smollm_test()
