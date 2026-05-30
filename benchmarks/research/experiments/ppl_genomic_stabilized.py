import os
import sys
import numpy as np

# Añadir el directorio python al path
sys.path.append(os.path.abspath("python"))

from stabilized_genomic_llm import GenomicLLM


def calculate_ppl(model, text):
    tokens = model.tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < 2:
        return 0.0

    print(f"[*] Evaluando PPL Genómico (2-bit) para {len(tokens)} tokens...")

    logits_all = model.forward(tokens)
    log_likelihoods = []

    for i in range(len(tokens) - 1):
        target_id = tokens[i + 1]
        logits = logits_all[i]

        # Softmax local
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        prob_target = probs[target_id]

        log_likelihoods.append(np.log(max(prob_target, 1e-10)))

        if (i + 1) % 5 == 0:
            print(
                f"    [~] Token {i+1}/{len(tokens)-1} evaluado. Prob del target: {prob_target:.4f}"
            )

    avg_log_likelihood = np.mean(log_likelihoods)
    ppl = np.exp(-avg_log_likelihood)
    return ppl


def run_test():
    model_path = "/data/data/com.termux/files/home/models/gguf/smollm2-135m-q8_0.gguf"
    if not os.path.exists(model_path):
        print("❌ Modelo no encontrado.")
        return

    # Usamos 30 bloques por defecto si no se especifica
    num_blocks = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    test_text = "The capital of France is Paris."

    print("=" * 60)
    print(f"🧬 TEST DE ESTABILIZACIÓN GENÓMICA (2-BIT) - {num_blocks} BLOQUES")
    print("=" * 60)

    try:
        model = GenomicLLM(model_path, num_blocks=num_blocks)
        ppl = calculate_ppl(model, test_text)
        print("\n" + "=" * 45)
        print("📊 REPORTE DE PERPLEXITY (GAJE 2-BIT)")
        print("=" * 45)
        print(f"✅ Perplexity (PPL): {ppl:.4f}")
        print("=" * 45)

        if ppl < 100:
            print("🔥 ESTADO: ÉXITO. El metabolismo de 2 bits es estable.")
        else:
            print("❌ ESTADO: DEGRADACIÓN. Se requiere ajuste de centroides.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_test()
