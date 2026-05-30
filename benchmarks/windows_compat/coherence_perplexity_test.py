"""
Benchmark de Perplejidad (PPL) adaptado para Windows.
Usa el modelo Qwen2-0.5B-Instruct GGUF local.
"""
import os
import sys
import numpy as np
import time

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, os.path.join(project_root, "python"))

from gaje.nn.stabilized import GenomicLLM


class PerplexityValidator:
    def __init__(self, model_path, num_blocks=4):
        print(f"[DNA] Inicializando Validador de Perplexity (Bloques: {num_blocks})")
        self.llm = GenomicLLM(model_path, num_blocks=num_blocks)

    def calculate_ppl(self, text):
        """
        Calcula la Perplexity: exp(-1/N * sum(log P(token_i | tokens_<i)))
        """
        tokens = self.llm.tokenizer.encode(text, add_special_tokens=False)
        n_tokens = len(tokens)

        if n_tokens < 2:
            return 0.0

        print(f"[*] Analizando {n_tokens} tokens para PPL...")
        log_likelihoods = []

        # Procesamos el prefijo completo y luego evaluamos cada siguiente token
        for i in range(1, n_tokens):
            context_tokens = tokens[:i]

            # Forward pass con cache (solo el ultimo token tras el primero)
            if i == 1:
                logits_all = self.llm.forward(context_tokens, clear_cache=True)
            else:
                logits_all = self.llm.forward([tokens[i - 1]], clear_cache=False)

            logits = logits_all[-1]
            target_token_id = tokens[i]

            # Softmax
            probs = np.exp(logits - np.max(logits))
            probs /= probs.sum()

            # Verosimilitud del token real
            prob_target = probs[target_token_id]
            log_likelihoods.append(np.log(max(prob_target, 1e-10)))

            if i % 10 == 0:
                print(
                    f"    [~] Progreso: {i}/{n_tokens} tokens evaluados...", flush=True
                )

        avg_log_likelihood = np.mean(log_likelihoods)
        ppl = np.exp(-avg_log_likelihood)
        return ppl


def run_ppl_test():
    model_path = os.path.join(project_root, "models", "Qwen2-0.5B-Instruct-Q8_0.gguf")
    if not os.path.exists(model_path):
        print("[ERROR] Modelo Qwen2 no encontrado.")
        print("  Ejecuta: python scripts/download_hf_model.py")
        return

    validator = PerplexityValidator(model_path, num_blocks=4)

    # Texto de prueba
    test_text = (
        "El Protocolo GAJE utiliza estructuras biologicas para comprimir informacion semantica. "
        "Al transformar vectores de alta dimension en cadenas de ADN de dos bits, logramos "
        "una eficiencia sin precedentes en dispositivos moviles con memoria limitada."
    )

    print(f"\n[TEXT] Texto de Validacion: '{test_text[:50]}...'")

    start_time = time.time()
    ppl_score = validator.calculate_ppl(test_text)
    duration = time.time() - start_time

    print("\n" + "=" * 50)
    print("  REPORTE DE PERPLEXITY (GAJE 2-BIT + Qwen2-0.5B)")
    print("=" * 50)
    print(f"  Perplexity (PPL):     {ppl_score:.4f}")
    print(f"  Tiempo de Evaluacion: {duration:.2f} s")
    print(f"  Tokens Procesados:    {len(validator.llm.tokenizer.encode(test_text))}")
    print("=" * 50)

    if ppl_score < 50:
        print("  ESTADO: EXCELENTE. El modelo mantiene coherencia gramatical.")
    elif ppl_score < 150:
        print("  ESTADO: ACEPTABLE. Hay ruido predictivo, pero el sentido se conserva.")
    else:
        print("  ESTADO: CRITICO. El modelo ha perdido la logica secuencial.")


if __name__ == "__main__":
    run_ppl_test()
