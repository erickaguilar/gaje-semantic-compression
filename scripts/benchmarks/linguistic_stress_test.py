import os
import sys
import numpy as np
import argparse

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def calculate_ppl(model, text, tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return None
    if hasattr(tokens, "ids"):
        tokens = tokens.ids

    logits_seq = model.forward(tokens, clear_cache=True)
    logits_seq = logits_seq[:-1]
    target_tokens = tokens[1:]

    log_probs = []
    for i, target_id in enumerate(target_tokens):
        probs = softmax(logits_seq[i])
        p = np.clip(probs[target_id], 1e-10, 1.0)
        log_probs.append(np.log(p))

    return np.exp(-np.mean(log_probs)) if log_probs else None


def run_stress_test(model_path):
    print("🔬 [DIAGNÓSTICO] Iniciando Stress Test Lingüístico")
    print(f"[*] Modelo: {model_path}")

    model = GenomicLLM.load_genomic(model_path)
    tokenizer = model.tokenizer

    strata = {
        "NIVEL 1: Identidad/Simple": "data/evaluation/strata/simple_identity.txt",
        "NIVEL 2: Periodístico/Medio": "data/evaluation/strata/journalistic_medium.txt",
        "NIVEL 3: Técnico/Abstracto": "data/evaluation/strata/technical_abstract.txt",
    }

    results = {}
    print("-" * 50)
    for label, path in strata.items():
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        ppl = calculate_ppl(model, text, tokenizer)
        results[label] = ppl
        print(f"📊 {label:<25} | PPL: {ppl:>10.2f}")
    print("-" * 50)

    # Análisis de Resultados
    ppls = list(results.values())
    variance = np.std(ppls) / np.mean(ppls)

    print("\n[VERDICTO TÉCNICO]")
    if variance < 0.2:
        print("🔴 COLAPSO UNIFORME: El modelo falla por igual en todos los niveles.")
        print(
            "   -> Causa: Falta de cobertura en los datos de calibración (Ruta A necesaria)."
        )
    elif (
        results["NIVEL 3: Técnico/Abstracto"] > results["NIVEL 1: Identidad/Simple"] * 5
    ):
        print(
            "🟡 COLAPSO POR RESOLUCIÓN: El modelo falla drásticamente al subir la complejidad."
        )
        print(
            "   -> Causa: La ε-net de 2 bits no tiene resolución para el ancho de banda técnico (Ruta B necesaria)."
        )
    else:
        print("🟢 ESTABILIDAD RELATIVA: El modelo mantiene la proporción.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="models/production/smollm2_mixed_v1.gaje"
    )
    args = parser.parse_args()
    run_stress_test(args.model)
