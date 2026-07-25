import os
import sys
import numpy as np
import argparse
from tqdm import tqdm

# Asegurar uso de código local
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
)

from gaje.nn.stabilized import GenomicLLM


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def calculate_ppl(model, text, tokenizer, max_length=128):
    """Calcula la perplejidad de un texto usando el modelo GAJE."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return None

    # Limitar longitud para evitar problemas de memoria en validación
    tokens = tokens[:max_length]

    # El forward de GAJE devuelve [seq_len, vocab_size]
    # Necesitamos las probabilidades de los tokens reales
    logits_seq = model.forward(tokens, clear_cache=True)

    # Ignoramos el último logit ya que no tenemos el target
    # Los logits en la posición i predicen el token en i+1
    logits_seq = logits_seq[:-1]
    target_tokens = tokens[1:]

    log_probs = []
    for i, target_id in enumerate(target_tokens):
        logits = logits_seq[i]
        # Softmax manual
        probs = softmax(logits)

        # Clip para evitar log(0)
        p = np.clip(probs[target_id], 1e-10, 1.0)
        log_probs.append(np.log(p))

    if not log_probs:
        return None

    avg_log_prob = np.mean(log_probs)
    ppl = np.exp(-avg_log_prob)
    return ppl


def run_phase_2_1(gaje_path, es_data_path, en_data_path, num_samples=50):
    print("🚀 Iniciando Fase 2.1: Análisis de Sesgo Lingüístico (PPL Diferencial)")
    print(f"[*] Organismo: {gaje_path}")
    print(f"[*] Dataset ES: {es_data_path}")
    print(f"[*] Dataset EN: {en_data_path}")

    # 1. Cargar GAJE
    print("[~] Cargando modelo GAJE...")
    gaje_model = GenomicLLM.load_genomic(gaje_path)
    tokenizer = gaje_model.tokenizer

    def get_ppl_for_file(path, label):
        print(f"[~] Calculando PPL para {label}...")
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if len(l.strip()) > 10]

        # Mezclar y tomar muestras
        import random

        random.seed(42)
        random.shuffle(lines)
        samples = lines[:num_samples]

        ppls = []
        for line in tqdm(samples, desc=f"Procesando {label}"):
            res = calculate_ppl(gaje_model, line, tokenizer)
            if res is not None and not np.isinf(res):
                ppls.append(res)

        return np.mean(ppls) if ppls else float("inf")

    ppl_es = get_ppl_for_file(es_data_path, "Español")
    ppl_en = get_ppl_for_file(en_data_path, "Inglés")

    # Calcular brecha
    # Brecha = |PPL_ES - PPL_EN| / min(PPL_ES, PPL_EN) * 100
    min_ppl = min(ppl_es, ppl_en)
    gap = (abs(ppl_es - ppl_en) / min_ppl) * 100

    print("\n" + "=" * 60)
    print("📊 RESULTADO FINAL FASE 2.1")
    print(f"  - PPL Promedio Español: {ppl_es:.4f}")
    print(f"  - PPL Promedio Inglés:  {ppl_en:.4f}")
    print(f"  - Brecha Lingüística:   {gap:.2f}%")
    print(f"  - KPI Meta (<20%):      {'✅ PASSED' if gap < 20 else '❌ FAILED'}")
    print("\n[Diagnóstico]:")
    if gap >= 20:
        biased_lang = "Español" if ppl_es > ppl_en else "Inglés"
        print(f"  ⚠️  Se detecta un sesgo significativo hacia el {biased_lang}.")
        print("     Esto sugiere que el vocabulario o los centroides no están")
        print("     bien equilibrados para esta gramática.")
    else:
        print("  ✨ El modelo muestra una estabilidad bilingüe saludable.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fase 2.1: Perplejidad Diferencial")
    parser.add_argument(
        "--gaje",
        type=str,
        default="models/checkpoints/smollm2_native.gaje",
        help="Ruta al archivo .gaje",
    )
    parser.add_argument("--es_data", type=str, default="data/datasets/dataset_es.txt")
    parser.add_argument(
        "--en_data", type=str, default="data/datasets/tiny_shakespeare.txt"
    )
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()

    run_phase_2_1(args.gaje, args.es_data, args.en_data, num_samples=args.samples)
