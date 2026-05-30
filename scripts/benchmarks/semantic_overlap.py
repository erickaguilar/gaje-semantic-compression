import os
import sys
import torch
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# Asegurar uso de código local
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
)

from gaje.nn.stabilized import GenomicLLM
from gaje.utils.metrics import calculate_top_k_overlap, calculate_jsd


def run_phase_1_1(master_id, gaje_path, prompts, k=10):
    print("🚀 Iniciando Fase 1.1: Alineación de Centroides (Top-K Overlap)")
    print(f"[*] Maestro (F32): {master_id}")
    print(f"[*] Organismo (2-bit): {gaje_path}")
    print(f"[*] Top-K: {k}")

    # 1. Cargar Maestro
    print("[~] Cargando modelo maestro (esto puede tardar si se descarga)...")
    try:
        master_model = AutoModelForCausalLM.from_pretrained(
            master_id, torch_dtype=torch.float32, device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(master_id)
    except Exception as e:
        print(f"❌ Error cargando el maestro: {e}")
        return

    # 2. Cargar GAJE
    print("[~] Cargando modelo GAJE...")
    try:
        gaje_model = GenomicLLM.load_genomic(gaje_path)
    except Exception as e:
        print(f"❌ Error cargando GAJE: {e}")
        return

    overlaps = []
    jsds = []

    print(f"\n| {'Prompt':<25} | {'Overlap @'+str(k):<12} | {'JSD':<8} |")
    print("|" + "-" * 27 + "|" + "-" * 14 + "|" + "-" * 10 + "|")

    for prompt in prompts:
        # A. Logits Maestro
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = master_model(**inputs)
            master_logits = outputs.logits[0, -1, :].float().numpy()

        # B. Logits GAJE
        # Usamos el mismo tokenizador del maestro para asegurar alineación de IDs
        gaje_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        # GAJE forward devuelve [seq_len, vocab_size]
        gaje_logits_seq = gaje_model.forward(gaje_tokens, clear_cache=True)
        gaje_logits = gaje_logits_seq[-1]

        # Sincronizar tamaños si hay desajuste
        min_vocab = min(len(master_logits), len(gaje_logits))
        m_logits = master_logits[:min_vocab]
        g_logits = gaje_logits[:min_vocab]

        # C. Métricas
        overlap = calculate_top_k_overlap(m_logits, g_logits, k=k)

        # Para JSD necesitamos distribuciones de probabilidad
        m_prob = softmax(m_logits)
        g_prob = softmax(g_logits)
        jsd = calculate_jsd(m_prob, g_prob)

        overlaps.append(overlap)
        jsds.append(jsd)

        display_prompt = (prompt[:22] + "...") if len(prompt) > 25 else prompt
        print(f"| {display_prompt:<25} | {overlap*100:>10.2f}% | {jsd:>8.4f} |")

    avg_overlap = np.mean(overlaps) * 100
    avg_jsd = np.mean(jsds)

    print("\n" + "=" * 60)
    print("📊 RESULTADO FINAL FASE 1.1")
    print(f"  - Top-{k} Overlap Promedio: {avg_overlap:.2f}%")
    print(f"  - JSD Promedio: {avg_jsd:.4f}")
    print(f"  - KPI Meta (>65%): {'✅ PASSED' if avg_overlap >= 65 else '❌ FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fase 1.1: Validación de Overlap Semántico"
    )
    parser.add_argument(
        "--master",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        help="ID o ruta del modelo maestro",
    )
    parser.add_argument(
        "--gaje",
        type=str,
        default="models/checkpoints/smollm2_native.gaje",
        help="Ruta al archivo .gaje",
    )
    parser.add_argument("--k", type=int, default=10, help="Valor de K para el overlap")
    args = parser.parse_args()

    prompts_es = [
        "El capital de España es",
        "La ley de la gravedad fue descubierta por",
        "En un lugar de la Mancha, de cuyo nombre",
        "Para programar en Rust, primero debes",
        "El ADN es la molécula que contiene",
        "La inteligencia artificial neuromórfica se basa en",
        "Si mezclamos azul y amarillo obtenemos",
        "El primer hombre en pisar la Luna fue",
        "La fotosíntesis es el proceso por el cual",
        "El lenguaje de programación Python es conocido por",
    ]

    run_phase_1_1(args.master, args.gaje, prompts_es, k=args.k)
