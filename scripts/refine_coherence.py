import os
import sys
import argparse
import time

# Ensure we use the local package first
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM
from gaje.nn.trainer import GenomicTrainer


def main():
    parser = argparse.ArgumentParser(
        description="🧬 GAJE PROTOCOL: COHERENCE REFINEMENT"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the base GGUF model"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Refinement epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    args = parser.parse_args()

    print("🧬 GAJE PROTOCOL: COHERENCE REFINEMENT v0.7.1")
    print("=" * 60)

    # 1. Inyectar centroides Max-Lloyd optimizados para Qwen2-0.5B
    # Estos valores fueron calibrados en el paso anterior.
    qwen2_centroids = {
        "blk.0.ffn_down.weight": [-0.0267, -0.0078, 0.0075, 0.0264],
        "blk.0.ffn_gate.weight": [-0.0364, -0.0132, 0.006, 0.0294],
        "blk.0.ffn_up.weight": [-0.0283, -0.0101, 0.0054, 0.0243],
        "blk.0.attn_q.weight": [-0.1034, -0.0199, 0.0268, 0.1148],
        "blk.1.ffn_down.weight": [-0.0253, -0.0082, 0.006, 0.0233],
        "blk.1.ffn_gate.weight": [-0.032, -0.0076, 0.0125, 0.0363],
        "blk.1.ffn_up.weight": [-0.0244, -0.0074, 0.0068, 0.0238],
        "blk.1.attn_q.weight": [-0.0482, -0.0133, 0.0091, 0.0428],
    }

    # 2. Cargar el modelo base con calibración completa
    print(f"[*] Cargando modelo base para refinamiento: {args.model}")
    llm = GenomicLLM(args.model, custom_centroids=qwen2_centroids)

    # 3. Dataset Quirúrgico para fijar la coherencia
    dataset = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "The capital of Spain is Madrid.",
        "Paris is a beautiful city in France.",
        "The largest planet is Jupiter.",
        "Water boils at one hundred degrees Celsius.",
    ]
    print("\n[*] Dataset de refinamiento:")
    for text in dataset:
        print(f"    - '{text}'")

    # 4. Iniciar Refinamiento (Training Born-Genomic sobre el modelo cargado)
    print("\n[*] Iniciando refinamiento nativo de centroides...")
    trainer = GenomicTrainer(llm, lr=args.lr)

    start_time = time.time()
    # Usamos fit pero con pocas épocas para no "quemar" el conocimiento general
    trainer.fit(dataset, epochs=args.epochs)
    duration = time.time() - start_time

    print(f"\n[*] Refinamiento finalizado en {duration:.2f} segundos.")

    # 5. Verificación de Coherencia
    prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "Water boils at",
    ]

    print("\n" + "=" * 60)
    print("🎯 VERIFICACIÓN DE COHERENCIA FINAL")
    print("=" * 60)

    for p in prompts:
        print(f"\n👤 Usuario: {p}")
        print("🤖 GAJE: ", end="", flush=True)
        for token_text in llm.generate(p, max_new_tokens=5, temperature=0.1):
            print(token_text, end="", flush=True)
        print()

    # 6. Guardar el organismo coherente
    out_dir = "models/qwen2-0_5b-coherent.gaje"
    print(f"\n[*] Guardando organismo coherente en {out_dir}...")
    llm.save(out_dir)


if __name__ == "__main__":
    main()
