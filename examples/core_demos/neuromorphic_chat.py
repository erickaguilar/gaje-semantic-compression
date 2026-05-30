import os
import sys
import argparse

# Asegurar que usamos el código local de 'python/'
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
)

from gaje.nn.stabilized import GenomicLLM


def main():
    parser = argparse.ArgumentParser(description="🧬 GAJE PROTOCOL: NEUROMORPHIC CHAT")
    parser.add_argument(
        "--model",
        type=str,
        default="models/silver_adult_anchored.gaje",
        help="Path to the GAJE model",
    )
    parser.add_argument("--prompt", type=str, default="Hola", help="Prompt de usuario")
    parser.add_argument(
        "--steps", type=int, default=32, help="Neuromorphic simulation steps"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.8, help="Spiking threshold"
    )
    parser.add_argument("--decay", type=float, default=0.9, help="Spiking decay")
    args = parser.parse_args()

    print("🧠 GAJE: NEUROMORPHIC DECODER (Spiking Inference)")
    print("=" * 60)

    if not os.path.exists(args.model):
        print(f"❌ Error: Modelo no encontrado en {args.model}")
        return

    print(f"[*] Cargando Organismo: {args.model}")
    model = GenomicLLM.load_genomic(args.model)

    print(f"\n👤 Usuario: {args.prompt}")
    print("\n🤖 GAJE (Neuromórfico): ", end="", flush=True)

    for token in model.generate(
        args.prompt,
        max_new_tokens=30,
        use_spiking=True,
        spiking_steps=args.steps,
        spiking_threshold=args.threshold,
        spiking_decay=args.decay,
    ):
        print(token, end="", flush=True)
    print("\n\n[!] Proceso finalizado.")


if __name__ == "__main__":
    main()
