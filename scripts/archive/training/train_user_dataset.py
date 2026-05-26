import os
import sys
import argparse
import time

# Aseguramos que usamos el paquete local
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM
from gaje.nn.trainer import GenomicTrainer
from gaje.processing.pipeline import DatasetProcessor


def main():
    parser = argparse.ArgumentParser(
        description="🧬 GAJE PROTOCOL: CONTINUOUS EDGE LEARNING"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Directorio del modelo genómico (.gaje) a refinar",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Ruta al dataset personalizado (.jsonl o .txt)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Épocas de refinamiento (default: 5)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        help="Learning rate (default: 0.0005 - Mantener bajo para evitar olvido catastrófico)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directorio de salida (por defecto sobrescribe el modelo original)",
    )
    args = parser.parse_args()

    print("🧬 GAJE PROTOCOL: CONTINUOUS EDGE LEARNING v1.0")
    print("=" * 60)

    # 1. Cargar Dataset Personalizado
    try:
        dataset = DatasetProcessor.load_dataset(args.dataset)
    except Exception as e:
        print(f"[!] Error al cargar dataset: {e}")
        return

    if not dataset:
        print("[!] El dataset está vacío. Abortando.")
        return

    # 2. Cargar Modelo Existente
    print(f"\n[*] Despertando organismo genómico: {args.model}")
    if not os.path.exists(args.model):
        print(f"[!] Error: No se encontró el directorio del modelo {args.model}")
        return

    try:
        # Cargamos el modelo existente desde el directorio especificado
        if args.model.endswith(".gaje") or os.path.exists(
            os.path.join(args.model, "model.gaje")
        ):
            llm = GenomicLLM.load_genomic(args.model)
        else:
            llm = GenomicLLM(model_path=args.model)

        print("[*] Organismo cargado.")
        if hasattr(llm, "tokenizer"):
            print(f"[*] Vocabulario: {len(llm.tokenizer)}")
    except Exception as e:
        print(f"[!] Error fatal al cargar el modelo: {e}")
        import traceback

        traceback.print_exc()
        return

    # 3. Refinamiento (Entrenamiento Continuo)
    print(
        f"\n[*] Iniciando fase de refinamiento epigenético (Learning Rate: {args.lr})..."
    )
    trainer = GenomicTrainer(llm, lr=args.lr)

    start_time = time.time()
    trainer.fit(dataset, epochs=args.epochs)
    duration = time.time() - start_time

    print(f"\n[*] Refinamiento finalizado en {duration:.2f} segundos.")

    # 4. Prueba rápida de identidad
    test_prompt = "Usuario: ¿Qué has aprendido?\nAsistente:"
    print("\n[*] Validación de Refinamiento:")
    print(f"    - Prompt: '{test_prompt}'")
    print("    - Respuesta: ", end="", flush=True)
    try:
        for token_text in llm.generate(test_prompt, max_new_tokens=20, temperature=0.2):
            print(token_text, end="", flush=True)
    except Exception as e:
        print(f"[Error en generación: {e}]")
    print("\n")

    # 5. Guardar la evolución
    out_dir = args.output if args.output else args.model
    print(f"\n[*] Guardando evolución en: {out_dir}")

    if out_dir.endswith(".gaje"):
        parent_dir = os.path.dirname(out_dir)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
    else:
        os.makedirs(out_dir, exist_ok=True)

    try:
        llm.save(out_dir)
        print("[+] Mutación consolidada exitosamente en la base de datos genómica.")
    except Exception as e:
        print(f"[!] Error al guardar la evolución: {e}")


if __name__ == "__main__":
    main()
