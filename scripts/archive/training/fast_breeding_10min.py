import os
import sys
import time
import numpy as np

# Asegurar que usamos el código local de 'python/'
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
)

from gaje.nn.stabilized import GenomicLLM


def main():
    model_path = "models/checkpoints/polyglot_organism.gaje"
    dataset_path = "data/datasets/hybrid_polyglot_dataset.txt"
    output_path = "models/checkpoints/mature_polyglot_organism.gaje"

    print("🧬 Iniciando Sesión de CRIANZA PROFUNDA (60 minutos)")
    print(f"[*] Cargando Organismo Políglota: {model_path}")
    model = GenomicLLM.load_genomic(model_path)

    # Aplicar h_scale balanceado (encontrado en la calibración)
    for block in model.blocks:
        block.rust_block.h_scale = 1.0

    print(f"[*] Cargando dataset híbrido: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if len(line.strip()) > 10]

    print(f"[*] Líneas totales: {len(lines)}")

    start_time = time.time()
    time_limit = 60 * 60  # 1 hora

    initial_lr = 0.005
    epoch = 0

    while (time.time() - start_time) < time_limit:
        epoch += 1
        # Decaimiento suave del learning rate por época
        current_lr = initial_lr * (0.85 ** (epoch - 1))

        print(
            f"\n🔥 Época {epoch} | LR: {current_lr:.6f} | Tiempo: {int(time.time() - start_time)}s"
        )

        total_loss = 0
        count = 0
        np.random.shuffle(lines)

        for i, text in enumerate(lines):
            if i % 20 == 0 and (time.time() - start_time) >= time_limit:
                break

            tokens = model.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) < 2:
                continue

            loss = model.rust_llm.train_on_sequence(tokens, current_lr)
            total_loss += loss
            count += 1

            if count % 50 == 0:
                elapsed = int(time.time() - start_time)
                print(
                    f"  - [{elapsed}s] Progreso: {count}/{len(lines)} | Loss: {loss:.4f}",
                    end="\r",
                )

        avg_loss = total_loss / count if count > 0 else 0
        print(f"\n[+] Época {epoch} finalizada. Loss promedio: {avg_loss:.4f}")

        # Guardar progreso cada época
        model.save(output_path)
        print("[*] Checkpoint 'Mature' actualizado.")

    print("\n" + "=" * 60)
    print("✅ CRIANZA PROFUNDA COMPLETADA")
    print(f"🚀 Organismo Maduro guardado en: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
