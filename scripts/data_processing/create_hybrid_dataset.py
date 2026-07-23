import random


def create_hybrid_dataset():
    en_path = "data/datasets/tiny_shakespeare.txt"
    es_path = "data/datasets/dataset_born_2000.txt"
    output_path = "data/datasets/hybrid_polyglot_dataset.txt"

    print("🧬 Generando Dataset Híbrido (EN/ES)...")

    with open(en_path, "r", encoding="utf-8") as f:
        en_lines = [line.strip() for line in f.readlines() if len(line.strip()) > 20]

    with open(es_path, "r", encoding="utf-8") as f:
        es_lines = [line.strip() for line in f.readlines() if len(line.strip()) > 20]

    print(f"[*] Líneas Inglés: {len(en_lines)}")
    print(f"[*] Líneas Español: {len(es_lines)}")

    # Balancear y mezclar
    # Tomamos una cantidad similar para que no haya sesgo
    min_len = min(len(en_lines), len(es_lines))
    hybrid_lines = en_lines[:min_len] + es_lines[:min_len]
    random.shuffle(hybrid_lines)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in hybrid_lines:
            f.write(line + "\n")

    print(f"✅ Dataset Híbrido creado en: {output_path} ({len(hybrid_lines)} líneas)")


if __name__ == "__main__":
    create_hybrid_dataset()
