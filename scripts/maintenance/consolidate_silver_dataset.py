import os


def consolidate_datasets(output_path="data/datasets/consolidated_silver_dataset.txt"):
    print("🧬 Iniciando Consolidación de Datos para Silver Fetus...")

    # Fuentes locales identificadas
    sources = [
        "data/datasets/tiny_shakespeare.txt",
        "data/datasets/hybrid_polyglot_dataset.txt",
        "data/datasets/dataset_born_2000.txt",
        "data/datasets/dataset_es_ext.txt",
        "data/datasets/expert_rust.txt",
        "data/datasets/coherence_es.txt",
    ]

    total_size = 0
    with open(output_path, "w", encoding="utf-8") as outfile:
        for src in sources:
            if os.path.exists(src):
                print(f"[*] Procesando: {src} ({os.path.getsize(src) / 1024:.2f} KB)")
                with open(src, "r", encoding="utf-8") as infile:
                    content = infile.read()
                    # Limpieza básica: asegurar que no haya nulos y normalizar saltos de línea
                    content = content.replace("\x00", "")
                    outfile.write(content)
                    outfile.write("\n\n")  # Separador de contexto
                    total_size += len(content.encode("utf-8"))
            else:
                print(f"⚠️  Fuente no encontrada: {src}")

    print(f"✅ Dataset consolidado creado en: {output_path}")
    print(f"📊 Tamaño final estimado: {total_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    consolidate_datasets()
