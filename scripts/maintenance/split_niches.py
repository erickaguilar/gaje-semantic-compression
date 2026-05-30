import os
import re

DATASET_PATH = "data/datasets/full_silver_adult_dataset.txt"
OUTPUT_DIR = "data/training"

NICHE_A_PATH = os.path.join(OUTPUT_DIR, "niche_A.txt")
NICHE_B_PATH = os.path.join(OUTPUT_DIR, "niche_B.txt")
NICHE_C_PATH = os.path.join(OUTPUT_DIR, "niche_C.txt")

# Conectores para Nicho B (Español)
CONNECTORS = [
    r"\by\b",
    r"\bo\b",
    r"\bpero\b",
    r"\bporque\b",
    r"\bauunque\b",
    r"\bentonces\b",
    r"\bademás\b",
    r"\bsin embargo\b",
    r"\bluego\b",
    r"\bpor lo tanto\b",
    r"\bpor eso\b",
    r"\bsi bien\b",
    r"\baunque\b",
]

# Sufijos verbales para Nicho C (Español)
VERB_SUFFIXES = [
    "amos",
    "iste",
    "ieron",
    "ará",
    "ería",
    "ando",
    "ido",
    "emos",
    "imos",
    "aste",
    "asteis",
    "arán",
    "erán",
    "irán",
    "aba",
    "abas",
    "ábamos",
    "abais",
    "aban",
]


def split_niches():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset {DATASET_PATH} no encontrado.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    niche_a = []
    niche_b = []
    niche_c = []

    print(f"Procesando {len(lines)} líneas...")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Heurística Nicho B (Conectores)
        has_connector = any(re.search(c, line, re.IGNORECASE) for c in CONNECTORS)

        # Heurística Nicho C (Morfología Verbal)
        words = line.split()
        has_verb_suffix = any(
            any(word.lower().endswith(s) for s in VERB_SUFFIXES) for word in words
        )

        # Heurística Nicho A (Sintaxis Base)
        # Oraciones cortas, declarativas, sin demasiados conectores
        is_short = 4 <= len(words) <= 15
        is_declarative = line.endswith(".") and not (
            line.endswith("?") or line.endswith("!")
        )

        if is_short and is_declarative and not has_connector:
            niche_a.append(line)

        if has_connector:
            niche_b.append(line)

        if has_verb_suffix:
            niche_c.append(line)

    # Escribir resultados
    with open(NICHE_A_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(niche_a))

    with open(NICHE_B_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(niche_b))

    with open(NICHE_C_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(niche_c))

    print("Fase 1 completada:")
    print(f"- Nicho A (Sintaxis): {len(niche_a)} líneas")
    print(f"- Nicho B (Conectores): {len(niche_b)} líneas")
    print(f"- Nicho C (Verbos): {len(niche_c)} líneas")


if __name__ == "__main__":
    split_niches()
