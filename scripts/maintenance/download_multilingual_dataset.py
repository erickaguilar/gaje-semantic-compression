import os
import json
import gzip
import requests

URL = "https://huggingface.co/datasets/AI-Culture-Commons/ai-culture-multilingual-json-dolma/resolve/main/ai-culture.jsonl.gz"
LOCAL_DIR = "data/datasets"
COMPRESSED_PATH = os.path.join(LOCAL_DIR, "ai-culture.jsonl.gz")
OUTPUT_TXT = "data/datasets/ai_culture_multilingual.txt"


def download_and_process():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    if not os.path.exists(COMPRESSED_PATH):
        print(f"[*] Descargando dataset desde {URL}...")
        response = requests.get(URL, stream=True)
        response.raise_for_status()

        with open(COMPRESSED_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[OK] Descarga finalizada: {COMPRESSED_PATH}")
    else:
        print(f"[*] Dataset comprimido ya existe: {COMPRESSED_PATH}")

    print("[*] Procesando JSONL y extrayendo texto...")
    count = 0
    try:
        with gzip.open(COMPRESSED_PATH, "rt", encoding="utf-8") as f_in:
            with open(OUTPUT_TXT, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    try:
                        data = json.loads(line)
                        text = data.get("text", "").strip()
                        if text:
                            # Limpiar saltos de línea internos
                            clean_text = " ".join(text.split())
                            f_out.write(clean_text + "\n")
                            count += 1
                            if count % 5000 == 0:
                                print(f"    - {count} documentos procesados...")
                    except Exception:
                        continue
    except Exception as e:
        print(f"[!] Error procesando el archivo: {e}")
        return None

    print(f"[OK] Dataset procesado: {OUTPUT_TXT} ({count} documentos)")
    return OUTPUT_TXT


if __name__ == "__main__":
    download_and_process()
