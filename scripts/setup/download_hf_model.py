"""
Script de descarga de modelo LLM desde HuggingFace para pruebas GAJE.

Descarga SmolLM2-135M-Instruct en formato GGUF (Q8_0) para pruebas
de inferencia con el motor de compresión semántica genómica.
"""

import os
from huggingface_hub import hf_hub_download

# Configuración del modelo
REPO_ID = "bartowski/SmolLM2-135M-Instruct-GGUF"
FILENAME = "SmolLM2-135M-Instruct-Q8_0.gguf"
LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    target_path = os.path.join(LOCAL_DIR, FILENAME)

    if os.path.exists(target_path):
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"[OK] Modelo ya existe: {target_path} ({size_mb:.1f} MB)")
        return target_path

    print(f"[*] Descargando {FILENAME} desde {REPO_ID}...")
    print(f"[*] Destino: {LOCAL_DIR}")

    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=LOCAL_DIR,
    )

    size_mb = os.path.getsize(downloaded_path) / (1024 * 1024)
    print(f"[OK] Descarga completada: {downloaded_path} ({size_mb:.1f} MB)")
    return downloaded_path


if __name__ == "__main__":
    path = main()
    print("\nUso:")
    print(f'  python tests/run_inference_test.py --model "{path}"')
