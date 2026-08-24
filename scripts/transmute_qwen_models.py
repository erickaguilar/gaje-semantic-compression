#!/usr/bin/env python3
"""
🧬 GAJE — Master Transmuter: Qwen2.5 Models to .gaje.flat
Descarga y transmuta modelos oficiales de Hugging Face a formato binario zero-copy GAJE.

Modelos soportados:
  1. qwen_1.5b  -> gaje_nano_1.5b.flat    (Qwen 2.5 1.5B Instruct)
  2. qwen_7b    -> gaje_ultra_7b.flat   (Qwen 2.5 7B Instruct)
  3. coder_1.5b -> gaje_coder_1.5b.flat   (Qwen 2.5 Coder 1.5B Instruct)
  4. coder_7b   -> gaje_coder_7b.flat   (Qwen 2.5 Coder 7B Instruct)
"""

import os
import sys
import argparse
import subprocess
import urllib.request
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models", "production")
TEMP_DIR = os.path.join(PROJECT_ROOT, "models", "source")

MODELS_CATALOG = {
    "qwen_1.5b": {
        "title": "GAJE Nano (Qwen 2.5 1.5B Instruct)",
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "gguf_name": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "output_flat": "gaje_nano_1.5b.flat",
        "tokenizer": "Qwen/Qwen2.5-1.5B-Instruct"
    },
    "qwen_7b": {
        "title": "GAJE Ultra (Qwen 2.5 7B Instruct)",
        "url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
        "gguf_name": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "output_flat": "gaje_ultra_7b.flat",
        "tokenizer": "Qwen/Qwen2.5-7B-Instruct"
    },
    "coder_1.5b": {
        "title": "GAJE Coder Nano (Qwen 2.5 Coder 1.5B Instruct)",
        "url": "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "gguf_name": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "output_flat": "gaje_coder_1.5b.flat",
        "tokenizer": "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    },
    "coder_7b": {
        "title": "GAJE Coder Ultra (Qwen 2.5 Coder 7B Instruct)",
        "url": "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "gguf_name": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "output_flat": "gaje_coder_7b.flat",
        "tokenizer": "Qwen/Qwen2.5-Coder-7B-Instruct"
    }
}


def download_with_progress(url, dest_path):
    print(f"\n📥 Descargando desde: {url}")
    print(f"📁 Guardando temporalmente en: {dest_path}")
    
    # Intentar usar curl o wget si están disponibles para mayor velocidad
    try:
        ret = subprocess.run(["curl", "-L", "--progress-bar", "-o", dest_path, url], check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    try:
        ret = subprocess.run(["wget", "-c", "-O", dest_path, url], check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Fallback con urllib
    start_time = time.time()
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            downloaded_mb = (count * block_size) / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  [▓▓] {percent}% ({downloaded_mb:.1f} MB / {total_mb:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    print("\n✅ Descarga completada.")
    return True


def transmute_model(key, keep_source=False):
    if key not in MODELS_CATALOG:
        print(f"❌ Error: Modelo '{key}' no encontrado. Opciones: {list(MODELS_CATALOG.keys())}")
        return False

    spec = MODELS_CATALOG[key]
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(MODELS_ROOT, exist_ok=True)

    gguf_file = os.path.join(TEMP_DIR, spec["gguf_name"])
    flat_file = os.path.join(MODELS_ROOT, spec["output_flat"])

    print("=" * 70)
    print(f"🧬 Transmutando Organismo: {spec['title']}")
    print(f"🎯 Archivo destino: {flat_file}")
    print("=" * 70)

    # 1. Descarga si no existe
    if not os.path.exists(gguf_file):
        download_with_progress(spec["url"], gguf_file)
    else:
        print(f"⚡ Archivo fuente ya presente en caché: {gguf_file}")

    # 2. Transmutación a .flat
    exporter_script = os.path.join(PROJECT_ROOT, "scripts", "export_gaje_flat.py")
    cmd = [
        sys.executable,
        exporter_script,
        "--input", gguf_file,
        "--output", flat_file,
        "--tokenizer", spec["tokenizer"],
        "--quant-embed"
    ]

    print("\n⚙️ Ejecutando transmutación de tensores a formato binario plano zero-copy...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Error durante la exportación a .flat del modelo {key}")
        return False

    # 3. Limpieza de archivo GGUF temporal si no se requiere conservar
    if not keep_source and os.path.exists(gguf_file):
        print(f"\n🧹 Liberando espacio en disco (eliminando archivo temporal {gguf_file})...")
        os.remove(gguf_file)

    print(f"\n🎉 ¡Éxito! Modelo {spec['output_flat']} generado correctamente en:")
    print(f"   📂 {flat_file}")
    size_mb = os.path.getsize(flat_file) / (1024 * 1024)
    print(f"   📊 Tamaño final comprimido: {size_mb:.1f} MB\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Master Transmuter de Modelos GAJE")
    parser.add_argument(
        "model",
        choices=["qwen_1.5b", "qwen_7b", "coder_1.5b", "coder_7b", "all"],
        help="Modelo a transmutar ('all' para generar todos)"
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Conservar el archivo .gguf temporal descargado"
    )
    args = parser.parse_args()

    if args.model == "all":
        for k in MODELS_CATALOG.keys():
            transmute_model(k, keep_source=args.keep_source)
    else:
        transmute_model(args.model, keep_source=args.keep_source)


if __name__ == "__main__":
    main()
