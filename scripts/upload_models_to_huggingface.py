#!/usr/bin/env python3
"""
🧬 GAJE — Script de Subida de Modelos .flat a Hugging Face Hub (100% Gratis)

Requisitos:
  pip install huggingface_hub

Uso:
  python3 scripts/upload_models_to_huggingface.py --repo tu-usuario/gaje-models --token hf_xxx
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "production")

MODELS_TO_UPLOAD = [
    "gaje_nano_1.5b.flat",
    "gaje_prime_3b.flat"
]


def main():
    parser = argparse.ArgumentParser(description="Subir modelos GAJE Flat a Hugging Face Hub")
    parser.add_argument("--repo", required=True, help="Nombre del repositorio (ej. tu-usuario/gaje-models)")
    parser.add_argument("--token", default=None, help="Token de acceso de Hugging Face (con permisos de Write)")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌ Error: Necesitas instalar 'huggingface_hub'. Ejecuta:")
        print("   pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=args.token)

    print("=" * 70)
    print("🧬 GAJE HELIX — SUBIDA DE MODELOS A HUGGING FACE HUB")
    print(f"🎯 Repositorio destino: {args.repo}")
    print("=" * 70)

    try:
        print(f"[*] Verificando o creando repositorio '{args.repo}' en Hugging Face...")
        create_repo(repo_id=args.repo, repo_type="model", token=args.token, exist_ok=True)
        print("✅ Repositorio listo.")
    except Exception as e:
        print(f"[!] Nota sobre repositorio: {e}")

    for model_name in MODELS_TO_UPLOAD:
        local_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(local_path):
            print(f"⚠️ Aviso: Archivo {local_path} no encontrado, omitiendo.")
            continue

        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"\n📤 Subiendo {model_name} ({file_size_mb:.1f} MB)... Por favor espera.")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=model_name,
                repo_id=args.repo,
                repo_type="model",
                token=args.token
            )
            print(f"🎉 ¡{model_name} subido exitosamente!")
            print(f"   🔗 URL Directa: https://huggingface.co/{args.repo}/resolve/main/{model_name}")
        except Exception as e:
            print(f"❌ Error subiendo {model_name}: {e}")

    print("\n" + "=" * 70)
    print("✅ Proceso de subida completado.")
    print("=" * 70)


if __name__ == "__main__":
    main()
