#!/usr/bin/env python3
"""
🧬 GAJE — Script de Subida de Modelos .flat a Hugging Face Hub

Requisitos:
  pip install huggingface_hub

Uso:
  python3 scripts/maintenance/upload_models_to_huggingface.py --file models/production/gaje_coder_3b.flat
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "production")

DEFAULT_MODELS = ["gaje_coder_3b.flat", "gaje_pico_135m.flat"]


def main():
    parser = argparse.ArgumentParser(
        description="Subir modelos GAJE Flat a Hugging Face Hub"
    )
    parser.add_argument(
        "--repo",
        default="eaguilar/gaje-models",
        help="Nombre del repositorio (por defecto: eaguilar/gaje-models)",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Ruta o nombre de un modelo .flat específico a subir",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="Token de acceso de Hugging Face (con permisos de Write)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌ Error: Necesitas instalar 'huggingface_hub'. Ejecuta:")
        print("   pip install huggingface_hub")
        sys.exit(1)

    if not args.token:
        print("❌ Error: No se encontró ningún token de Hugging Face.")
        print("   Define la variable HF_TOKEN o pasa --token <hf_...>")
        sys.exit(1)

    api = HfApi(token=args.token)

    print("=" * 70)
    print("🧬 GAJE HELIX — SUBIDA DE MODELOS A HUGGING FACE HUB")
    print(f"🎯 Repositorio destino: {args.repo}")
    print("=" * 70)

    try:
        print(f"[*] Verificando o creando repositorio '{args.repo}' en Hugging Face...")
        create_repo(
            repo_id=args.repo, repo_type="model", token=args.token, exist_ok=True
        )
        print("✅ Repositorio listo.")
    except Exception as e:
        print(f"[!] Nota sobre repositorio: {e}")

    targets = []
    if args.file:
        if os.path.isabs(args.file) or os.path.exists(args.file):
            targets.append(args.file)
        else:
            targets.append(os.path.join(MODELS_DIR, args.file))
    else:
        for m in DEFAULT_MODELS:
            p = os.path.join(MODELS_DIR, m)
            if os.path.exists(p):
                targets.append(p)

    for target_path in targets:
        if not os.path.exists(target_path):
            print(f"⚠️ Aviso: Archivo {target_path} no encontrado, omitiendo.")
            continue

        model_name = os.path.basename(target_path)
        file_size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(
            f"\n📤 Subiendo {model_name} ({file_size_mb:.1f} MB)... Por favor espera."
        )
        try:
            api.upload_file(
                path_or_fileobj=target_path,
                path_in_repo=model_name,
                repo_id=args.repo,
                repo_type="model",
                token=args.token,
            )
            print(f"🎉 ¡{model_name} subido exitosamente!")
            print(
                f"   🔗 URL Directa: https://huggingface.co/{args.repo}/resolve/main/{model_name}"
            )
        except Exception as e:
            print(f"❌ Error subiendo {model_name}: {e}")

    print("\n" + "=" * 70)
    print("✅ Proceso de subida completado.")
    print("=" * 70)


if __name__ == "__main__":
    main()
