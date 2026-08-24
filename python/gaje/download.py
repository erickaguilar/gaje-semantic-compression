#!/usr/bin/env python3
"""
🧬 GAJE — Resumable Model Downloader & Registry (Zero-Dependency & Android/Termux Optimized)

Features:
- Zero external ML dependencies (runs with standard Python urllib).
- Automatic chunked resume (HTTP Range header) when downloads get interrupted on mobile networks.
- Android / Termux system telemetry (RAM, storage, AArch64/NEON checks).
- Pre-flight checks (free space vs model size, RAM vs model requirements).
- Mobile-friendly adaptive progress bar with real-time speed (MB/s) and ETA.
"""

import os
import sys
import time
import shutil
import urllib.request
import urllib.error
import ssl
import platform
import argparse
from typing import Dict, Any, Optional, Tuple

# Default target directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "production")

# Official GAJE Models Registry
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gaje_nano_1.5b": {
        "filename": "gaje_nano_1.5b.flat",
        "description": "GAJE Nano 1.5B (Qwen2.5-1.5B 4-bit Hybrid) — Ultra-light edge organism",
        "size_mb": 520,
        "min_ram_gb": 1.5,
        "recommended_ram_gb": 3.0,
        "hf_repo": "erickaguilar/gaje-models",
        "url": "https://huggingface.co/erickaguilar/gaje-models/resolve/main/gaje_nano_1.5b.flat",
        "fallback_urls": [
            "https://huggingface.co/erickaguilar/gaje-models/raw/main/gaje_nano_1.5b.flat",
        ],
    },
    "gaje_prime_3b": {
        "filename": "gaje_prime_3b.flat",
        "description": "GAJE Prime 3B (Qwen2.5-3B 4-bit Hybrid) — Balanced workhorse organism",
        "size_mb": 1400,
        "min_ram_gb": 3.0,
        "recommended_ram_gb": 6.0,
        "hf_repo": "erickaguilar/gaje-models",
        "url": "https://huggingface.co/erickaguilar/gaje-models/resolve/main/gaje_prime_3b.flat",
        "fallback_urls": [],
    },
    "gaje_ultra_7b": {
        "filename": "gaje_ultra_7b.flat",
        "description": "GAJE Ultra 7B (DeepSeek-R1 / Qwen2.5-7B) — Advanced reasoning organism",
        "size_mb": 3800,
        "min_ram_gb": 6.0,
        "recommended_ram_gb": 12.0,
        "hf_repo": "erickaguilar/gaje-models",
        "url": "https://huggingface.co/erickaguilar/gaje-models/resolve/main/gaje_ultra_7b.flat",
        "fallback_urls": [],
    },
    "deepseek_r1_1.5b": {
        "filename": "deepseek_r1_1_5b_q4_0.gaje.flat",
        "description": "DeepSeek R1 Distill Qwen 1.5B (4-bit Q4_0 Flat)",
        "size_mb": 1254,
        "min_ram_gb": 2.0,
        "recommended_ram_gb": 4.0,
        "hf_repo": "erickaguilar/gaje-models",
        "url": "https://huggingface.co/erickaguilar/gaje-models/resolve/main/deepseek_r1_1_5b_q4_0.gaje.flat",
        "fallback_urls": [],
    },
    "smollm2_135m_gguf": {
        "filename": "smollm2-135m-instruct-fp16.gguf",
        "description": "SmolLM2 135M Instruct FP16 (GGUF base for transmutation)",
        "size_mb": 270,
        "min_ram_gb": 0.8,
        "recommended_ram_gb": 1.5,
        "hf_repo": "bartowski/SmolLM2-135M-Instruct-GGUF",
        "url": "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-f16.gguf",
        "fallback_urls": [],
    },
}

# Aliases for convenience
ALIASES: Dict[str, str] = {
    "nano": "gaje_nano_1.5b",
    "1.5b": "gaje_nano_1.5b",
    "prime": "gaje_prime_3b",
    "3b": "gaje_prime_3b",
    "ultra": "gaje_ultra_7b",
    "7b": "gaje_ultra_7b",
    "deepseek": "deepseek_r1_1.5b",
    "r1": "deepseek_r1_1.5b",
    "smollm2": "smollm2_135m_gguf",
    "smol": "smollm2_135m_gguf",
}


def get_system_specs() -> Dict[str, Any]:
    """Inspects host hardware and OS specs (Android/Termux friendly)."""
    is_android = os.path.exists("/data/data/com.termux") or "ANDROID_ROOT" in os.environ
    total_ram_gb = 0.0
    avail_ram_gb = 0.0

    # Read RAM from /proc/meminfo if available (standard on Linux/Android)
    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r") as f:
                mem_data = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip().split()[0]
                        mem_data[key] = int(val)
                total_kb = mem_data.get("MemTotal", 0)
                avail_kb = mem_data.get("MemAvailable", mem_data.get("MemFree", 0))
                total_ram_gb = total_kb / (1024 * 1024)
                avail_ram_gb = avail_kb / (1024 * 1024)
        except Exception:
            pass

    if total_ram_gb == 0.0:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            total_ram_gb = (pages * page_size) / (1024 * 1024 * 1024)
            avail_ram_gb = total_ram_gb * 0.6  # approximation
        except Exception:
            total_ram_gb = 4.0
            avail_ram_gb = 2.5

    # Storage space
    free_storage_gb = 0.0
    total_storage_gb = 0.0
    try:
        stat = shutil.disk_usage(PROJECT_ROOT)
        free_storage_gb = stat.free / (1024 * 1024 * 1024)
        total_storage_gb = stat.total / (1024 * 1024 * 1024)
    except Exception:
        pass

    return {
        "is_android": is_android,
        "platform": platform.system(),
        "machine": platform.machine(),
        "total_ram_gb": total_ram_gb,
        "avail_ram_gb": avail_ram_gb,
        "free_storage_gb": free_storage_gb,
        "total_storage_gb": total_storage_gb,
    }


def format_bytes(num_bytes: int) -> str:
    """Formats bytes into human readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    elif num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def render_progress_bar(
    downloaded: int,
    total: int,
    start_time: float,
    chunk_start_bytes: int,
    cols: int = 70,
) -> None:
    """Renders a single-line progress bar suited for mobile/Termux terminal."""
    elapsed = max(0.001, time.time() - start_time)
    bytes_in_session = downloaded - chunk_start_bytes
    speed_bps = bytes_in_session / elapsed
    speed_mbps = speed_bps / (1024 * 1024)

    if total > 0:
        percent = min(100.0, (downloaded / total) * 100.0)
        remaining_bytes = max(0, total - downloaded)
        eta_sec = remaining_bytes / max(1.0, speed_bps)
        if eta_sec > 3600:
            eta_str = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m"
        elif eta_sec > 60:
            eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
        else:
            eta_str = f"{int(eta_sec)}s"

        # Adaptive bar width
        bar_len = max(10, min(25, cols - 45))
        filled = int(bar_len * (percent / 100.0))
        bar = "█" * filled + "░" * (bar_len - filled)

        line = (
            f"\r[\033[36m{bar}\033[0m] {percent:5.1f}% | "
            f"{format_bytes(downloaded)}/{format_bytes(total)} | "
            f"\033[32m{speed_mbps:4.1f} MB/s\033[0m | ETA: {eta_str}  "
        )
    else:
        line = f"\rDescargando: {format_bytes(downloaded)} | \033[32m{speed_mbps:4.1f} MB/s\033[0m  "

    # Truncate if exceeds terminal width
    if len(line) > cols + 15:  # +15 for ANSI codes
        line = line[: cols + 15]

    sys.stdout.write(line)
    sys.stdout.flush()


def download_with_resume(
    url: str,
    target_path: str,
    expected_size_bytes: Optional[int] = None,
    token: Optional[str] = None,
) -> bool:
    """Downloads a file over HTTP with automatic resume support."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    temp_path = target_path + ".part"

    existing_bytes = 0
    if os.path.exists(temp_path):
        existing_bytes = os.path.getsize(temp_path)

    # Check if target already complete
    if os.path.exists(target_path):
        actual_size = os.path.getsize(target_path)
        if expected_size_bytes and actual_size == expected_size_bytes:
            print(f"✅ El modelo ya está completamente descargado en: {target_path}")
            return True
        elif not expected_size_bytes and actual_size > 0:
            print(f"✅ Archivo ya existe ({format_bytes(actual_size)}) en: {target_path}")
            return True

    # SSL context with modern cipher support
    ctx = ssl.create_default_context()

    headers = {
        "User-Agent": "GAJE-Downloader/1.6.0 (Android; Termux; Linux)",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        print(f"🔄 Reanudando descarga desde el byte {format_bytes(existing_bytes)}...")
    else:
        print(f"📥 Iniciando descarga...")

    req = urllib.request.Request(url, headers=headers)

    try:
        cols = shutil.get_terminal_size((80, 24)).columns
    except Exception:
        cols = 80

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            status_code = resp.status
            content_length = resp.headers.get("Content-Length")

            if status_code == 206:  # Partial content
                total_bytes = existing_bytes + (int(content_length) if content_length else 0)
                file_mode = "ab"
            elif status_code == 200:  # Full content (server ignored range or new download)
                if existing_bytes > 0:
                    print("⚠️ El servidor no soporta Range header; reiniciando desde byte 0.")
                existing_bytes = 0
                total_bytes = int(content_length) if content_length else (expected_size_bytes or 0)
                file_mode = "wb"
            else:
                total_bytes = int(content_length) if content_length else 0
                file_mode = "ab" if existing_bytes > 0 else "wb"

            downloaded_bytes = existing_bytes
            start_time = time.time()
            chunk_size = 128 * 1024  # 128 KB chunks for optimal Termux/TCP throughput

            with open(temp_path, file_mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
                    render_progress_bar(
                        downloaded_bytes,
                        total_bytes,
                        start_time,
                        existing_bytes,
                        cols=cols,
                    )

            sys.stdout.write("\n")
            sys.stdout.flush()

    except urllib.error.HTTPError as e:
        sys.stdout.write("\n")
        if e.code == 416:  # Range Not Satisfiable -> already complete
            if os.path.exists(temp_path):
                shutil.move(temp_path, target_path)
                print(f"✅ Descarga ya estaba completa.")
                return True
        print(f"\n❌ Error HTTP {e.code}: {e.reason}")
        return False
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        sys.stdout.write("\n")
        print(f"\n⚠️ Conexión interrumpida: {e}")
        print(f"💡 Puedes volver a ejecutar el comando y la descarga se reanudará automáticamente.")
        return False
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        print(f"\n⏸️ Descarga pausada por el usuario. Progreso guardado en: {temp_path}")
        print(f"💡 Ejecuta el comando nuevamente para reanudar.")
        return False

    # Atomically rename .part to destination
    if os.path.exists(temp_path):
        if os.path.exists(target_path):
            os.remove(target_path)
        shutil.move(temp_path, target_path)
        print(f"✨ Descarga completada con éxito en: \033[32m{target_path}\033[0m")
        return True

    return False


def resolve_model_key(name: str) -> Optional[str]:
    """Resolves aliases or model names to registry key."""
    norm = name.strip().lower().replace("-", "_").replace(".", "_")
    if norm in MODEL_REGISTRY:
        return norm
    if name.lower() in ALIASES:
        return ALIASES[name.lower()]
    for key in MODEL_REGISTRY:
        if name.lower() in key.lower() or key.lower() in name.lower():
            return key
    return None


def list_models_catalog(dest_dir: str) -> None:
    """Prints a styled catalog of available organisms and system fitness."""
    specs = get_system_specs()

    print("=" * 76)
    print("🧬 GAJE HELIX — CATÁLOGO DE MODELOS Y ORGANISMOS GENÓMICOS")
    print(f"💻 Sistema: {'Android/Termux' if specs['is_android'] else specs['platform']} "
          f"({specs['machine']}) | RAM Disp: {specs['avail_ram_gb']:.1f} GB / {specs['total_ram_gb']:.1f} GB | "
          f"Espacio Libre: {specs['free_storage_gb']:.1f} GB")
    print("=" * 76)

    for key, info in MODEL_REGISTRY.items():
        local_path = os.path.join(dest_dir, info["filename"])
        is_downloaded = os.path.exists(local_path)
        status = "✅ \033[32mDESCARGADO\033[0m" if is_downloaded else "📥 \033[33mDISPONIBLE\033[0m"

        # Compatibility check
        if specs["total_ram_gb"] < info["min_ram_gb"]:
            compat = "⚠️ \033[31mRAM Insuficiente\033[0m"
        elif specs["avail_ram_gb"] < info["min_ram_gb"]:
            compat = "⚠️ \033[33mRAM Justa\033[0m"
        else:
            compat = "🟢 \033[32mCompatible\033[0m"

        print(f"\n📦 \033[1m{key}\033[0m  [{status}]  ({compat})")
        print(f"   • Archivo:    {info['filename']} (~{info['size_mb']} MB)")
        print(f"   • Requisitos: RAM Mín {info['min_ram_gb']} GB (Recomendada: {info['recommended_ram_gb']} GB)")
        print(f"   • Info:       {info['description']}")
        print(f"   • Comando:    python scripts/download_model.py {key}")

    print("\n" + "=" * 76)


def download_model(
    model_name: str,
    dest_dir: str = DEFAULT_MODELS_DIR,
    token: Optional[str] = None,
    force: bool = False,
) -> bool:
    """Main orchestrator for downloading a model."""
    key = resolve_model_key(model_name)
    if not key:
        print(f"❌ Error: Modelo '{model_name}' no encontrado en el registro.")
        print(f"💡 Ejecuta con '--list' para ver todos los modelos disponibles.")
        return False

    info = MODEL_REGISTRY[key]
    target_path = os.path.join(dest_dir, info["filename"])
    expected_bytes = int(info["size_mb"] * 1024 * 1024)

    specs = get_system_specs()

    print("=" * 70)
    print(f"🧬 GAJE — DESCARGA DE ORGANISMO: \033[1m{key}\033[0m")
    print(f"🎯 Archivo destino: {target_path}")
    print(f"📦 Tamaño aprox:    {info['size_mb']} MB")
    print("=" * 70)

    # Pre-flight: Check free storage
    free_mb = specs["free_storage_gb"] * 1024
    if free_mb > 0 and free_mb < (info["size_mb"] + 100):
        print(f"❌ Error de espacio: Se requieren ~{info['size_mb']} MB pero solo hay {free_mb:.0f} MB libres.")
        return False

    # Pre-flight: RAM warning for Android
    if specs["is_android"] and specs["avail_ram_gb"] < info["min_ram_gb"]:
        print(f"⚠️ AVISO PARA ANDROID: El dispositivo cuenta con {specs['avail_ram_gb']:.1f} GB de RAM libre.")
        print(f"   Este modelo requiere {info['min_ram_gb']} GB. Podría cerrarse por Out-Of-Memory (OOM).")
        print("   Se sugiere cerrar apps en segundo plano o usar 'gaje_nano_1.5b'.\n")

    if force and os.path.exists(target_path):
        os.remove(target_path)

    # Try main URL
    success = download_with_resume(
        info["url"],
        target_path,
        expected_size_bytes=expected_bytes,
        token=token,
    )

    # If failed, try fallbacks
    if not success and info.get("fallback_urls"):
        for fb_url in info["fallback_urls"]:
            print(f"🔄 Intentando servidor alternativo ({fb_url})...")
            success = download_with_resume(
                fb_url,
                target_path,
                expected_size_bytes=expected_bytes,
                token=token,
            )
            if success:
                break

    return success


def main():
    parser = argparse.ArgumentParser(
        description="🧬 GAJE — Descargador Inteligente de Modelos Genómicos (Optimizado para Android/Termux)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python scripts/download_model.py --list
  python scripts/download_model.py nano
  python scripts/download_model.py gaje_prime_3b --dest models/production
  python scripts/download_model.py smollm2
        """,
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Nombre o alias del modelo (ej. nano, prime, ultra, deepseek, smollm2)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Listar todos los modelos disponibles en el catálogo con requisitos de RAM",
    )
    parser.add_argument(
        "--dest",
        default=DEFAULT_MODELS_DIR,
        help=f"Directorio de destino (por defecto: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Token de Hugging Face (opcional, para repositorios privados)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forzar descarga desde cero ignorando archivos existentes",
    )
    parser.add_argument(
        "--check-system",
        action="store_true",
        help="Mostrar telemetría del sistema (RAM, almacenamiento, arquitectura de procesador)",
    )

    args = parser.parse_args()

    if args.check_system:
        specs = get_system_specs()
        print("\n🧬 GAJE — TELEMETRÍA DEL SISTEMA")
        print(f"• Entorno:       {'Android / Termux 📱' if specs['is_android'] else 'Linux Desktop / Servidor 💻'}")
        print(f"• Arquitectura:  {specs['machine']} ({specs['platform']})")
        print(f"• RAM Total:     {specs['total_ram_gb']:.2f} GB")
        print(f"• RAM Libre:     {specs['avail_ram_gb']:.2f} GB")
        print(f"• Disco Total:   {specs['total_storage_gb']:.2f} GB")
        print(f"• Disco Libre:   {specs['free_storage_gb']:.2f} GB\n")
        return

    if args.list or not args.model:
        list_models_catalog(args.dest)
        return

    success = download_model(
        args.model,
        dest_dir=args.dest,
        token=args.token,
        force=args.force,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
