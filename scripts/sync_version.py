#!/usr/bin/env python3
"""
🧬 GAJE Helix — Sincronizador Automático de Versión y Cache-Busting
Sincroniza el número de versión y hash de Git en Cargo.toml, Web UI, Service Worker y plantillas HTML.
"""

import os
import re
import sys
import subprocess
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEB_UI_DIR = os.path.join(ROOT_DIR, "examples", "ui", "web_ui")

def get_git_hash():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR)
        return out.decode("utf-8").strip()
    except Exception:
        return "latest"

def sync_version(new_version=None):
    # 1. Determinar versión
    config_js_path = os.path.join(WEB_UI_DIR, "static", "js", "config.js")
    if not new_version:
        if os.path.exists(config_js_path):
            with open(config_js_path, "r", encoding="utf-8") as f:
                m = re.search(r"var VERSION = ['\"]([^'\"]+)['\"];", f.read())
                if m:
                    new_version = m.group(1)
        if not new_version:
            new_version = "1.7.4"

    git_hash = get_git_hash()
    today_str = datetime.now().strftime("%Y-%m-%d")

    print(f"🧬 Sincronizando GAJE Helix a Versión: {new_version} (Hash: {git_hash}, Fecha: {today_str})")

    # 2. Sincronizar Cargo.toml
    cargo_path = os.path.join(ROOT_DIR, "Cargo.toml")
    if os.path.exists(cargo_path):
        with open(cargo_path, "r", encoding="utf-8") as f:
            content = f.read()
        content = re.sub(r'version = "[^"]+"', f'version = "{new_version}"', content, count=1)
        with open(cargo_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("  ✅ Cargo.toml sincronizado")

    # 3. Sincronizar static/js/config.js
    if os.path.exists(config_js_path):
        with open(config_js_path, "r", encoding="utf-8") as f:
            content = f.read()
        content = re.sub(r"var VERSION = ['\"][^'\"]+['\"];", f"var VERSION = '{new_version}';", content)
        content = re.sub(r"var BUILD_DATE = ['\"][^'\"]+['\"];", f"var BUILD_DATE = '{today_str}';", content)
        if "BUILD_HASH" in content:
            content = re.sub(r"var BUILD_HASH = ['\"][^'\"]+['\"];", f"var BUILD_HASH = '{git_hash}';", content)
        else:
            content = content.replace(f"var BUILD_DATE = '{today_str}';", f"var BUILD_DATE = '{today_str}';\n  var BUILD_HASH = '{git_hash}';")
        with open(config_js_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("  ✅ config.js sincronizado")

    # 4. Sincronizar sw.js
    sw_path = os.path.join(WEB_UI_DIR, "sw.js")
    if os.path.exists(sw_path):
        with open(sw_path, "r", encoding="utf-8") as f:
            content = f.read()
        content = re.sub(r":\s*['\"][0-9a-zA-Z\.\-_]+['\"];", f": '{new_version}';", content, count=1)
        with open(sw_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("  ✅ sw.js sincronizado")

    # 5. Sincronizar toolbar.js
    tb_path = os.path.join(WEB_UI_DIR, "static", "js", "chat", "toolbar.js")
    if os.path.exists(tb_path):
        with open(tb_path, "r", encoding="utf-8") as f:
            content = f.read()
        content = re.sub(r"chat_toolbar\.html\?v=[^'\"]+", f"chat_toolbar.html?v={new_version}", content)
        with open(tb_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("  ✅ toolbar.js sincronizado")

    # 6. Sincronizar archivos HTML (reemplazar ?v=... por la versión unificada)
    html_files = ["index.html", "docs.html", "architecture.html"]
    for hf in html_files:
        hp = os.path.join(WEB_UI_DIR, hf)
        if os.path.exists(hp):
            with open(hp, "r", encoding="utf-8") as f:
                content = f.read()
            content = re.sub(r'\?v=[0-9a-zA-Z\.\-_]+', f'?v={new_version}', content)
            with open(hp, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  ✅ {hf} sincronizado con ?v={new_version}")

    print("🎉 ¡Sincronización de versiones completada con éxito!")

if __name__ == "__main__":
    v = sys.argv[1] if len(sys.argv) > 1 else None
    sync_version(v)
