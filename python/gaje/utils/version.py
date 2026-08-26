import os
import re


def get_project_version():
    """Reads the project version from Cargo.toml at the root of the repository."""
    try:
        # 1. Intentar desde el CWD (más común en scripts de ejecución)
        search_paths = [
            "Cargo.toml",
            "../Cargo.toml",
            "../../Cargo.toml",
            # 2. Intentar relativo al archivo version.py
            os.path.join(os.path.dirname(__file__), "..", "..", "Cargo.toml"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "Cargo.toml"),
        ]

        for path in search_paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Match version = "x.y.z" under [package]
                    # Usamos una expresión más flexible para evitar problemas con comentarios o espacios
                    match = re.search(
                        r'\[package\].*?version\s*=\s*"([^"]+)"', content, re.DOTALL
                    )
                    if match:
                        return match.group(1)
    except Exception:
        pass
    return "1.7.0-alpha"  # Fallback a la versión actual si falla todo
