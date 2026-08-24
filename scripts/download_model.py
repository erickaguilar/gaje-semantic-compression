#!/usr/bin/env python3
"""
🧬 GAJE — Script CLI de Descarga Inteligente de Modelos (Optimizado para Android / Termux)

Uso:
  python scripts/download_model.py --list
  python scripts/download_model.py nano
  python scripts/download_model.py prime
  python scripts/download_model.py --check-system
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_DIR = os.path.join(PROJECT_ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from gaje.download import main

if __name__ == "__main__":
    main()
