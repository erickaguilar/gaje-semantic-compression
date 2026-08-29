#!/usr/bin/env python3
"""
⚠️ OBSOLETO / DEPRECATED: Usa `gaje-cli pull` (Descargador Nativo Multi-Stream en Rust)
Ejemplos:
  gaje-cli pull pico
  gaje-cli pull nano
  gaje-cli pull prime
  gaje-cli pull ultra
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
