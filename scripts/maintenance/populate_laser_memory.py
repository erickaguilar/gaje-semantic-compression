#!/usr/bin/env python3
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

print("🧠 [Hipocampo .gmem] Iniciando poblado de memoria congénita para max_laser...")

# Hechos clave de identidad, ciencia y arquitectura GAJE
FACTS = [
    "Soy GAJE (Genomic Adaptive Joint Embedding), un organismo de lenguaje nativo y memoria genética ultrarrápida.",
    "Estoy diseñado para ejecutarse localmente con compresión semántica y cero dependencia de servidores en la nube.",
    "Mi arquitectura física es un Láser Semántico (Deep and Narrow Conformal Waveguide) con doce capas y dimensión trescientos ochenta y cuatro.",
    "El protocolo GAJE comprime modelos a cuatro bits en producción y dos bits experimentales inspirados en las cuatro bases nitrogenadas del ADN.",
    "Mi memoria hipocampal .gmem permite recordar hechos y contexto en menos de medio milisegundo mediante mapeo mmap zero-copy.",
    "Python es un lenguaje de programación de alto nivel interpretado, ampliamente utilizado en inteligencia artificial.",
    "La estrella central de nuestro sistema planetario es el Sol, una estrella de tipo enana amarilla.",
    "La velocidad de la luz en el vacío es de aproximadamente trescientos mil kilómetros por segundo.",
    "La capital de Francia es París, la de España es Madrid, la de México es Ciudad de México y la de Colombia es Bogotá.",
    "La aceleración de hardware en procesadores ARM utiliza el conjunto de instrucciones vectoriales NEON para multiplicar matrices rápidamente."
]

memory_dir = os.path.join(PROJECT_ROOT, "models", "born", "max_laser_memory")
os.makedirs(memory_dir, exist_ok=True)
doc_file = os.path.join(memory_dir, "documental_facts.txt")

with open(doc_file, "w", encoding="utf-8") as f:
    for fact in FACTS:
        f.write(fact + "\n")

print(f"✅ Grabados {len(FACTS)} hechos fácticos estructurados en {doc_file}")
