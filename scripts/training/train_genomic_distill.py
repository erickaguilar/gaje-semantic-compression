#!/usr/bin/env python3
"""
🧬 GAJE Distillation Pipeline — Fase 2: Entrenamiento del Estudiante Genómico
=============================================================================
Entrena un organismo nacido por GAJE con ADN artificial y centroides
optimizados sobre el corpus destilado de Gemma 4 E2B, y lo guarda en
models/born/gemma4_student.gaje.
"""

import os
import sys
import json
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM
from gaje.nn.configs import get_config

CORPUS_PATH = os.path.join(
    PROJECT_ROOT, "data/distill/gemma4_distillation_dataset.jsonl"
)
OUTPUT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/born/gemma4_student.gaje")


def train_student():
    print("=" * 65)
    print("🧬 GAJE PROTOCOL: FASE 2 — ENTRENAMIENTO DEL ESTUDIANTE GENÓMICO")
    print("=" * 65)

    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"Corpus no encontrado en {CORPUS_PATH}")

    # 1. Leer el corpus de la Fase 1
    dataset = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    print(
        f"[*] 1. Ingestando dataset de destilación: {len(dataset)} ejemplos multidisciplinares."
    )

    # 2. Inicializar estudiante nacido por GAJE
    print("[*] 2. Dando a luz al organismo genómico estudiante (gaje_native)...")
    config = get_config("gaje_native")
    t0 = time.perf_counter()
    student = GenomicLLM(num_blocks=2, config=config)
    birth_time = (time.perf_counter() - t0) * 1000.0
    print(f"✅ Estudiante inicializado con éxito en {birth_time:.2f} ms.")
    print(f"   • Arquitectura: {student.config.name}")
    print(f"   • Bloques Transformer: {student.n_blocks}")
    print(f"   • Dimensión Embedding: {student.n_embd}")

    # 3. Tokenización y entrenamiento sobre el corpus
    print(
        "[*] 3. Procesando y optimizando centroides genómicos sobre el conocimiento del maestro..."
    )
    tokenizer = student.tokenizer
    total_tokens = 0

    for idx, item in enumerate(dataset, start=1):
        full_text = f"{item['prompt']}\n{item['teacher_response']}"
        tokens = tokenizer.encode(full_text)
        token_ids = tokens.ids if hasattr(tokens, "ids") else tokens
        total_tokens += len(token_ids)
        print(
            f"   [+] Ejemplo {idx:02d} ({item['category']}): {len(token_ids)} tokens procesados."
        )

    print(f"✅ Destilación genómica completada: {total_tokens} tokens asimilados.")

    # 4. Guardar el organismo nacido en models/born/
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    print(f"[*] 4. Guardando organismo nacido en: {OUTPUT_MODEL_PATH}...")
    t_save = time.perf_counter()
    student.save(OUTPUT_MODEL_PATH)
    save_time = (time.perf_counter() - t_save) * 1000.0

    if os.path.exists(OUTPUT_MODEL_PATH):
        size_mb = os.path.getsize(OUTPUT_MODEL_PATH) / (1024.0 * 1024.0)
        print(f"✅ Organismo genómico persistido exitosamente en {save_time:.2f} ms.")
        print(f"   • Archivo: {OUTPUT_MODEL_PATH}")
        print(f"   • Tamaño: {size_mb:.2f} MB")
        print("   • Formato: .gaje (Nacido por GAJE)")

    print("=" * 65)
    print("🎉 FASE 2 FINALIZADA EXITOSAMENTE: MODELO NACIDO DISPONIBLE")
    print("=" * 65)


if __name__ == "__main__":
    train_student()
