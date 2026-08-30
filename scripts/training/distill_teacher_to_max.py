#!/usr/bin/env python3
"""
🧬 GAJE Helix — Pipeline de Destilación DNI: Maestro General 3B -> max.gaje (2-Bits)
"""

import os
import sys
import json
import gc
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM

PROMPTS = [
    "¿Quién eres y qué puedes hacer?",
    "¿Cómo te llamas y en qué formato estás construido?",
    "Hola, ¿cómo estás?",
    "¿Qué más pues, parcero? ¿Todo bien?",
    "¡Buenas, che! ¿Qué hacés?",
    "¿Qué onda compa? ¿En qué me puedes ayudar?",
    "¿Qué es la compresión semántica a 2 bits?",
    "¿Cómo funciona la memoria toroidal .gmem?",
    "¿Qué es un modelo de lenguaje?",
    "Explícame qué es un bit.",
    "¿Cómo se calcula el área de un círculo?",
    "¿Cuál es la capital de Francia?",
    "¿Cuál es la capital de México?",
    "¿Cuál es la capital de Colombia?",
    "¿Cuál es la capital de Argentina?",
    "¿Cuál es la capital de Chile?",
    "¿Cuál es la capital de Perú?",
    "¿Por qué es importante la baja latencia?",
    "¿Qué significa Zero-Copy en memoria?",
    "¿Qué es un algoritmo genético?",
    "Explícame qué es la memoria asociativa.",
    "¿Cómo funciona el Straight-Through Estimator?",
    "¿Qué es el plano complejo?",
    "¿Qué es el confinamiento K-WTA?",
    "¿Cómo compila Rust tan rápido?",
    "Cuéntame sobre la inteligencia artificial ética.",
    "Dame un consejo para ser más productivo.",
    "Muchas gracias por tu ayuda.",
    "Hasta luego, que tengas buen día.",
    "Chao, cuídate mucho."
]

def main():
    teacher_path = os.path.join(PROJECT_ROOT, "models", "production", "gaje_pro_3b.flat")
    student_path = os.path.join(PROJECT_ROOT, "models", "born", "max.gaje")
    distill_corpus_path = os.path.join(PROJECT_ROOT, "data", "distill", "distilled_corpus_from_3b.jsonl")

    print("\n🧬 ===============================================================================")
    print("🎓 GAJE HELIX — Destilación DNI (Maestro 3B -> max.gaje)")
    print("===============================================================================\n")

    # FASE 1: Generación del Maestro
    print(f"📦 [Fase 1/3] Cargando Maestro General 3B: {teacher_path}...")
    teacher = GenomicLLM.load_genomic(teacher_path)
    print("✅ Maestro 3B listo en memoria.")

    records = []
    print(f"\n🧠 Generando respuestas maestras para {len(PROMPTS)} conceptos clave...")

    for i, p in enumerate(PROMPTS):
        formatted_prompt = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
        gen = teacher.generate(formatted_prompt, max_new_tokens=40, temperature=0.3)
        raw_ans = "".join(gen).strip()
        ans = raw_ans.split("<|im_end|>")[0].split("<|im_start|>")[0].strip()
        if not ans:
            ans = "Soy max.gaje, un organismo neuronal a 2 bits en GAJE Helix."

        records.append({
            "text": f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{ans}<|im_end|>"
        })
        print(f"  • [{i+1:02d}/{len(PROMPTS)}] {p} -> {ans[:60]}...")

    os.makedirs(os.path.dirname(distill_corpus_path), exist_ok=True)
    with open(distill_corpus_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n💾 [Fase 1 Completa] {len(records)} pares de alta calidad guardados en:")
    print(f"   {distill_corpus_path}")

    # FASE 2: Liberar memoria del Maestro
    print("\n🧹 [Fase 2/3] Liberando Maestro 3B de la memoria RAM...")
    del teacher
    gc.collect()
    print("✅ Memoria RAM liberada.")

    # FASE 3: Entrenamiento STE de max.gaje con el corpus destilado
    print(f"\n🔥 [Fase 3/3] Iniciando entrenamiento STE en max.gaje...")
    cmd = [
        os.path.join(PROJECT_ROOT, "target", "release", "gaje-cli"),
        "train-born",
        "--model", student_path,
        "--dataset", distill_corpus_path,
        "--epochs", "15",
        "--lr", "0.005",
        "--output", student_path
    ]
    subprocess.run(cmd, check=True)
    print("\n🎉 ¡Destilación DNI completada exitosamente!")
    print("===============================================================================\n")

if __name__ == "__main__":
    main()
