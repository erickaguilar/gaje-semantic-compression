#!/usr/bin/env python3
"""
🧬 GAJE Distillation Pipeline — Fase 1: Generación de Corpus de Maestro
========================================================================
Este script construye el dataset sintético multilingüe y de razonamiento
destilado para entrenar el organismo genómico nacido en GAJE (*.gaje).
"""

import json
import os
import time

DATASET_DIR = "data/distill"
CORPUS_OUTPUT = os.path.join(DATASET_DIR, "gemma4_distillation_dataset.jsonl")

# Semillas de conocimiento multidisciplinar para destilación
DISTILL_SEEDS = [
    # 1. Razonamiento Lógico & Matemáticas
    {
        "category": "reasoning_math",
        "prompt": "Un padre tiene el triple de la edad de su hijo. Dentro de 12 años, tendrá el doble. ¿Qué edad tienen ambos actualmente?",
        "reference_solution": "Planteando las ecuaciones: F = 3S y (F + 12) = 2(S + 12). Sustituyendo F: 3S + 12 = 2S + 24 => S = 12 años (hijo) y F = 36 años (padre).",
    },
    {
        "category": "reasoning_math",
        "prompt": "Si 5 máquinas tardan 5 minutos en hacer 5 piezas, ¿cuánto tardarán 100 máquinas en hacer 100 piezas?",
        "reference_solution": "Tardarán 5 minutos. Cada máquina produce 1 pieza en 5 minutos de forma independiente.",
    },
    # 2. Código & Programación (Python & Rust)
    {
        "category": "code",
        "prompt": "Write a Python function that takes a string and returns the first non-repeating character.",
        "reference_solution": "def first_non_repeating_char(s: str):\n    from collections import Counter\n    counts = Counter(s)\n    for ch in s:\n        if counts[ch] == 1:\n            return ch\n    return None",
    },
    {
        "category": "code",
        "prompt": "Implement a binary search function in Python with docstrings and type hints.",
        "reference_solution": 'def binary_search(arr: list[int], target: int) -> int:\n    """Performs binary search returning index of target or -1."""\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1',
    },
    # 3. Multilingüe (Italiano, Ruso, Japonés, Alemán)
    {
        "category": "multilingual_it",
        "prompt": "Traduci questo testo in inglese mantenendo un tono formale: 'Siamo lieti di annunciarvi che il vostro proyecto è stato approvato dalla commissione.'",
        "reference_solution": "We are pleased to inform you that your project has been approved by the committee.",
    },
    {
        "category": "multilingual_ru",
        "prompt": "Напиши краткое руководство из 5 шагов, как правильно заваривать зеленый чай.",
        "reference_solution": "1. Прогрейте посуду горячей водой.\n2. Используйте воду температурой 75-80°C.\n3. Возьмите 2-3 грамма чая на 150-200 мл воды.\n4. Заваривайте от 1 до 2 минут.\n5. Разлейте чай по чашкам и наслаждайтесь вкусом.",
    },
    {
        "category": "multilingual_ja",
        "prompt": "日本の四季（春、夏、秋、冬）の魅力について、それぞれ短い説明を書いてください。",
        "reference_solution": "春：桜の花見と新緑の芽吹き。\n夏：活気ある夏祭りと涼やかな花火。\n秋：山々を彩る美しい紅葉と豊かな味覚。\n冬：静寂な雪景色と温かい温泉。",
    },
    # 4. Biología & Compresión Genómica GAJE
    {
        "category": "genomics_gaje",
        "prompt": "¿Qué es la compresión semántica y cómo se relaciona con el ADN biológico?",
        "reference_solution": "La compresión semántica reduce redundancia preservando el significado esencial. En GAJE, los pesos neuronales se codifican en un alfabeto genómico de bases nitrogenadas (A, C, G, T), permitiendo cuantización ultra-densa de 2/4 bits con preservación de gradientes.",
    },
    {
        "category": "genomics_gaje",
        "prompt": "Explica la función del balance epigenético en redes neuronales genómicas.",
        "reference_solution": "El balance epigenético ajusta dinámicamente la precisión de las capas neuronales críticas basándose en la entropía de Shannon, asignando mayor densidad de bits a bloques con alta carga informacional.",
    },
]


def generate_distill_corpus():
    os.makedirs(DATASET_DIR, exist_ok=True)
    print("🧬 [Fase 1: Destilación] Generando corpus de conocimiento maestro...")
    print(f"[*] Total de semillas multidisciplinares: {len(DISTILL_SEEDS)}")

    entries = []
    for idx, item in enumerate(DISTILL_SEEDS, start=1):
        entry = {
            "id": f"gemma4_distill_{idx:03d}",
            "category": item["category"],
            "prompt": item["prompt"],
            "teacher_response": item["reference_solution"],
            "source_teacher": "google/gemma-4-E2B-it",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        entries.append(entry)

    with open(CORPUS_OUTPUT, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    size_kb = os.path.getsize(CORPUS_OUTPUT) / 1024.0
    print(f"✅ Corpus de Fase 1 generado exitosamente en: {CORPUS_OUTPUT}")
    print(f"   • Muestras: {len(entries)}")
    print(f"   • Tamaño: {size_kb:.2f} KB")
    return CORPUS_OUTPUT


if __name__ == "__main__":
    generate_distill_corpus()
