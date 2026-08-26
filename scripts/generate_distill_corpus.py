#!/usr/bin/env python3
"""Genera un corpus de destilación LIMPIO y DELIMITADO (prompt -> answer) a gran escala.

Reemplaza al viejo `distill_smollm2_1teacher.py` (5 prompts repetidos, 16 tokens, solo
lm_head). Objetivo: producir un `.jsonl` de pares {"prompt", "answer"} diverso y con
respuestas NO truncadas, para entrenar el CUERPO del estudiante (lm_head congelado) vía
`examples/export_trained.rs`.

Lección del diagnóstico C: el CE medio sobre un stream concatenado/ruidoso NO correlaciona
con la calidad de generación. Por eso este corpus es por-ejemplo (prompt delimitado ->
respuesta del maestro), NO un `.txt` concatenado. Cada línea = secuencia independiente.

Uso:
    python scripts/generate_distill_corpus.py \
        --teacher models/production/qwen2_5_3b_q4_0_q8_0_embd.gaje.flat \
        --prompts 150 --max_tokens 96 \
        --out data/distill/train_clean_150.jsonl
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM  # noqa: E402

# Banco de prompts DIVERSO (factual, definición, razonamiento, traducción, código,
# conocimiento general, completar, historia, naturaleza). Se usan por orden.
PROMPT_BANK = [
    # --- Geografía / capitales / países ---
    "¿Cuál es la capital de Francia?",
    "¿Cuál es la capital de España?",
    "¿Cuál es la capital de Japón?",
    "¿Cuál es la capital de México?",
    "¿Cuál es la capital de Argentina?",
    "¿Cuál es la capital de Alemania?",
    "¿Cuál es la capital de Italia?",
    "¿Cuál es la capital de Brasil?",
    "¿Cuál es la capital de Canadá?",
    "¿Cuál es la capital de Egipto?",
    "¿En qué continente está España?",
    "¿Cuál es el río más largo del mundo?",
    "¿Cuál es la montaña más alta del mundo?",
    "¿Cuál es el océano más grande?",
    "¿Cuántos continentes hay y cuáles son?",
    "¿Cuál es el país más grande del mundo por superficie?",
    "¿Dónde está el desierto del Sahara?",
    # --- Ciencia / definiciones ---
    "¿Qué es la fotosíntesis?",
    "¿Qué es el ADN?",
    "¿Qué es la gravedad?",
    "¿Qué es un átomo?",
    "¿Qué es una célula?",
    "¿Qué es la evolución?",
    "¿Qué es el cambio climático?",
    "¿Qué es la energía cinética?",
    "¿Qué es la electricidad?",
    "¿Qué es un ecosistema?",
    "Explica qué es la teoría de la relatividad.",
    "¿Qué es la tabla periódica?",
    "¿Qué es un virus?",
    "¿Qué es la ósmosis?",
    "¿Qué es un imán y cómo funciona?",
    # --- Matemáticas / razonamiento ---
    "Si x = 12 y el padre tiene 36 años, ¿cuánto tiene el hijo?",
    "¿Cuánto es 17 por 6?",
    "Si un tren viaja a 90 km/h durante 3 horas, ¿qué distancia recorre?",
    "Un triángulo tiene dos ángulos de 45 grados, ¿cuánto mide el tercero?",
    "¿Cuál es el área de un cuadrado de lado 5?",
    "Si tengo 20 manzanas y doy 7, ¿cuántas me quedan?",
    "¿Cuál es el 25% de 80?",
    "¿Cuántos minutos hay en una hora y media?",
    "Si una pizza se corta en 8 porciones y como 3, ¿qué fracción queda?",
    "Resuelve la ecuación 2x + 4 = 14.",
    # --- Traducción ---
    "Traduce al inglés: 'hola, ¿cómo estás?'",
    "Traduce al inglés: 'el gato está sobre la mesa'.",
    "Traduce al inglés: 'me gusta leer libros'.",
    "Traduce al español: 'the sky is blue'.",
    "Traduce al español: 'water is essential for life'.",
    "Traduce al inglés: '¿dónde está la estación?'",
    # --- Código / programación ---
    "Escribe una función en Python que sume dos números.",
    "Escribe un bucle en Python que imprima del 1 al 5.",
    "¿Qué es un bucle 'for' en programación?",
    "¿Qué es una variable en programación?",
    "¿Qué es una función en Python?",
    "Escribe una lista en Python con tres frutas.",
    # --- Conocimiento general ---
    "¿Quién escribió 'Don Quijote de la Mancha'?",
    "¿Cuántos huesos tiene el cuerpo humano adulto?",
    "¿Quién pintó la Mona Lisa?",
    "¿Cuánto tiempo tarda la Tierra en dar la vuelta al Sol?",
    "¿Cuál es el planeta más grande del Sistema Solar?",
    "¿Qué es la Revolución Industrial?",
    "¿Quién fue Albert Einstein?",
    "¿Qué es la Organización de las Naciones Unidas?",
    "¿En qué año llegó el hombre a la Luna?",
    "¿Qué es la democracia?",
    "¿Cuál es el metal más abundante en la Tierra?",
    "¿Qué es la fotografía?",
    "¿Qué es una novela?",
    "¿Qué es el arte?",
    "¿Qué es la historia?",
    # --- Completar frases ---
    "El planeta más grande del Sistema Solar es",
    "La capital de España es",
    "El agua hierve a",
    "La fotosíntesis ocurre en las",
    "Los seres vivos necesitan oxígeno para",
    "La Tierra gira alrededor del",
    # --- Naturaleza / cuerpo humano ---
    "¿Qué es el corazón y cuál es su función?",
    "¿Para qué sirven los pulmones?",
    "¿Qué es el sistema solar?",
    "¿Qué son las plantas?",
    "¿Qué es un animal herbívoro?",
    "¿Qué es el agua?",
    "¿Qué es la luz?",
    "¿Qué es el sonido?",
    # --- Instrucciones / explicaciones cortas ---
    "Explica en dos frases cómo preparar un té.",
    "Explica en dos frases qué es una bicicleta.",
    "Explica por qué es importante reciclar.",
    "Explica qué es un volcán.",
    "Explica qué es la lluvia.",
    "Explica qué es un telescopio.",
]

# Categorías de respuestas degeneradas a filtrar.
DEGENERATE_MARKERS = [
    "[error",
    "error:",
    "Traceback",
    "nan",
    "Por,",
    ",,,,,",
    "…",
    "???",
]


def run(genomic, prompt, max_tokens):
    try:
        return "".join(genomic.generate(prompt, max_new_tokens=max_tokens))
    except Exception as e:  # noqa: BLE001
        return f"[error: {e}]"


def is_degenerate(text, prompt):
    t = text.strip()
    if len(t) < 3:
        return True
    if any(m in text for m in DEGENERATE_MARKERS):
        return True
    # Respuesta que solo repite el prompt literalmente sin añadir nada.
    if t.rstrip("?. ") == prompt.strip().rstrip("?. "):
        return True
    # Repetición excesiva de una misma palabra corta.
    words = t.split()
    if not words:
        return True
    if len(words) >= 3 and len(set(words)) <= 1:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--teacher", default="models/production/qwen2_5_3b_q4_0_q8_0_embd.gaje.flat"
    )
    ap.add_argument("--prompts", type=int, default=150)
    ap.add_argument("--max_tokens", type=int, default=96)
    ap.add_argument("--out", default="data/distill/train_clean_150.jsonl")
    args = ap.parse_args()

    teacher_path = os.path.join(PROJECT_ROOT, args.teacher)
    out_path = os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"[1/2] Cargando maestro: {teacher_path}")
    teacher = GenomicLLM.load_genomic(teacher_path)

    # Ciclar el banco hasta alcanzar el número de prompts pedido.
    prompts = (PROMPT_BANK * (args.prompts // len(PROMPT_BANK) + 1))[: args.prompts]

    records = []
    skipped = 0
    for i, p in enumerate(prompts):
        ans = run(teacher, p, args.max_tokens)
        if is_degenerate(ans, p):
            skipped += 1
            print(
                f"  [{i + 1}/{len(prompts)}] (filtrado) {p[:50]} -> {ans.strip()[:40]!r}"
            )
            continue
        records.append({"prompt": p, "answer": ans})
        print(f"  [{i + 1}/{len(prompts)}] {p[:50]} -> {ans.strip()[:60]!r}")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"\n[2/2] {len(records)} pares guardados en {out_path} (filtrados: {skipped})"
    )
    del teacher


if __name__ == "__main__":
    main()
