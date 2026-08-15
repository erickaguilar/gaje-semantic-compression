#!/usr/bin/env python3
"""Destilación 1-a-1 (texto) con estudiante SmolLM2 — MVP para pruebas rápidas.

Vía A (text-level distillation, offline):
  1. El maestro (.gaje.flat) genera respuestas para un corpus pequeño de prompts.
  2. El estudiante SmolLM2 (.gaje.flat) hace SFT (CE) SOLO sobre su lm_head (FP32),
     usando los textos generados por el maestro como objetivo.
  3. Evaluación in-place: deltas en prompts de control antes/después.

Por qué esta vía:
  - Sin mapeo de vocabulario maestro->estudiante (los textos se re-tokenizan en el
    tokenizer del estudiante).
  - Sin problema de memoria: se carga 1 modelo a la vez (generar -> guardar -> liberar).
  - Entrena solo lm_head (FP32), evitando el cuerpo Q4_0 (frágil en training).

Uso:
    python scripts/distill_smollm2_1teacher.py \
        --teacher models/production/qwen2_5_0_5b_q4_0_q8_0_embd.gaje.flat \
        --student models/production/smollm2_4bit.gaje.flat \
        --prompts 10 --epochs 2 --lr 1e-5
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.core import _impl as dna_semantic_compression  # noqa: E402
from gaje.nn.stabilized import GenomicLLM  # noqa: E402

# Prompts de control: factual, razonamiento simple, multilingüe.
CONTROL_PROMPTS = [
    "¿Cuál es la capital de Francia?",
    "La capital de España es",
    "Si x = 12 y el padre tiene 36 años, ¿cuánto tiene el hijo?",
    "Traduce al inglés: 'hola, ¿cómo estás?'",
    "El planeta más grande del Sistema Solar es",
]


def run(model, prompt, max_tokens=16):
    try:
        return "".join(model.generate(prompt, max_new_tokens=max_tokens))
    except Exception as e:  # noqa: BLE001
        return f"[error: {e}]"


def eval_prompts(model, label):
    print(f"\n=== {label} ===")
    for p in CONTROL_PROMPTS:
        out = run(model, p)
        print(f"  IN : {p}\n  OUT: {out.strip()}\n")


def tokenize_ids(tokenizer, text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(tokens, "ids"):
        tokens = tokens.ids
    return list(tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="models/production/qwen2_5_0_5b_q4_0_q8_0_embd.gaje.flat")
    ap.add_argument("--student", default="models/production/smollm2_4bit.gaje.flat")
    ap.add_argument("--prompts", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_tokens", type=int, default=16)
    ap.add_argument("--out", default=None, help="Guardar estudiante entrenado (.gaje)")
    args = ap.parse_args()

    teacher_path = os.path.join(PROJECT_ROOT, args.teacher)
    student_path = os.path.join(PROJECT_ROOT, args.student)
    distill_dir = os.path.join(PROJECT_ROOT, "data", "distill")
    os.makedirs(distill_dir, exist_ok=True)
    jsonl_path = os.path.join(distill_dir, "train_smollm2_1t.jsonl")

    # ---------- FASE 1: Generación offline del maestro ----------
    print(f"[Fase 1] Cargando maestro: {teacher_path}")
    teacher = GenomicLLM.load_genomic(teacher_path)
    prompts = [
        f"Responde brevemente y con precisión: {p}"
        for p in CONTROL_PROMPTS * max(1, (args.prompts + len(CONTROL_PROMPTS) - 1) // len(CONTROL_PROMPTS))
    ][: args.prompts]

    records = []
    for i, p in enumerate(prompts):
        ans = run(teacher, p, args.max_tokens)
        records.append({"prompt": p, "answer": ans})
        print(f"  [{i + 1}/{len(prompts)}] {p[:50]}... -> {ans.strip()[:60]}")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {len(records)} pares guardados en {jsonl_path}")
    del teacher  # liberar memoria antes de cargar el estudiante

    # ---------- FASE 2: SFT del estudiante (solo lm_head FP32) ----------
    print(f"\n[Fase 2] Cargando estudiante: {student_path}")
    student = GenomicLLM.load_genomic(student_path)

    # Secuencias de entrenamiento = texto (prompt + respuesta) en el tokenizer del estudiante
    dataset = []
    for r in records:
        ids = tokenize_ids(student.tokenizer, r["prompt"] + r["answer"])
        if len(ids) >= 4:
            dataset.append(ids)
    print(f"  -> {len(dataset)} secuencias (tam medio: {sum(len(s) for s in dataset) / max(1, len(dataset)):.0f} tokens)")

    eval_prompts(student, "ANTES del SFT")

    trainer = dna_semantic_compression.NativeGenomicTrainer(args.lr, 0.0)
    for epoch in range(args.epochs):
        loss = trainer.fit_lm_head(student.rust_llm, dataset, args.lr)
        print(f"  Época {epoch + 1}/{args.epochs} | Loss: {loss:.4f} | PPL: {loss ** 0.5:.3f}")

    eval_prompts(student, "DESPUÉS del SFT")

    # ---------- FASE 3: (opcional) persistir ----------
    if args.out:
        out = os.path.join(PROJECT_ROOT, args.out)
        student.save(out)
        print(f"  [Fase 3] Estudiante guardado en {out} (formato .gaje db).")

    print("\n[✓] Destilación 1-a-1 completada.")


if __name__ == "__main__":
    main()