#!/usr/bin/env python3
"""GAJE Helix — Scientific Benchmark CLI Runner.

Usage:
    python3 scripts/gaje_benchmark.py [options]

Options:
    --models MODEL1 MODEL2 ...   Nombres o patrones de modelos a evaluar
    --dataset PATH               Ruta al banco de prompts (default: data/eval/benchmark_prompts.json)
    --format FORMAT              Formato de reporte: table, markdown, json (default: table)
    --output PATH                Ruta de salida del reporte (default: stdout / docs/reports/BENCHMARK_OFFICIAL.md)
    --limit N                    Limitar a los primeros N prompts de prueba
"""

import argparse
import json
import os
import sys
import time
from typing import List

# Rutas del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "examples", "ui", "web_ui"))

from gaje.nn.stabilized import GenomicLLM
from gaje.utils.version import get_project_version
from gaje.metrics.benchmark import evaluate_model_benchmark, BenchmarkReport
from model_manager import get_model, list_available_models, unload_model
from prompt_templates import format_prompt, get_stop_tokens

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DEFAULT_DATASET = os.path.join(PROJECT_ROOT, "data", "eval", "benchmark_prompts.json")


def format_markdown_table(reports: List[BenchmarkReport]) -> str:
    """Genera una tabla comparativa Markdown de nivel científico."""
    md = []
    md.append("# 📊 GAJE Helix — Reporte Oficial de Benchmarks Científicos\n")
    md.append(f"**Fecha:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    md.append(f"**Versión del Motor:** GAJE v{get_project_version()} (Rust SIMD AVX2 + PyO3 / Python {sys.version.split()[0]})  ")
    md.append(f"**Hardware de Referencia:** AMD Ryzen 7 5800H (16 hilos) - x86_64  \n")
    md.append("---\n")
    md.append("## 🏆 1. Resumen Comparativo de Modelos\n")
    md.append("| Modelo | Arquitectura | Tamaño Disco | Cold-Start | Peak RSS | Gen Speed | Diversidad ($d_1/d_2$) | Recall Semántico | Degeneración | Compresión |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in reports:
        d1_d2 = f"{r.avg_distinct_1:.2f} / {r.avg_distinct_2:.2f}"
        recall = f"{r.avg_keyword_recall * 100.0:.1f}%"
        degen = f"{r.repetition_rate_pct:.1f}%"
        comp = f"{r.compression_ratio:.1f}x ({r.memory_savings_pct:.1f}%)"
        md.append(
            f"| **`{r.model_name}`** | {r.architecture} | {r.file_size_mb:.1f} MB | {r.cold_start_ms:.1f} ms | {r.peak_rss_mb:.1f} MB | **{r.avg_tokens_per_sec:.1f} tok/s** | {d1_d2} | {recall} | {degen} | **{comp}** |"
        )

    md.append("\n---\n")
    md.append("## 🔬 2. Definición de Métricas Evaluadas\n")
    md.append("* **Cold-Start (ms):** Tiempo para mapear el archivo binario a memoria vía Zero-Copy Mmap.")
    md.append("* **Peak RSS (MB):** Memoria física residente máxima alcanzada en el sistema.")
    md.append("* **Gen Speed (tok/s):** Throughput sostenido de generación autoregresiva token a token.")
    md.append("* **Diversidad ($d_1 / d_2$):** Fracción de unigramas y bigramas únicos generados (métrica Distinct).")
    md.append("* **Recall Semántico (%):** Porcentaje de conceptos clave esperados respondidos con éxito.")
    md.append("* **Degeneración (%):** Tasa de respuestas que caen en bucles repetitivos (objetivo: 0.0%).")
    md.append("* **Compresión:** Ratio de ahorro de memoria vs FP32 equivalente.")
    md.append("\n---\n*Reporte generado automáticamente por `scripts/gaje_benchmark.py`.*")
    return "\n".join(md)


def print_ascii_table(reports: List[BenchmarkReport]):
    """Imprime un resumen formateado en la terminal."""
    print("\n" + "=" * 115)
    print(f"{'MODELO':<28} | {'ARQ':<11} | {'DISCO':<8} | {'COLD(ms)':<8} | {'RSS(MB)':<8} | {'SPEED':<10} | {'RECALL':<8} | {'COMPRESIÓN'}")
    print("=" * 115)
    for r in reports:
        speed = f"{r.avg_tokens_per_sec:.1f} tok/s"
        recall = f"{r.avg_keyword_recall * 100:.1f}%"
        comp = f"{r.compression_ratio:.1f}x ({r.memory_savings_pct:.0f}%)"
        print(f"{r.model_name:<28} | {r.architecture:<11} | {r.file_size_mb:>6.1f}MB | {r.cold_start_ms:>8.1f} | {r.peak_rss_mb:>7.1f} | {speed:>10} | {recall:>8} | {comp}")
    print("=" * 115 + "\n")


def main():
    parser = argparse.ArgumentParser(description="GAJE Scientific Benchmark CLI")
    parser.add_argument("--models", nargs="*", help="Lista de nombres de modelos a evaluar")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Ruta al archivo JSON de prompts")
    parser.add_argument("--format", choices=["table", "markdown", "json"], default="table", help="Formato de salida")
    parser.add_argument("--output", help="Ruta del archivo de salida")
    parser.add_argument("--limit", type=int, default=None, help="Limitar a N prompts")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"❌ Error: Dataset no encontrado en {args.dataset}")
        sys.exit(1)

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.limit and args.limit > 0:
        dataset["test_cases"] = dataset["test_cases"][:args.limit]

    # Modelos a evaluar
    available = list_available_models(MODELS_DIR)
    available_names = [m["name"] for m in available]

    target_models = args.models if args.models else available_names
    # Filtrar existentes
    target_models = [m for m in target_models if m in available_names]

    if not target_models:
        print("❌ No se encontraron modelos válidos para evaluar.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("🧬 EJECUTANDO SUITE DE BENCHMARKS CIENTÍFICOS GAJE HELIX")
    print(f"Modelos seleccionados ({len(target_models)}): {target_models}")
    print(f"Prompts a evaluar por modelo: {len(dataset.get('test_cases', []))}")
    print("=" * 80)

    reports: List[BenchmarkReport] = []

    for idx, model_name in enumerate(target_models, 1):
        print(f"\n[{idx}/{len(target_models)}] Evaluando modelo: {model_name}...")
        try:
            report = evaluate_model_benchmark(
                models_root=MODELS_DIR,
                model_name=model_name,
                prompts_dataset=dataset,
                get_model_fn=get_model,
                genomic_llm_cls=GenomicLLM,
                format_prompt_fn=format_prompt,
                get_stop_tokens_fn=get_stop_tokens,
            )
            reports.append(report)
            print(f"  ✓ Cold-Start: {report.cold_start_ms} ms | Speed: {report.avg_tokens_per_sec} tok/s | Recall: {report.avg_keyword_recall * 100:.1f}%")
        except Exception as e:
            print(f"  ❌ Error evaluando {model_name}: {e}")
        finally:
            unload_model()

    # Formatear salida
    if args.format == "table" or not args.output:
        print_ascii_table(reports)

    if args.format == "markdown" or (args.output and args.output.endswith(".md")):
        md_content = format_markdown_table(reports)
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(md_content)
            print(f"📄 Reporte Markdown guardado en: {args.output}")
        else:
            print(md_content)

    elif args.format == "json" or (args.output and args.output.endswith(".json")):
        json_data = [r.to_dict() for r in reports]
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"📄 Reporte JSON guardado en: {args.output}")
        else:
            print(json.dumps(json_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
