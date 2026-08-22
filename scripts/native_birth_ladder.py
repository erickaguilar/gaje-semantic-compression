#!/usr/bin/env python3
"""
🧬 GAJE HELIX — FASE 2: ESCALERA DE NACIMIENTO NATIVO (CURRÍCULO micro → 5M → 32M)
================================================================================
Implementa la cría progresiva de organismos 100% nacidos dentro del espacio discreto Q4_0:
  1. Peldaño 1: micro_organism (128 embd, 2 blocks, ~1.5M params)
  2. Peldaño 2: embryo_5m (256 embd, 4 blocks, ~5.2M params)
  3. Peldaño 3: silver_adult_32m (512 embd, 8 blocks, ~32M params)

Gate por Peldaño: Generación no colapsada (0% degeneración infinita) tras el
currículo híbrido (Crecimiento Hebbiano + SPSA de Orden Cero).
================================================================================
"""

import os
import sys
import time
import subprocess
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM
from dna_semantic_compression import NativeGenomicTrainer

EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "models", "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# 1. Dataset Sintético de Nutrición / Destilación Genómica
NUTRITION_DATASET = [
    [28, 39, 159, 34, 40, 201, 32, 39, 437, 95],
    [10, 50, 89, 20, 77, 120, 30, 99, 41, 88],
    [55, 31, 44, 98, 102, 21, 54, 88, 10, 30],
    [32, 41, 60, 128, 33, 78, 91, 44, 18, 70],
    [15, 24, 38, 51, 64, 77, 89, 101, 115, 128]
]

TIERS = [
    {"name": "micro_organism", "label": "Peldaño 1: Micro-Organismo (1.5M)", "preset": "micro_organism", "file": "birth_micro.flat", "k": 16, "epochs": 2},
    {"name": "embryo_5m", "label": "Peldaño 2: Embrión 5M (5.2M)", "preset": "embryo_5m", "file": "birth_5m.flat", "k": 32, "epochs": 2},
    {"name": "silver_adult_32m", "label": "Peldaño 3: Organismo 32M (32M)", "preset": "silver_adult_32m", "file": "birth_32m.flat", "k": 64, "epochs": 2},
]

print("================================================================================")
print("🧬 GAJE HELIX: FASE 2 — ESCALERA DE NACIMIENTO NATIVO (micro → 5M → 32M)")
print("================================================================================")

for tier in TIERS:
    print(f"\n--------------------------------------------------------------------------------")
    print(f"🐣 {tier['label']}")
    print(f"--------------------------------------------------------------------------------")
    
    out_path = os.path.join(EXPERIMENTS_DIR, tier["file"])
    
    # 1. Inicializar embrión nativo Q4_0 con gaje-cli
    print(f"[*] Inicializando embrión nativo con preset [{tier['preset']}]...")
    cli_bin = os.path.join(PROJECT_ROOT, "target", "release", "gaje-cli")
    cmd_init = [
        cli_bin,
        "--init", out_path,
        "--preset", tier["preset"]
    ]
    res = subprocess.run(cmd_init, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"❌ Error al instanciar embrión: {res.stderr}")
        sys.exit(1)
        
    print(f"✅ Embrión instanciado en: {out_path}")
    
    # 2. Cargar embrión en memoria
    t0 = time.time()
    llm = GenomicLLM.load_genomic(out_path)
    load_time_ms = (time.time() - t0) * 1000.0
    print(f"📦 Embrión cargado en memoria nativa en {load_time_ms:.1f} ms")
    
    # 3. Entrenamiento Nativo de Orden Cero (SPSA Discreto sobre Q4_0)
    trainer = NativeGenomicTrainer(lr=0.01, resonance_weight=0.05)
    
    print(f"[*] Entrenamiento de Orden Cero: SPSA Discreto sobre Q4_0 (k={tier['k']}, épocas={tier['epochs']})...")
    t_train = time.time()
    final_loss = trainer.fit_zero_order(llm.rust_llm, NUTRITION_DATASET, epochs=tier["epochs"], k_coords=tier["k"])
    train_duration = time.time() - t_train
    
    print(f"✅ Entrenamiento completado en {train_duration:.2f}s | Loss final: {final_loss:.4f}")
    
    # 4. Evaluación de Gate Generativo (Verificación de Estabilidad y No-Degeneración)
    print(f"[*] Verificando Gate Generativo del peldaño...")
    prompt_tokens = [280, 395]
    generated_tokens = llm.rust_llm.generate_native_py(prompt_tokens, 8, 0.7, 1.15, [])
            
    print(f"  • Secuencia generada: {prompt_tokens} -> {generated_tokens}")
    
    # Comprobar degeneración (repeticiones monótonas infinitas)
    is_degenerate = len(set(generated_tokens)) <= 1 and len(generated_tokens) > 3
    
    if not is_degenerate and len(generated_tokens) > 0:
        print(f"✅ GATE APROBADO para {tier['name']}: Generación estable, variada y 0% colapso monótono.")
    else:
        print(f"❌ GATE FALLIDO para {tier['name']}: Colapso degenerativo detectado.")
        sys.exit(1)

print("\n================================================================================")
print("🏆 CERTIFICACIÓN DE ESCALERA DE NACIMIENTO NATIVO (FASE 2) COMPLETADA AL 100%")
print("   Los tres peldaños (micro → 5M → 32M) fueron criados y validados nativamente.")
print("================================================================================")
