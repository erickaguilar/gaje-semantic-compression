#!/usr/bin/env python3
"""
🧬 GAJE-WASM: Test de Paridad Bit a Bit (Native CPU vs WebAssembly)
Valida determinismo idéntico en decodificación greedy (temperature=0.0)
entre el motor nativo en Rust (PyO3) y el motor compilado a WebAssembly (GajeWasmEngine).
"""

import os
import sys
import subprocess
import json
from gaje.nn.stabilized import GenomicLLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "production", "smollm2_135m.flat")

print(
    "================================================================================"
)
print("🧬 GAJE-WASM: TEST DE PARIDAD BIT A BIT (NATIVE CPU VS WEBASSEMBLY)")
print(
    "================================================================================"
)

if not os.path.exists(MODEL_PATH):
    print(f"❌ Modelo {MODEL_PATH} no encontrado.")
    sys.exit(1)

# 1. Inferencia Nativa (CPU Rust)
print("\n[1/3] Ejecutando inferencia determinista en Motor Nativo CPU...")
llm = GenomicLLM.load_genomic(MODEL_PATH)

prompt_tokens = [10, 42, 128, 256, 512]
max_new_tokens = 10
temperature = 0.0
repetition_penalty = 1.0
stop_ids = [2]

native_tokens = llm.rust_llm.generate_native_py(
    prompt_tokens, max_new_tokens, temperature, repetition_penalty, stop_ids
)
print(f"[+] Tokens generados en Nativo (CPU): {native_tokens}")

# 2. Inferencia en WebAssembly (GajeWasmEngine vía NodeJS)
print("\n[2/3] Ejecutando inferencia determinista en WebAssembly (GajeWasmEngine)...")

wasm_script = f"""
import fs from 'fs';
import {{ GajeWasmEngine }} from './pkg/wasm_node/_impl.js';

const fileBuffer = fs.readFileSync('{MODEL_PATH}');
const engine = GajeWasmEngine.load_from_bytes(new Uint8Array(fileBuffer));

const promptIds = new Uint32Array({json.dumps(prompt_tokens)});
const stopIds = new Uint32Array({json.dumps(stop_ids)});

const genIds = engine.generate(promptIds, {max_new_tokens}, {temperature}, {repetition_penalty}, stopIds);
console.log(JSON.stringify(Array.from(genIds)));
"""

wasm_res = subprocess.run(
    ["node", "--input-type=module", "-e", wasm_script],
    cwd=PROJECT_ROOT,
    capture_output=True,
    text=True,
    check=True,
)

wasm_tokens = json.loads(wasm_res.stdout.strip())
print(f"[+] Tokens generados en WebAssembly:  {wasm_tokens}")

# 3. Comparación y Certamen de Paridad
print("\n[3/3] Evaluando concordancia token a token...")
print(f"  • Nativo CPU: {native_tokens}")
print(f"  • WASM:       {wasm_tokens}")

assert len(native_tokens) == len(
    wasm_tokens
), f"Longitud dispar: {len(native_tokens)} vs {len(wasm_tokens)}"

discrepancies = []
for idx, (n_tok, w_tok) in enumerate(zip(native_tokens, wasm_tokens)):
    if n_tok != w_tok:
        discrepancies.append((idx, n_tok, w_tok))

if discrepancies:
    print(f"❌ Discrepancias encontradas: {discrepancies}")
    sys.exit(1)
else:
    print(
        "✅ PARIDAD BIT A BIT 100% PERFECTA: Nativo y WebAssembly producen exactamente los mismos tokens."
    )

print(
    "\n================================================================================"
)
print("🎯 CERTIFICACIÓN DE DETERMINISMO GAJE-WASM: APROBADA")
print(
    "================================================================================"
)
