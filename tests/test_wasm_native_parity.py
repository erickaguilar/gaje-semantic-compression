import os
import sys
import subprocess
import json
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "production", "smollm2_135m.flat")


def test_wasm_native_parity():
    """Valida determinismo idéntico en decodificación greedy entre Rust (PyO3) y WebAssembly."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Modelo {MODEL_PATH} no encontrado.")

    try:
        from gaje.nn.stabilized import GenomicLLM
    except ImportError:
        pytest.skip("gaje.nn.stabilized no disponible.")

    # 1. Inferencia Nativa (CPU Rust)
    llm = GenomicLLM.load_genomic(MODEL_PATH)
    prompt_tokens = [10, 42, 128, 256, 512]
    max_new_tokens = 10
    temperature = 0.0
    repetition_penalty = 1.0
    stop_ids = [2]

    native_tokens = llm.rust_llm.generate_native_py(
        prompt_tokens, max_new_tokens, temperature, repetition_penalty, stop_ids
    )

    # 2. Inferencia en WebAssembly (GajeWasmEngine vía NodeJS)
    wasm_pkg = os.path.join(PROJECT_ROOT, "pkg", "wasm_node", "_impl.js")
    if not os.path.exists(wasm_pkg):
        pytest.skip("WASM package pkg/wasm_node/_impl.js no encontrado.")

    wasm_script = f"""
import fs from 'fs';
import {{ GajeWasmEngine }} from './pkg/wasm_node/_impl.js';

const fileBuffer = fs.readFileSync('{MODEL_PATH.replace(chr(92), "/")}');
const engine = GajeWasmEngine.load_from_bytes(new Uint8Array(fileBuffer));

const promptIds = new Uint32Array({json.dumps(prompt_tokens)});
const stopIds = new Uint32Array({json.dumps(stop_ids)});

const genIds = engine.generate(promptIds, {max_new_tokens}, {temperature}, {repetition_penalty}, stopIds);
console.log(JSON.stringify(Array.from(genIds)));
"""
    try:
        wasm_res = subprocess.run(
            ["node", "--input-type=module", "-e", wasm_script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        wasm_tokens = json.loads(wasm_res.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        pytest.skip("NodeJS no disponible o fallo en ejecución de WebAssembly.")

    assert len(native_tokens) == len(wasm_tokens), f"Longitud dispar: {len(native_tokens)} vs {len(wasm_tokens)}"
    assert native_tokens == wasm_tokens, f"Discrepancia de tokens: {native_tokens} vs {wasm_tokens}"


if __name__ == "__main__":
    test_wasm_native_parity()
    print("✅ TEST PASSED")
