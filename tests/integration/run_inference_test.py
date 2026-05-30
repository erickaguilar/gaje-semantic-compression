"""
Test de inferencia end-to-end con motor GAJE en x86_64.

Este script:
1. Carga un modelo GGUF (SmolLM2-135M-Instruct)
2. Comprime sus pesos a formato de ADN genomico via GAJE
3. Ejecuta inferencia usando los kernels nativos (AVX2/FMA en x86_64, NEON en ARM)
4. Reporta tokens/segundo y texto generado
"""
import os
import sys
import time
import argparse
import platform

# Asegurar que el path del proyecto este en PYTHONPATH
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "python",
    ),
)


def detect_simd_capabilities():
    """Detecta las capacidades SIMD del hardware actual."""
    arch = platform.machine().lower()
    info = {
        "architecture": arch,
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
    }
    if "amd64" in arch or "x86_64" in arch:
        info["simd_target"] = "AVX2 + FMA (x86_64)"
    elif "aarch64" in arch or "arm" in arch:
        info["simd_target"] = "NEON (ARM aarch64)"
    else:
        info["simd_target"] = "Escalar (fallback)"
    return info


def run_inference_test(
    model_path, prompt, max_tokens=30, temperature=0.7, num_blocks=None
):
    """Ejecuta el test completo de inferencia."""
    from gaje.nn.stabilized import GenomicLLM

    # --- Info del sistema ---
    hw = detect_simd_capabilities()
    print("=" * 60)
    print("  GAJE - Test de Inferencia Nativa")
    print("=" * 60)
    print(f"  Arquitectura:   {hw['architecture']}")
    print(f"  Procesador:     {hw['processor']}")
    print(f"  SIMD Target:    {hw['simd_target']}")
    print(f"  OS:             {hw['os']}")
    print(f"  Python:         {hw['python_version']}")
    print(f"  Modelo:         {os.path.basename(model_path)}")
    print(f"  Num bloques:    {num_blocks or 'todos'}")
    print("=" * 60)

    # --- Carga del modelo (genomizacion) ---
    print("\n[1/3] Cargando y comprimiendo modelo a ADN genomico...")
    t0 = time.perf_counter()
    llm = GenomicLLM(model_path, num_blocks=num_blocks)
    load_time = time.perf_counter() - t0
    print(f"  -> Carga completada en {load_time:.2f}s")
    print(
        f"  -> Dimensiones: n_embd={llm.n_embd}, n_head={llm.n_head}, n_blocks={llm.n_blocks}"
    )

    # --- Tokenizacion ---
    print(f'\n[2/3] Tokenizando prompt: "{prompt}"')
    tokens = llm.tokenizer.encode(prompt, add_special_tokens=False)
    print(f"  -> {len(tokens)} tokens: {tokens}")

    # --- Inferencia ---
    print(f"\n[3/3] Generando texto ({max_tokens} tokens max, temp={temperature})...")
    print("-" * 40)

    generated_text = ""
    token_count = 0
    t_start = time.perf_counter()

    for token_text in llm.generate(
        prompt, max_new_tokens=max_tokens, temperature=temperature
    ):
        generated_text += token_text
        token_count += 1
        # Imprimir cada token en tiempo real (sin newline)
        sys.stdout.write(token_text)
        sys.stdout.flush()

    t_end = time.perf_counter()
    gen_time = t_end - t_start

    print("\n" + "-" * 40)

    # --- Reporte ---
    tps = token_count / gen_time if gen_time > 0 else 0
    print("\n[RESULTADOS]")
    print(f'  Prompt:           "{prompt}"')
    print(f'  Texto generado:   "{generated_text}"')
    print(f"  Tokens generados: {token_count}")
    print(f"  Tiempo total:     {gen_time:.3f}s")
    print(f"  Velocidad:        {tps:.2f} tokens/s")
    print(f"  Motor SIMD:       {hw['simd_target']}")

    # Verificacion basica de coherencia
    is_coherent = len(generated_text.strip()) > 0 and token_count > 0
    print(
        f"\n  Estado:           {'COHERENTE' if is_coherent else 'FALLO - sin texto generado'}"
    )

    return {
        "coherent": is_coherent,
        "tokens_per_second": tps,
        "generated_text": generated_text,
        "token_count": token_count,
        "load_time": load_time,
        "gen_time": gen_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAJE - Test de inferencia nativa")
    parser.add_argument(
        "--model", type=str, default=None, help="Ruta al archivo .gguf del modelo"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The meaning of life is",
        help="Prompt para la generacion de texto",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=30, help="Numero maximo de tokens a generar"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperatura de muestreo"
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Numero de bloques transformer a cargar (None = todos)",
    )

    args = parser.parse_args()

    # Auto-detectar modelo si no se especifica
    if args.model is None:
        default_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "models",
            "SmolLM2-135M-Instruct-Q8_0.gguf",
        )
        if os.path.exists(default_path):
            args.model = default_path
        else:
            print("[ERROR] No se encontro modelo. Ejecuta primero:")
            print("  python scripts/download_hf_model.py")
            sys.exit(1)

    result = run_inference_test(
        model_path=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_blocks=args.num_blocks,
    )

    sys.exit(0 if result["coherent"] else 1)
