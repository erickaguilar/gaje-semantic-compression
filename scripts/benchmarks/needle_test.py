import os
import sys
import time
from tokenizers import Tokenizer

# Asegurar uso de código local
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
)

from gaje.core._impl import NativeLoader
from gaje.nn.stabilized import GenomicLLM


def run_needle_test(model_path, tokenizer_path, context_lengths=[128, 256, 512, 1024]):
    print("🧪 Needle in a Haystack - Silver Fetus Edition (Phase 5.0) 🧪")
    print(f"[*] Modelo: {model_path}")
    print("-" * 60)

    # 1. Cargar Organismo
    loader = NativeLoader(model_path)
    config = loader.py_load_config()
    rust_llm = loader.py_load_llm()

    # Inyectar topología si existe (esencial para Stage 5)
    topo_path = "models/core/topology_es.json"
    if os.path.exists(topo_path):
        rust_llm.load_topology(topo_path)
        print("[*] Topología CAM inyectada.")

    tokenizer = Tokenizer.from_file(tokenizer_path)

    student = GenomicLLM(
        None, config=config.config, n_embd=config.n_embd, num_blocks=config.n_blocks
    )
    student.rust_llm = rust_llm
    student.tokenizer = tokenizer

    # 2. Configurar la "Aguja" y el "Pajar"
    needle = "La clave secreta para la soberanía genómica es el código 7421."
    question = "¿Cual es el código para la soberanía genómica?"
    expected_answer = "7421"

    # Relleno (Pajar)
    haystack_base = (
        """
    La compresión semántica es un campo de estudio que busca reducir el tamaño de los modelos sin perder su esencia.
    El protocolo GAJE utiliza cuantización de 2 bits para lograr densidades extremas.
    Los centroides algebraicos proporcionan una rejilla rígida para la inteligencia.
    El entrenamiento por resonancia permite que modelos pequeños aprendan de maestros densos.
    """
        * 20
    )

    results = []

    for length in context_lengths:
        print(f"[*] Probando longitud de contexto: {length} tokens...")

        # Construir contexto
        # Insertar aguja en el medio
        full_haystack = haystack_base * (length // 50 + 1)
        words = full_haystack.split()
        insert_pos = len(words) // 2
        words.insert(insert_pos, needle)

        context_text = " ".join(words[:length])
        prompt = f"Contexto: {context_text}\nPregunta: {question}\nRespuesta:"

        start_time = time.time()
        # Generar respuesta
        # Nota: Usamos una temperatura baja para precisión
        gen = student.generate(prompt, max_new_tokens=10, temperature=0.1)
        response_tokens = []
        for token in gen:
            response_tokens.append(token)
        response = "".join(response_tokens)
        duration = time.time() - start_time

        success = expected_answer in response
        print(f"    - Respuesta: '{response.strip()}'")
        print(
            f"    - Resultado: {'✅ ÉXITO' if success else '❌ FALLO'} | {duration:.2f}s"
        )

        results.append(
            {
                "length": length,
                "success": success,
                "response": response.strip(),
                "time": duration,
            }
        )

    # 3. Reporte Final
    print("-" * 60)
    score = sum(1 for r in results if r["success"]) / len(results) * 100
    print(f"🏆 Puntuación Needle in a Haystack: {score:.2f}%")
    print(f"[*] Estado final: {'PASADO' if score >= 80 else 'FALLIDO'}")

    with open("docs/reports/NEEDLE_VALIDATION_20260525.md", "w") as f:
        f.write("# 🧪 Reporte de Validación: Needle in a Haystack (Stage 5)\n\n")
        f.write("**Modelo:** Silver Fetus (12.3 MB)\n")
        f.write("**Fecha:** 25 de mayo de 2026\n\n")
        f.write("| Contexto (tokens) | Resultado | Respuesta | Latencia |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for r in results:
            res_str = "✅" if r["success"] else "❌"
            f.write(
                f"| {r['length']} | {res_str} | {r['response']} | {r['time']:.2f}s |\n"
            )
        f.write(f"\n**Puntuación Total: {score:.2f}%**\n")


if __name__ == "__main__":
    run_needle_test(
        "models/checkpoints/silverfetus-distilled-cam.gaje",
        "models/core/tokenizer.json",
    )
