import os
import numpy as np
import subprocess
from tqdm import tqdm

# Asegurar uso de código local
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python")))


from gaje.nn.stabilized import GenomicLLM


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def calculate_ppl(model, text, tokenizer, max_length=128):
    """Calcula la perplejidad de un texto usando el modelo GAJE."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return None

    tokens = tokens[:max_length]
    logits_seq = model.forward(tokens, clear_cache=True)

    logits_seq = logits_seq[:-1]
    target_tokens = tokens[1:]

    log_probs = []
    for i, target_id in enumerate(target_tokens):
        logits = logits_seq[i]
        probs = softmax(logits)
        safe_target_id = target_id % len(probs)
        p = np.clip(probs[safe_target_id], 1e-10, 1.0)
        log_probs.append(np.log(p))

    if not log_probs:
        model.clear_cache()
        return None

    avg_log_prob = np.mean(log_probs)
    model.clear_cache()
    return np.exp(-avg_log_prob)


def run_needle_test(model, needle, question, expected_answer_part):
    """Prueba de recuperación 'Needle in a Haystack'."""
    prompt = f"Contexto: {needle}\nPregunta: {question}\nRespuesta:"
    tokens = model.tokenizer.encode(prompt, add_special_tokens=False)

    # Generar respuesta
    generated_tokens = []
    curr_tokens = tokens
    model.clear_cache()
    for _ in range(20):
        logits = model.forward(curr_tokens, clear_cache=False)[-1]
        next_token = int(np.argmax(logits))
        generated_tokens.append(next_token)
        curr_tokens = [next_token]
        if next_token == model.tokenizer.eos_token_id:
            break

    response = model.tokenizer.decode(generated_tokens)
    success = expected_answer_part.lower() in response.lower()
    return success, response


def main():
    print("🧬 GAJE Certification Suite: Nivel 3 - Ingesta No-Destructiva")

    # Configuración
    model_path = (
        "models/silver_adult.gaje"
        if os.path.exists("models/silver_adult.gaje")
        else "models/production/silver_adult_anchored.gaje"
    )
    control_data_path = "data/datasets/coherence_es.txt"
    needle_data = "El código de acceso secreto para el nivel 3 es 'SILVER_SOUL_2026'."
    needle_file = "temp_needle.txt"

    if not os.path.exists(model_path):
        print(f"❌ Error: No se encuentra el modelo en {model_path}")
        return

    # 1. Cargar modelo
    print("[~] Cargando modelo base...")
    model = GenomicLLM.load_genomic(model_path)
    tokenizer = model.tokenizer

    # 2. Medir PPL Pre-Ingesta
    print("[~] Midiendo PPL Pre-Ingesta (Control)...")
    with open(control_data_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if len(line.strip()) > 20][:3]

    ppls_pre = []
    for line in tqdm(lines, desc="PPL Pre"):
        val = calculate_ppl(model, line, tokenizer)
        if val:
            ppls_pre.append(val)

    avg_ppl_pre = np.mean(ppls_pre)
    print(f"  - PPL Pre: {avg_ppl_pre:.4f}")

    # 3. Realizar Ingesta (DNI)
    print("[~] Realizando Ingesta (DNI) de dato nuevo...")
    with open(needle_file, "w") as f:
        f.write(needle_data)

    # Ejecutar gaje-cli ingest
    ingest_cmd = [
        "./target/release/gaje-cli",
        "--model",
        model_path,
        "--dni-ingest",
        needle_file,
        "--intensity",
        "0.005",
        "--pop",
        "8",
        "--gens",
        "20",
        "--output",
        "temp_dni_model.gaje",
    ]

    try:
        subprocess.run(ingest_cmd, check=True, capture_output=True)
    except Exception as e:
        print(f"❌ Error al ejecutar gaje-cli: {e}")
        return

    # 4. Cargar modelo Post-Ingesta
    print("[~] Cargando modelo modificado...")
    model_post = GenomicLLM.load_genomic("temp_dni_model.gaje")

    # 5. Medir PPL Post-Ingesta
    print("[~] Midiendo PPL Post-Ingesta (Control)...")
    ppls_post = []
    for line in tqdm(lines, desc="PPL Post"):
        val = calculate_ppl(model_post, line, tokenizer)
        if val:
            ppls_post.append(val)

    avg_ppl_post = np.mean(ppls_post)
    delta_ppl = ((avg_ppl_post - avg_ppl_pre) / avg_ppl_pre) * 100
    print(f"  - PPL Post: {avg_ppl_post:.4f}")
    print(f"  - Delta PPL: {delta_ppl:.2f}%")

    # 6. Test de Recuperación (Needle)
    print("[~] Verificando recuperación del conocimiento inyectado...")
    success, resp = run_needle_test(
        model_post,
        "",  # No pasamos el contexto en el prompt para verificar memoria interna
        "¿Cuál es el código de acceso secreto para el nivel 3?",
        "SILVER_SOUL_2026",
    )

    print(f"  - Respuesta del modelo: '{resp.strip()}'")
    print(f"  - Recuperación: {'✅ EXITOSA' if success else '❌ FALLIDA'}")

    # 7. Veredicto Final
    print("\n" + "=" * 60)
    print("📊 RESULTADOS NIVEL 3")
    print(
        f"  - Delta PPL (<1%):    {delta_ppl:.2f}% {'✅' if delta_ppl < 1.0 else '❌'}"
    )
    print(f"  - Recuperación DNI:   {'✅ PASSED' if success else '❌ FAILED'}")

    cert_passed = delta_ppl < 1.0 and success
    print(
        f"  - ESTADO FINAL:       {'✅ CERTIFICABLE' if cert_passed else '❌ NO CERTIFICABLE'}"
    )
    print("=" * 60)

    # Limpieza
    if os.path.exists(needle_file):
        os.remove(needle_file)
    # if os.path.exists("temp_dni_model.gaje"): os.remove("temp_dni_model.gaje")


if __name__ == "__main__":
    main()
