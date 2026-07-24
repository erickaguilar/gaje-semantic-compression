import os
import sys
import numpy as np

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM


def run_sovereignty_test(model_path):
    print("=" * 70)
    print("🛡️  PROTOCOLO DE CERTIFICACIÓN DE SOBERANÍA NATIVA (V1.0)")
    print("=" * 70)

    if not os.path.exists(model_path):
        print(f"❌ Error: Modelo no encontrado en {model_path}")
        return

    print("[1/3] [*] Cargando Organismo en el espacio de fase...")
    try:
        llm = GenomicLLM.load_genomic(model_path)
        tokenizer = llm.tokenizer
    except Exception as e:
        print(f"❌ FALLO CRÍTICO DE CARGA: {e}")
        return

    # --- PRUEBA 1: ECO SEMÁNTICO ---
    print("\n[2/3] [*] Prueba de Eco Semántico (English Style)...")
    prompt = "To be, or not to be, that is the"
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if hasattr(tokens, "ids"):
        tokens = tokens.ids

    llm.rust_llm.clear_cache_py()
    current_id = tokens[0]

    print(f"💬 Prompt: '{prompt}'")
    print("🤖 Respuesta: ", end="", flush=True)

    response_tokens = []
    for _ in range(10):
        logits = llm.rust_llm.forward(current_id, False)
        next_id = int(np.argmax(logits))

        # Verificar integridad de logits
        if np.isnan(logits).any() or np.isinf(logits).any():
            print("\n❌ FALLO: Logits contienen NaNs o Infinitos (Explosión de Fase)")
            break

        token_text = tokenizer.decode([next_id])
        print(token_text, end="", flush=True)
        response_tokens.append(next_id)
        current_id = next_id
        if next_id == tokenizer.eos_token_id:
            break

    # --- PRUEBA 2: HOMEOSTASIS ---
    print("\n\n[3/3] [*] Prueba de Homeostasis (Repetición Estacionaria)...")
    # Inyectamos el mismo token 5 veces y vemos si la distribución de salida es estable
    distributions = []
    token_id = tokenizer.encode(" Hello", add_special_tokens=False)
    if hasattr(token_id, "ids"):
        token_id = token_id.ids
    token_id = token_id[0]

    is_stable = True
    for i in range(5):
        logits = llm.rust_llm.forward(token_id, False)
        top_k = np.argsort(logits)[-5:]
        distributions.append(top_k)
        if i > 0 and not np.array_equal(distributions[i], distributions[0]):
            is_stable = False
            print(
                f"    [!] Inestabilidad en iteración {i}: {top_k} vs {distributions[0]}"
            )

    if is_stable:
        print("✅ HOMEOSTASIS: Estable (La señal no se degrada por repetición)")
    else:
        print("⚠️  HOMEOSTASIS: Inestable (Deriva de fase detectada)")

    print("\n" + "=" * 70)
    print("📊 RESULTADO FINAL")
    print("-" * 70)
    coherence = len(response_tokens) > 0 and not any(
        t == response_tokens[0] for t in response_tokens[1:4]
    )
    if coherence:
        print("ESTADO: 🏆 SOBERANÍA NATIVA CERTIFICADA")
    else:
        print("ESTADO: ❌ SOBERANÍA DENEGADA (Delirio Semántico)")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="models/production/silver_adult_calibrated.gaje"
    )
    args = parser.parse_args()
    run_sovereignty_test(args.model)
