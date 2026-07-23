import os
import sys

# Asegurar que usamos el código local de 'python/'
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
)

from gaje.nn.stabilized import GenomicLLM


def calculate_coherence_score(text):
    """Simple heuristic: ratio of alphanumeric characters and spaces."""
    if not text:
        return 0
    words = text.split()
    if len(words) < 5:
        return 0

    # Check for character variety (not just repeating one character)
    unique_chars = len(set(text))
    if unique_chars < 10:
        return 0

    return len(words)


def calibrate_homeostasis(model_path):
    print(f"🧬 Calibrando Homeostasis Políglota para: {model_path}")
    model = GenomicLLM.load_genomic(model_path)

    prompts = {
        "EN": "To be, or not to be, that is the question",
        "ES": "La compresión semántica genómica es una tecnología",
    }

    scales = [0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    results = []

    for scale in scales:
        print(f"\n[*] Probando h_scale = {scale:.2f}")
        for block in model.blocks:
            block.rust_block.h_scale = scale

        total_score = 0
        outputs = {}

        for lang, prompt in prompts.items():
            output = ""
            try:
                for token in model.generate(
                    prompt,
                    max_new_tokens=15,
                    use_spiking=True,
                    spiking_steps=64,
                    spiking_threshold=0.1,
                    spiking_decay=0.9,
                ):
                    output += token

                score = calculate_coherence_score(output)
                total_score += score
                outputs[lang] = output
                print(f"    [{lang}] Score: {score} | {output[:40]}...")
            except Exception as e:
                print(f"    [{lang}] Error: {e}")

        results.append((scale, total_score, outputs))

    # Encontrar el mejor balance
    best_res = max(results, key=lambda x: x[1])

    print("\n" + "=" * 60)
    print("✅ CALIBRACIÓN POLÍGLOTA FINALIZADA")
    print(f"🚀 Mejor h_scale balanceado: {best_res[0]}")
    for lang, out in best_res[2].items():
        print(f"📝 Salida {lang}: {out}")
    print("=" * 60)

    return best_res[0]


if __name__ == "__main__":
    calibrate_homeostasis("models/checkpoints/polyglot_organism.gaje")
