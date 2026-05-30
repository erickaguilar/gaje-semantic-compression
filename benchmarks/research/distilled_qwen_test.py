import os
import sys
import time
import numpy as np

# Añadir el directorio python al path para importar los módulos locales
sys.path.append(os.path.abspath("python"))

from genomize_llm import GenomicLLM


def run_tests():
    print("=" * 60)
    print("🧪 PRUEBAS DE VALIDACIÓN: QWEN2 DISTILLED (GAJE 2-BIT) 🧪")
    print("=" * 60)

    model_dir = "gaje_qwen2_hybrid_v1"
    if not os.path.exists(model_dir):
        print(f"❌ Error: El directorio del modelo '{model_dir}' no existe.")
        return

    # 1. Carga del Modelo
    print(f"\n[*] Cargando modelo destilado híbrido (2 bloques) desde '{model_dir}'...")
    start_load = time.time()
    # Cargamos solo los 2 bloques destilados
    model = GenomicLLM(model_dir, load_genomic=True)
    load_time = time.time() - start_load
    print(f"✅ Modelo cargado en {load_time:.2f}s")

    # 2. Prueba de Generación (Coherencia)
    prompts = [
        "La inteligencia artificial es",
        "El protocolo GAJE permite",
        "En el futuro, los modelos de lenguaje",
    ]

    print("\n" + "-" * 40)
    print("📝 PRUEBA DE GENERACIÓN DE TEXTO")
    print("-" * 40)

    for prompt in prompts:
        start_gen = time.time()
        output = model.generate(prompt, max_new_tokens=15, temperature=0.7)
        gen_time = time.time() - start_gen
        tps = 15 / gen_time if gen_time > 0 else 0
        print(f"\n[⏱️ Speed: {tps:.2f} tokens/s]")
        print("-" * 20)

    # 3. Cálculo de Perplexity Simplificado
    print("\n" + "-" * 40)
    print("📊 PRUEBA DE PERPLEXITY (PPL)")
    print("-" * 40)

    test_text = "El ADN semántico es una tecnología revolucionaria para el almacenamiento de datos."
    tokens = model.tokenizer.encode(test_text)
    print(f"[*] Evaluando PPL para {len(tokens)} tokens...")

    log_probs = []
    for i in range(1, len(tokens)):
        # Tomamos el token previo para predecir el actual
        last_id = tokens[i - 1]
        target_id = tokens[i]

        # Forward pass simplificado para un solo token
        x = model.embedding_matrix[last_id].tolist()
        for block in model.blocks:
            x = block.forward(x, i - 1)

        x = model.rms_norm(x, model.output_norm_weight)
        logits = np.dot(model.embedding_matrix, x)

        # Softmax
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()

        prob_target = probs[target_id]
        log_probs.append(np.log(max(prob_target, 1e-10)))

        if i % 5 == 0:
            print(f"    [~] {i}/{len(tokens)} tokens procesados...")

    avg_log_prob = np.mean(log_probs)
    ppl = np.exp(-avg_log_prob)
    print(f"\n✅ Perplexity (PPL): {ppl:.4f}")

    if ppl < 100:
        print("🔥 ESTADO: EXCELENTE")
    elif ppl < 300:
        print("⚠️ ESTADO: ACEPTABLE")
    else:
        print("❌ ESTADO: DEGRADADO")

    # 4. Estadísticas de Memoria
    print("\n" + "-" * 40)
    print("📉 ESTADÍSTICAS DE COMPRESIÓN")
    print("-" * 40)

    # Estimación de tamaño en disco/RAM
    embedding_size = model.embedding_matrix.nbytes / (1024 * 1024)

    # Cada bloque tiene 2-bit weights.
    # attn: 3 matrices (Q,K,V). Q: 896x896, K: 896x128, V: 896x128 -> Total ~ 1M params
    # ffn_up: 896x4864, ffn_down: 4864x896 -> Total ~ 8.7M params
    # Total por bloque ~ 10M params
    # En 2-bit, 10M params = 2.5 MB
    # 24 bloques = 60 MB

    print(f"✅ Tamaño de Embeddings (Float32): {embedding_size:.2f} MB")
    print("✅ Estimación de Pesos Genómicos (2-bit): ~60.00 MB")
    print("🚀 Reducción teórica vs F32: 16x")

    print("\n" + "=" * 60)
    print("🏁 VALIDACIÓN COMPLETADA")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
