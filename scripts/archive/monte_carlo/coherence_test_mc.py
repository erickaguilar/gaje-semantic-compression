import os
import sys
import numpy as np
import time

# Asegurar que usamos el código local de 'python/'
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
)

from gaje.core import _impl as gaje


def coherence_test_monte_carlo():
    print("🧬 GAJE-Flow: Prueba de Coherencia Semántica (Monte Carlo vs Baseline)")
    print("-" * 60)

    # 1. Configuración de la capa (Simulando una capa de embedding o proyección)
    out_features = 256
    in_features = 1024
    block_size = 32

    # 2. Generar Pesos de "Referencia" (Lo que el modelo intentaría representar)
    # Usamos una distribución con algo de estructura (no puro ruido)
    np.random.seed(42)
    W_ref = np.random.normal(0, 0.1, (out_features, in_features)).astype(np.float32)

    # 3. Genomización Inicial (2-bits)
    # Simulamos la base de datos de ADN
    # Nota: Usamos una simplificación para la prueba
    database = np.random.randint(
        0, 256, (out_features, in_features // 4), dtype=np.uint8
    )

    # Centroides base (Heurística estándar)
    std = np.std(W_ref)
    base_centroids = [-1.51 * std, -0.45 * std, 0.45 * std, 1.51 * std]
    all_centroids = base_centroids * (out_features * (in_features // block_size))

    layer = gaje.GenomicLinear(
        database.tobytes(), b"", all_centroids, out_features, in_features, block_size
    )

    # 4. Definir Input y Target "Coherente"
    # El target es lo que la capa F32 produciría idealmente
    input_vec = np.random.normal(0, 1.0, in_features).astype(np.float32)
    target_output = np.dot(W_ref, input_vec)

    def evaluate_coherence(lyr, inp, target, label):
        output = np.array(lyr.forward(inp.tolist()))
        mse = np.mean((output - target) ** 2)
        cos_sim = np.dot(output, target) / (
            np.linalg.norm(output) * np.linalg.norm(target)
        )

        # Entropía de los logits (indicador de "certeza" o coherencia)
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        probs = softmax(output)
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        print(f"[{label}]")
        print(f"   MSE:      {mse:.6f}")
        print(f"   Cos Sim:  {cos_sim:.6f}")
        print(f"   Entropy:  {entropy:.4f} bits")
        return mse, cos_sim, entropy

    # --- Evaluación Baseline ---
    mse_b, cos_b, ent_b = evaluate_coherence(
        layer, input_vec, target_output, "BASELINE (2-BIT)"
    )

    # 5. Aplicar Refinamiento Monte Carlo
    # Intentamos "reparar" la coherencia ajustando los centroides probabilísticamente
    print("\n[*] Ejecutando Refinamiento de Monte Carlo para restaurar coherencia...")
    start = time.time()
    layer.monte_carlo_refine(
        input_vec.tolist(), target_output.tolist(), iterations=2000, noise_scale=0.02
    )
    duration = time.time() - start

    # --- Evaluación Post-Monte Carlo ---
    print(f"[*] Refinamiento completado en {duration:.2f}s\n")
    mse_m, cos_m, ent_m = evaluate_coherence(
        layer, input_vec, target_output, "MONTE CARLO (OPTIMIZED)"
    )

    # 6. Conclusión
    print("-" * 60)
    print("📊 RESULTADO FINAL")
    improvement_mse = (mse_b - mse_m) / mse_b * 100
    improvement_cos = (cos_m - cos_b) * 100

    print(f"✅ Mejora en Precisión (MSE): {improvement_mse:.2f}%")
    print(f"✅ Incremento en Fidelidad (Cos): +{improvement_cos:.4f}")

    if cos_m > 0.95:
        print("🔥 ESTADO: ALTA COHERENCIA. El modelo ha recuperado la señal semántica.")
    elif cos_m > cos_b:
        print("⚠️ ESTADO: MEJORA DETECTADA. La coherencia ha aumentado.")
    else:
        print("❌ ESTADO: SIN CAMBIOS. Se requieren más iteraciones o ajustes.")


if __name__ == "__main__":
    coherence_test_monte_carlo()
