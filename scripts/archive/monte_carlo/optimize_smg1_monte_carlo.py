import os
import sys
import json
import time
import numpy as np
import argparse

# Asegurar uso de código local
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
)

import gaje.core._impl as engine
from gaje.nn.stabilized import GenomicLLM
from gaje.utils.version import get_project_version


def evaluate_fitness(
    layers, dataset, tokenizer, centroides, thresholds, sample_size=50
):
    """Evalúa la aptitud basada en flujo de señal y precisión de hits."""
    hits = 0
    tokens_processed = 0
    flow_score = 0

    # Recreamos las capas con los nuevos umbrales para evaluación precisa
    # (Ya que no hay setter para threshold en el binding actual)
    test_layers = []
    for i, layer in enumerate(layers):
        l = engine.GajeNeuromorphicLayer(
            layer.num_neurons, layer.weights_per_neuron, thresholds[i], 0.8
        )
        l.load_packed_weights(layer.packed_weights)
        test_layers.append(l)

    for text in dataset[:sample_size]:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if hasattr(tokens, "ids"):
            tokens = tokens.ids
        if len(tokens) < 2:
            continue

        for t in range(len(tokens) - 1):
            input_id = tokens[t]
            target_id = tokens[t + 1]

            # Capa 0
            test_layers[0].integrate_batch(input_id, centroides, 1.0)
            s0 = test_layers[0].check_spikes()
            if not s0:
                continue
            flow_score += 1

            # Capa 1
            for idx, intensity, _ in s0:
                test_layers[1].integrate_batch(idx, centroides, intensity)
            s1 = test_layers[1].check_spikes()
            if not s1:
                continue
            flow_score += 2

            # Capa 2
            for idx, intensity, _ in s1:
                test_layers[2].integrate_batch(idx, centroides, intensity)
            s2 = test_layers[2].check_spikes()
            if not s2:
                continue
            flow_score += 3

            if any(spike[0] == target_id for spike in s2):
                hits += 1
            tokens_processed += 1

            for l in test_layers:
                l.apply_homeostasis(0.0)

    # El score prioriza flujo y luego precisión
    normalized_flow = flow_score / (max(1, tokens_processed) * 6)
    accuracy = (hits / tokens_processed) if tokens_processed > 0 else 0

    return normalized_flow + (accuracy * 10)


def main():
    parser = argparse.ArgumentParser(
        description="🎲 GAJE: Monte Carlo Centroid Optimization (Phase 2)"
    )
    parser.add_argument(
        "--model", type=str, default="models/checkpoints/smg1_overnight_gold.gaje"
    )
    parser.add_argument(
        "--dataset", type=str, default="data/datasets/tiny_shakespeare.txt"
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--sample_size", type=int, default=20)
    args = parser.parse_args()

    print("🎲 Iniciando Fase 2: Optimización Monte Carlo")
    print(f"[*] Modelo base: {args.model}")

    # 1. Cargar Modelo
    reader = engine.GajeDatabaseReader(args.model)
    config = json.loads(reader.read_metadata("config"))
    centroides_base = config["centroides"]

    with open(args.dataset, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if len(line.strip()) > 10]
    np.random.shuffle(lines)

    teacher = GenomicLLM("models/gguf/smollm2-135m-f16.gguf")
    tokenizer = teacher.tokenizer

    # Cargar capas originales
    base_layers = []
    for i, layer_cfg in enumerate(config["layers"]):
        l = engine.GajeNeuromorphicLayer(layer_cfg["out"], layer_cfg["in"], 0.4, 0.8)
        packed = reader.read_tensor(f"layer.{i}.packed_weights")
        l.load_packed_weights(packed)
        base_layers.append(l)

    # 2. Bucle Monte Carlo
    best_centroides = list(centroides_base)
    best_thresholds = [0.4, 0.4, 0.4]
    best_score = evaluate_fitness(
        base_layers,
        lines,
        tokenizer,
        best_centroides,
        best_thresholds,
        sample_size=args.sample_size,
    )

    print(f"[*] Score Inicial: {best_score:.4f}")
    start_time = time.time()

    for i in range(args.iterations):
        # Mutación aleatoria
        c_mutation = np.random.normal(0, args.noise, 4)
        t_mutation = np.random.normal(0, args.noise * 0.5, 3)

        candidate_c = [c + m for c, m in zip(best_centroides, c_mutation)]
        candidate_c.sort()

        candidate_t = [max(0.05, t + m) for t, m in zip(best_thresholds, t_mutation)]

        score = evaluate_fitness(
            base_layers,
            lines,
            tokenizer,
            candidate_c,
            candidate_t,
            sample_size=args.sample_size,
        )

        if score > best_score:
            best_score = score
            best_centroides = candidate_c
            best_thresholds = candidate_t
            print(
                f"🔥 Iter {i + 1}: Score: {best_score:.4f} | C: {best_centroides} | T: {best_thresholds}"
            )

    duration = time.time() - start_time
    print(f"\n✅ Optimización Finalizada en {duration:.1f}s")
    print(f"[*] Score Final: {best_score:.4f}")
    print(f"[*] Centroides Evolucionados: {best_centroides}")
    print(f"[*] Umbrales Evolucionados: {best_thresholds}")

    # 3. Guardado
    config["centroides"] = best_centroides
    config["thresholds"] = best_thresholds
    config["version"] = get_project_version()

    output_path = args.model.replace(".gaje", "_evolved.gaje")
    writer = engine.GajeDatabaseWriter(output_path)
    writer.write_metadata("config", json.dumps(config))

    for i in range(len(base_layers)):
        packed = reader.read_tensor(f"layer.{i}.packed_weights")
        writer.write_tensor_compressed(f"layer.{i}.packed_weights", packed)

    print(f"✨ Modelo Evolucionado guardado en: {output_path}")


if __name__ == "__main__":
    main()
