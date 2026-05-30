import os
import sys
import json
import time
import numpy as np

# Asegurar uso de código local
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
)

import gaje.core._impl as engine
from gaje.nn.stabilized import GenomicLLM


def calculate_coherence_v095(model_path, dataset_path, num_samples=100):
    print(f"📊 Iniciando Coherence Benchmark (v0.9.5-alpha) para: {model_path}")

    # 1. Cargar Modelo
    reader = engine.GajeDatabaseReader(model_path)
    config = json.loads(reader.read_metadata("config"))
    centroides = config["centroides"]
    thresholds = config.get("thresholds", [0.4, 0.4, 0.4])

    layers = []
    for i, layer_cfg in enumerate(config["layers"]):
        l = engine.GajeNeuromorphicLayer(
            layer_cfg["out"], layer_cfg["in"], thresholds[i], 0.8
        )
        packed = reader.read_tensor(f"layer.{i}.packed_weights")
        l.load_packed_weights(packed)
        layers.append(l)

    # 2. Tokenizador
    teacher = GenomicLLM("models/gguf/smollm2-135m-f16.gguf")
    tokenizer = teacher.tokenizer

    # 3. Cargar Dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if len(line.strip()) > 5]

    print(f"[*] Dataset cargado: {len(lines)} líneas útiles.")
    np.random.shuffle(lines)

    # 4. Métricas
    hits = 0
    total_tokens = 0
    spike_ratios = []
    prediction_latencies = []

    print(f"[*] Evaluando {num_samples} muestras...")

    start_bench = time.time()
    for idx, text in enumerate(lines[:num_samples]):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if hasattr(tokens, "ids"):
            tokens = tokens.ids

        if len(tokens) < 2:
            continue

        for t in range(len(tokens) - 1):
            input_id = tokens[t]
            target_id = tokens[t + 1]

            t_start = time.time()
            # Forward Spiking
            layers[0].integrate_batch(input_id, centroides, 1.0)
            s0 = layers[0].check_spikes()

            if s0:
                for n_idx, intensity, _ in s0:
                    layers[1].integrate_batch(n_idx, centroides, intensity)
                s1 = layers[1].check_spikes()

                if s1:
                    for n_idx, intensity, _ in s1:
                        layers[2].integrate_batch(n_idx, centroides, intensity)
                    s2 = layers[2].check_spikes()

                    if s2:
                        spike_ratios.append(len(s2))
                        prediction_latencies.append(time.time() - t_start)

                        top_5 = sorted(s2, key=lambda x: x[1], reverse=True)[:5]
                        if any(s[0] == target_id for s in top_5):
                            hits += 1
                    else:
                        if total_tokens % 100 == 0:
                            print(f"      [Debug] L2 SILENCIO | S1 count: {len(s1)}")
                else:
                    if total_tokens % 100 == 0:
                        print(f"      [Debug] L1 SILENCIO | S0 count: {len(s0)}")
            else:
                if total_tokens % 100 == 0:
                    print(f"      [Debug] L0 SILENCIO | Input ID: {input_id}")

            total_tokens += 1
            for l in layers:
                l.apply_homeostasis(0.0)

        if (idx + 1) % 10 == 0:
            current_acc = (hits / max(1, total_tokens)) * 100
            print(
                f"    [~] Muestra {idx+1}/{num_samples} | Hits: {hits}/{total_tokens} | Acc: {current_acc:.2f}%"
            )

    duration = time.time() - start_bench

    # 5. Reporte
    accuracy = (hits / max(1, total_tokens)) * 100
    avg_spikes = np.mean(spike_ratios) if spike_ratios else 0
    avg_latency = np.mean(prediction_latencies) * 1000 if prediction_latencies else 0

    print("-" * 50)
    print("✅ RESULTADOS COHERENCIA (v0.9.5-alpha)")
    print(f"[*] Precisión Top-5 (Semántica): {accuracy:.2f}%")
    print(f"[*] Actividad Promedio (Spikes/Token): {avg_spikes:.1f}")
    print(f"[*] Latencia de Inferencia (Sincronía): {avg_latency:.2f} ms")
    print(f"[*] Tiempo Total de Benchmark: {duration:.2f}s")
    print("-" * 50)


if __name__ == "__main__":
    calculate_coherence_v095(
        "models/checkpoints/smg1_debug.gaje", "data/datasets/tiny_shakespeare.txt"
    )
