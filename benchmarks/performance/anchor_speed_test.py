import time
import numpy as np
import os
import sys

# Ajustar paths para importar el core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "python")))

from gaje.core._impl import GenomicLinear

def benchmark_anchors():
    print("🧬 GAJE BENCHMARK: F16 Anchor Acceleration 🧬")
    print("-" * 60)
    
    n_embd = 768
    out_features = 3072
    block_size = 32
    
    # Generar componentes con anclas
    print(f"[*] Generando capa con anclas ({out_features} x {n_embd})...")
    
    # Simular anclas reales (no todas son cero para que el cómputo sea real)
    anchors_np = np.random.normal(0, 0.1, (out_features * n_embd,)).astype(np.float16)
    anchors_bytes = anchors_np.tobytes()
    
    dna = bytes(np.random.randint(0, 256, (out_features * n_embd // 4,), dtype=np.uint8).tolist())
    centroids = [0.0] * (out_features * n_embd // block_size * 4)
    
    layer = GenomicLinear(
        dna,
        anchors_bytes,
        centroids,
        out_features,
        n_embd,
        block_size
    )

    x = np.random.normal(0, 1.0, (n_embd,)).astype(np.float32).tolist()
    
    # Warmup
    print("[*] Calentando motor...")
    for _ in range(5):
        _ = layer.forward(x)
    
    # Benchmark
    iterations = 50
    print(f"[*] Ejecutando {iterations} iteraciones de forward con anclas...")
    
    start_time = time.time()
    for _ in range(iterations):
        _ = layer.forward(x)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = (total_time / iterations) * 1000
    
    print("-" * 60)
    print(f"⏱️ Latencia promedio (Linear + Anchors): {avg_latency:.4f} ms")
    print(f"🚀 Throughput: {iterations / total_time:.2f} ops/s")
    print("-" * 60)

if __name__ == "__main__":
    benchmark_anchors()
