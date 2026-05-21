import os
import sys
import time
import numpy as np
import json
from gaje.core import _impl as dna_semantic_compression

# Configuración
# Escalas: Serious (100k), Very Serious (1M)
# 10M deshabilitado temporalmente para asegurar estabilidad en Termux
SCALES = [
    {"name": "Serious", "n": 100_000, "dims": 768},
    {"name": "Very Serious", "n": 1_000_000, "dims": 768},
]

# Parámetros biológicos (GAJE)
THRESHOLDS = [-0.34, 0.0, 0.34]
CENTROIDS = [-0.68, -0.17, 0.17, 0.68]

def run_scale(scale_info):
    name = scale_info["name"]
    n = scale_info["n"]
    dims = scale_info["dims"]
    
    print(f"\n--- 🧬 BENCHMARK SCALE: {name} ({n:,} embeddings) ---", flush=True)
    
    # 1. Generación y Cuantización
    print(f"[*] Generando {n:,} vectores y cuantizando...", flush=True)
    start_time = time.time()
    
    chunk_size = 50_000
    db_dna = []
    sample_size = 100
    sample_vecs = []
    
    for i in range(0, n, chunk_size):
        actual_chunk = min(chunk_size, n - i)
        vecs = np.random.normal(0, 1, (actual_chunk, dims)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / (norms + 1e-10)
        
        if i == 0:
            sample_vecs = vecs[:sample_size].copy()
            
        for v in vecs:
            db_dna.append(dna_semantic_compression.quantize_embedding(v.tolist(), THRESHOLDS))
            
    quant_time = time.time() - start_time
    est_mem = (n * 192) / (1024 * 1024)
    
    print(f"[+] Cuantización completada en {quant_time:.2f}s", flush=True)
    print(f"[+] Memoria DNA estimada: {est_mem:.2f} MB", flush=True)
    
    # 2. Búsqueda Flat ADC
    current_sample_size = 20 if n >= 1_000_000 else 100
    print(f"[*] Ejecutando búsqueda Flat ADC (Query: {current_sample_size})...", flush=True)
    latencies = []
    
    for i in range(current_sample_size):
        q = sample_vecs[i % len(sample_vecs)]
        start_search = time.time()
        _ = dna_semantic_compression.dna_similarity_search_adc(q.tolist(), db_dna, CENTROIDS)
        latencies.append(time.time() - start_search)
        
    avg_latency = np.mean(latencies) * 1000
    throughput = n / (avg_latency / 1000)
    
    print(f"[+] Latencia media (Flat): {avg_latency:.2f} ms", flush=True)
    print(f"[+] Throughput: {int(throughput):,} ops/seg", flush=True)
    
    # 3. Recall Check (Top-1 simple)
    print(f"[*] Validando Recall@1...", flush=True)
    q0 = sample_vecs[0].tolist()
    res0 = dna_semantic_compression.dna_similarity_search_adc(q0, db_dna, CENTROIDS)
    top1_found = res0[0][0] == 0
    print(f"[+] Top-1 Correcto (Query 0): {top1_found}", flush=True)

    return {
        "scale": name,
        "n": n,
        "quant_time_s": quant_time,
        "avg_latency_ms": avg_latency,
        "throughput_ops_s": int(throughput),
        "est_mem_mb": est_mem,
        "top1_correct": top1_found
    }

def main():
    results = []
    for scale in SCALES:
        try:
            res = run_scale(scale)
            results.append(res)
        except Exception as e:
            print(f"Error en escala {scale['name']}: {e}", flush=True)
            
    with open("benchmarks/logs/large_scale_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n🚀 Resultados finales guardados en benchmarks/large_scale_results.json", flush=True)

if __name__ == "__main__":
    main()
