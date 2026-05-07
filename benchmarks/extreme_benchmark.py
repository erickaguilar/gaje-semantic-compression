import os
import sys
import time
import numpy as np
import json
import gc
import dna_semantic_compression

# Configuración Extrema
SCALES = [
    {"name": "Paper-grade", "n": 10_000_000, "dims": 768},
]

THRESHOLDS = [-0.34, 0.0, 0.34]
CENTROIDS = [-0.68, -0.17, 0.17, 0.68]

def run_scale(scale_info):
    name = scale_info["name"]
    n = scale_info["n"]
    dims = scale_info["dims"]
    
    print(f"\n--- 🧬 EXTREME BENCHMARK: {name} ({n:,} embeddings) ---", flush=True)
    
    # 0. Inicializar Índice de Rust (vacio)
    # Esto mantendrá los datos en memoria de Rust, mucho más compacta que Python
    index = dna_semantic_compression.GajeIndex([], CENTROIDS)
    
    # 1. Generación y Cuantización
    print(f"[*] Generando {n:,} vectores y cuantizando a Rust (Incremental)...", flush=True)
    start_time = time.time()
    
    chunk_size = 10_000 # Reducido para mayor estabilidad en Termux
    sample_size = 5
    sample_vecs = []
    
    for i in range(0, n, chunk_size):
        actual_chunk = min(chunk_size, n - i)
        # Generar en lotes más pequeños para evitar picos de RAM en Numpy
        vecs = np.random.normal(0, 1, (actual_chunk, dims)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / (norms + 1e-10)
        
        if i == 0:
            sample_vecs = vecs[:sample_size].copy()
            
        # Cuantizar lote
        batch = []
        for v in vecs:
            # Ahora devuelve bytes, mucho más ligero que list[int]
            batch.append(dna_semantic_compression.quantize_embedding(v.tolist(), THRESHOLDS))
            
        # Enviar lote a Rust
        index.add_batch(batch)
        
        # Limpieza inmediata del lote de Python
        del batch
        del vecs
        del norms
        
        if i % 100_000 == 0 and i > 0:
            gc.collect() 
            print(f"    [~] Progreso: {i:,}/{n:,} embeddings en Rust Index...", flush=True)
            
    quant_time = time.time() - start_time
    est_mem = (n * 192) / (1024 * 1024)
    
    print(f"[+] Cuantización completada en {quant_time:.2f}s", flush=True)
    print(f"[+] Memoria DNA en Rust: {est_mem:.2f} MB (Cero overhead de Python)", flush=True)
    
    # 2. Búsqueda Flat ADC (Directo en Rust Index)
    print(f"[*] Ejecutando búsqueda Flat ADC desde Rust Index (Query: {sample_size})...", flush=True)
    latencies = []
    
    for i in range(sample_size):
        q = sample_vecs[i]
        start_search = time.time()
        # k=10 para limitar resultados y evitar picos de memoria en Python
        _ = index.search(q.tolist(), k=10)
        latencies.append(time.time() - start_search)
        
    avg_latency = np.mean(latencies) * 1000
    throughput = n / (avg_latency / 1000)
    
    print(f"[+] Latencia media (Flat): {avg_latency:.2f} ms", flush=True)
    print(f"[+] Throughput: {int(throughput):,} ops/seg", flush=True)
    
    # 3. Recall Check
    print(f"[*] Validando Recall@1...", flush=True)
    q0 = sample_vecs[0].tolist()
    res0 = index.search(q0, k=1)
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
            print(f"Error crítico en escala {scale['name']}: {e}", flush=True)
            
    with open("benchmarks/extreme_scale_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n🚀 Resultados guardados en benchmarks/extreme_scale_results.json", flush=True)

if __name__ == "__main__":
    main()
