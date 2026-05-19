import time
import numpy as np
import torch
import os
import sys

# Ajustar paths para importar el core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "python")))

from gaje.core._impl import RustGenomicBlock, GenomicAttention, GenomicLinear
# ArchConfig y ModelConfig pueden no estar expuestos de la misma forma o tener nombres distintos

def benchmark_swiglu():
    print("🧬 GAJE BENCHMARK: SwiGLU Vectorization (Rayon) 🧬")
    print("-" * 60)
    
    # Configuración típica de un LLM (ej. 768 embd, 3072 FFN hidden)
    n_embd = 768
    ffn_hidden = 3072
    block_size = 32
    
    # Mock de un bloque genómico
    print("[*] Inicializando bloque de prueba (768 hidden, 3072 ffn)...")
    
    # Generar componentes mínimos
    def get_linear(in_f, out_f):
        dna = bytes(np.random.randint(0, 256, (out_f * in_f // 4,), dtype=np.uint8).tolist())
        anchors = bytes([0] * (out_f * in_f * 2)) # F16 zeros
        centroids = [0.0] * (out_f * in_f // block_size * 4)
        return GenomicLinear(dna, anchors, centroids, out_f, in_f, block_size)

    attn = GenomicAttention(12, 12, 64, [1.0]*n_embd, 1e-6, 10000.0)
    
    block = RustGenomicBlock(
        idx=0,
        attn=attn,
        q_gen=get_linear(n_embd, n_embd),
        k_gen=get_linear(n_embd, n_embd),
        v_gen=get_linear(n_embd, n_embd),
        w_o=get_linear(n_embd, n_embd),
        gate_gen=get_linear(n_embd, ffn_hidden),
        up_gen=get_linear(n_embd, ffn_hidden),
        w_down=get_linear(ffn_hidden, n_embd),
        ffn_norm=[1.0]*n_embd,
        eps=1e-6,
        act_fn="swiglu"
    )

    x = np.random.normal(0, 0.02, (n_embd,)).astype(np.float32).tolist()
    
    # Warmup
    print("[*] Calentando motor (Warmup)...")
    for _ in range(5):
        _ = block.forward(x, 0)
    
    # Benchmark
    iterations = 100
    print(f"[*] Ejecutando {iterations} iteraciones de forward pass...")
    
    start_time = time.time()
    for _ in range(iterations):
        _ = block.forward(x, 0)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = (total_time / iterations) * 1000
    
    print("-" * 60)
    print(f"⏱️ Latencia promedio por bloque: {avg_latency:.4f} ms")
    print(f"🚀 Tokens por segundo (simulado 1 bloque): {1000 / avg_latency:.2f} tps")
    
    # Si tenemos 24 bloques, la velocidad real sería:
    est_full_latency = avg_latency * 24
    print(f"📉 Velocidad estimada (24 bloques): {1000 / est_full_latency:.2f} tps")
    print("-" * 60)

if __name__ == "__main__":
    benchmark_swiglu()
