import time
import numpy as np
import os
import sys

# Ajustar paths para importar el core
sys.path.insert(0, os.path.abspath("python"))

try:
    from gaje.core._impl import GenomicLinear, RustGenomicBlock, GenomicAttention, dna_similarity_search_adc, quantize_embedding
except ImportError:
    print("❌ Error: Library not built. Run 'maturin develop' first.")
    sys.exit(1)

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def benchmark_search_adc():
    print("\n⚡ BENCHMARK 1: Search ADC (Asymmetric Distance Computation)")
    num_records, dims = 10000, 768
    db_vecs = np.array([normalize(v) for v in np.random.normal(0, 1, (num_records, dims)).astype(np.float32)])
    thresholds, centroids = [-0.34, 0.0, 0.34], [-0.68, -0.17, 0.17, 0.68]
    
    db_dna = [quantize_embedding(v.tolist(), thresholds) for v in db_vecs]
    query = db_vecs[0].tolist()
    
    start_time = time.time()
    _ = dna_similarity_search_adc(query, db_dna, centroids)
    latency = (time.time() - start_time) * 1000
    print(f"⏱️ Latencia de búsqueda ({num_records} registros): {latency:.2f} ms")
    print(f"🚀 Rendimiento: {int(num_records / (latency/1000)):,} registros/seg")

def benchmark_linear_anchors():
    print("\n🧬 BENCHMARK 2: Linear Layer + F16 Anchors")
    n_embd, out_features, block_size = 768, 3072, 32
    anchors_bytes = np.random.normal(0, 0.1, (out_features * n_embd,)).astype(np.float16).tobytes()
    dna = bytes(np.random.randint(0, 256, (out_features * n_embd // 4,), dtype=np.uint8).tolist())
    centroids = [0.0] * (out_features * n_embd // block_size * 4)
    
    layer = GenomicLinear(dna, anchors_bytes, centroids, out_features, n_embd, block_size)
    x = np.random.normal(0, 1.0, (n_embd,)).astype(np.float32).tolist()
    
    for _ in range(5): _ = layer.forward(x) # Warmup
    
    iterations = 50
    start_time = time.time()
    for _ in range(iterations): _ = layer.forward(x)
    avg_latency = ((time.time() - start_time) / iterations) * 1000
    print(f"⏱️ Latencia promedio (Linear {out_features}x{n_embd}): {avg_latency:.4f} ms")

def benchmark_full_block():
    print("\n🌀 BENCHMARK 3: Full Genomic Block (Attention + SwiGLU)")
    n_embd, ffn_hidden, block_size = 768, 3072, 32
    def get_linear(in_f, out_f):
        dna = bytes(np.random.randint(0, 256, (out_f * in_f // 4,), dtype=np.uint8).tolist())
        anchors = bytes([0] * (out_f * in_f * 2))
        centroids = [0.0] * (out_f * in_f // block_size * 4)
        return GenomicLinear(dna, anchors, centroids, out_f, in_f, block_size)

    attn = GenomicAttention(12, 12, 64, [1.0]*n_embd, 1e-6, 10000.0)
    block = RustGenomicBlock(0, attn, get_linear(n_embd, n_embd), get_linear(n_embd, n_embd), get_linear(n_embd, n_embd), get_linear(n_embd, n_embd), get_linear(n_embd, ffn_hidden), get_linear(n_embd, ffn_hidden), get_linear(ffn_hidden, n_embd), [1.0]*n_embd, 1e-6, "swiglu")
    x = np.random.normal(0, 0.02, (n_embd,)).astype(np.float32).tolist()
    
    for _ in range(5): _ = block.forward(x, 0)
    
    iterations = 100
    start_time = time.time()
    for _ in range(iterations): _ = block.forward(x, 0)
    avg_latency = ((time.time() - start_time) / iterations) * 1000
    print(f"⏱️ Latencia promedio por bloque: {avg_latency:.4f} ms")
    print(f"📉 Velocidad estimada (24 bloques): {1000 / (avg_latency * 24):.2f} tokens/s")

if __name__ == "__main__":
    print("=" * 60)
    print("🏟️ GAJE PERFORMANCE SUITE: ARM NATIVE BENCHMARK")
    print("=" * 60)
    benchmark_search_adc()
    benchmark_linear_anchors()
    benchmark_full_block()
    print("\n" + "=" * 60)
