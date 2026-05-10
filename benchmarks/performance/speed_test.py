import time
import numpy as np
from gaje.core import _impl as dna_semantic_compression

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def run_speed_test():
    print("⚡ GAJE SPEED BENCHMARK: PARALLEL RUST ENGINE ⚡")
    print("-" * 50)
    
    num_records = 10000
    dims = 768
    
    print(f"[*] Generating {num_records} vectors ({dims} dimensions)...")
    db_vecs = np.random.normal(0, 1, (num_records, dims)).astype(np.float32)
    db_vecs = np.array([normalize(v) for v in db_vecs])
    
    # Fake thresholds for quantization
    thresholds = [-0.34, 0.0, 0.34]
    centroids = [-0.68, -0.17, 0.17, 0.68]
    
    print("[*] Quantizing database...")
    db_dna = [dna_semantic_compression.quantize_embedding(v.tolist(), thresholds) for v in db_vecs]
    
    query = db_vecs[0].tolist()
    
    print(f"[*] Starting search over {num_records} strands...")
    start_time = time.time()
    
    # Run the parallel search
    results = dna_semantic_compression.dna_similarity_search_adc(query, db_dna, centroids)
    
    end_time = time.time()
    latency = (end_time - start_time) * 1000
    
    print("-" * 50)
    print(f"⏱️ Latencia de búsqueda: {latency:.2f} ms")
    print(f"🚀 Rendimiento: {int(num_records / (latency/1000)):,} registros/seg")
    print("-" * 50)

if __name__ == "__main__":
    run_speed_test()
