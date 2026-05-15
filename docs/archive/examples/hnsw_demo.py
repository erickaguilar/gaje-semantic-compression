from gaje.core import _impl as dna_semantic_compression
import numpy as np
import time

def run_hnsw_demo():
    print("🕸️ GAJE PROTOCOL: HNSW GENOMIC INDEX DEMO 🕸️")
    print("-" * 50)
    
    num_records = 5000
    dims = 768
    
    print(f"[*] Generating {num_records} vectors...")
    data = np.random.normal(0, 1, (num_records, dims)).astype(np.float32)
    
    thresholds = [-0.34, 0.0, 0.34]
    centroids = [-0.68, -0.17, 0.17, 0.68]
    
    print("[*] Quantizing database into DNA strands...")
    db_dna = [dna_semantic_compression.quantize_embedding(v.tolist(), thresholds) for v in data]
    
    # Initialize GajeIndex
    print("[*] Initializing GajeIndex (HNSW)...")
    index = dna_semantic_compression.GajeIndex(db_dna, centroids, m=16, ef_construction=100)
    
    # Build the graph
    start_build = time.time()
    index.build()
    end_build = time.time()
    print(f"[+] Index built in {(end_build - start_build)*1000:.2f} ms")
    
    # Search
    query = data[0].tolist()
    print("[*] Searching with HNSW...")
    start_search = time.time()
    results = index.search(query, ef=50)
    end_search = time.time()
    
    print(f"[+] Search completed in {(end_search - start_search)*1000:.2f} ms")
    print(f"[+] Top result index: {results[0][0]}")
    print("-" * 50)

if __name__ == "__main__":
    run_hnsw_demo()
