import os
import sys
import numpy as np
import time
import json

# Add 'python' directory to path to find the module
sys.path.append(os.path.abspath("python"))

try:
    import dna_semantic_compression
except ImportError:
    print("Error: Library not built. Run 'maturin develop' first.")
    sys.exit(1)

def run_benchmarks():
    results = {
        "compression": {},
        "latency": {},
        "search_throughput": {}
    }

    dims_list = [128, 384, 768, 1536]
    num_records = 1000

    print("🚀 Starting DNA Semantic Compression Benchmarks...")

    # 1. Compression Analysis
    for dims in dims_list:
        float32_size = dims * 4
        # Our engine packs 4 dimensions (2-bit each) into 1 byte
        dna_size = (dims + 3) // 4
        ratio = float32_size / dna_size
        results["compression"][f"{dims}_dims"] = {
            "float32_bytes": float32_size,
            "dna_bytes": dna_size,
            "ratio": f"{ratio:.1f}x",
            "saving_pct": f"{(1 - 1/ratio)*100:.2f}%"
        }

    # 2. Encoding/Decoding Latency
    dims = 768
    vector = np.random.uniform(-1, 1, dims).astype(np.float32).tolist()
    
    start = time.perf_counter()
    for _ in range(100):
        dna = dna_semantic_compression.quantize_embedding(vector)
    end = time.perf_counter()
    avg_enc = (end - start) / 100
    
    start = time.perf_counter()
    for _ in range(100):
        _ = dna_semantic_compression.dequantize_embedding(dna, dims)
    end = time.perf_counter()
    avg_dec = (end - start) / 100
    
    results["latency"][f"{dims}_dims"] = {
        "avg_encoding_ms": avg_enc * 1000,
        "avg_decoding_ms": avg_dec * 1000
    }

    # 3. Search Throughput (ADC)
    print(f"[*] Testing ADC search throughput with {num_records} records...")
    db_embeddings = np.random.uniform(-1, 1, (num_records, dims)).astype(np.float32)
    db_dna = [dna_semantic_compression.quantize_embedding(row.tolist()) for row in db_embeddings]
    query_vector = db_embeddings[0].tolist()

    start = time.perf_counter()
    _ = dna_semantic_compression.dna_similarity_search_adc(query_vector, db_dna)
    end = time.perf_counter()
    search_duration = end - start
    
    results["search_throughput"][f"{num_records}_records"] = {
        "duration_s": search_duration,
        "ops_per_sec": num_records / search_duration if search_duration > 0 else 0
    }

    print("✅ Benchmarks completed.")
    
    # Ensure directory exists
    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/raw_data.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to benchmarks/raw_data.json")
    return results

if __name__ == "__main__":
    run_benchmarks()
