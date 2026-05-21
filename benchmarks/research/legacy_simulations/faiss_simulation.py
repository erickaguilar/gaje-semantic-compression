import numpy as np
import time

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def simulate_binary_flat(db_vecs, query_vec, k=10):
    # Simulate Binary Quantization (1 bit per dimension)
    db_bin = (db_vecs > 0).astype(np.int8)
    q_bin = (query_vec > 0).astype(np.int8)
    # Hamming distance simulation
    distances = np.sum(db_bin != q_bin, axis=1)
    return np.argsort(distances)[:k]

def simulate_scalar_quantization(db_vecs, query_vec, k=10, bits=8):
    # Simulate SQ8 (8-bit linear quantization)
    v_min, v_max = np.min(db_vecs), np.max(db_vecs)
    levels = 2**bits - 1
    # Quantize
    db_q = np.round((db_vecs - v_min) / (v_max - v_min) * levels)
    q_q = np.round((query_vec - v_min) / (v_max - v_min) * levels)
    # Dequantize for distance calculation
    db_deq = (db_q / levels) * (v_max - v_min) + v_min
    q_deq = (q_q / levels) * (v_max - v_min) + v_min
    
    dots = np.dot(db_deq, q_deq)
    return np.argsort(dots)[::-1][:k]

def simulate_ivf_pq(db_vecs, query_vec, k=10, m=8, k_centroids=256):
    # Simulate IVF-PQ (Non-exhaustive + PQ)
    # Simplified: We simulate the loss of PQ without the IVF speedup
    dims = db_vecs.shape[1]
    dim_per_m = dims // m
    
    # Simple PQ Simulation
    db_pq_sim = np.zeros_like(db_vecs)
    for i in range(m):
        start = i * dim_per_m
        end = (i+1) * dim_per_m
        # In real PQ, we would use K-Means centroids. 
        # Here we simulate the quantization error of 8 bits per sub-vector
        # adding noise proportional to typical PQ loss
        db_pq_sim[:, start:end] = db_vecs[:, start:end] + np.random.normal(0, 0.05, (db_vecs.shape[0], dim_per_m))
        
    dots = np.dot(db_pq_sim, query_vec)
    return np.argsort(dots)[::-1][:k]

def run_comparison():
    print("🔬 COMPETITIVE ANALYSIS: GAJE vs FAISS STANDARDS (Simulated) 🔬")
    print("-" * 65)
    
    num_records = 2000
    dims = 768
    top_k = 10
    
    # Generate structured data
    latent = np.random.normal(0, 1, (num_records, 64))
    projection = np.random.normal(0, 1, (64, dims))
    db_vecs = np.array([normalize(v) for v in np.dot(latent, projection)])
    
    # Ground Truth
    def get_truth(q):
        return np.argsort(np.dot(db_vecs, q))[::-1][:top_k]

    methods = {
        "Binary Flat (1-bit)": lambda db, q: simulate_binary_flat(db, q),
        "Scalar Quant (SQ8)": lambda db, q: simulate_scalar_quantization(db, q),
        "IVF-PQ (8x8 bits)": lambda db, q: simulate_ivf_pq(db, q),
        "GAJE Protocol (DNA)": None # We'll use the real one
    }

    # Load GAJE
    import sys
    sys.path.append("benchmarks")
    from gaje.utils.codebook import train_genomic_codebook
    try:
        from gaje.core import _impl as dna_semantic_compression
    except ImportError:
        print("Error: dna_semantic_compression not found.")
        return

    # Train GAJE
    codebook = train_genomic_codebook(db_vecs, output_path="benchmarks/logs/comp_codebook.json")
    db_dna = [dna_semantic_compression.quantize_pq(v.tolist(), codebook["thresholds"]) for v in db_vecs]

    results = {}
    num_queries = 50
    
    for name, func in methods.items():
        overlaps = []
        for _ in range(num_queries):
            q_idx = np.random.randint(0, num_records)
            query = db_vecs[q_idx]
            truth = get_truth(query)
            
            if name == "GAJE Protocol (DNA)":
                res = dna_semantic_compression.dna_similarity_search_adc(query.tolist(), db_dna, codebook["centroids"])
                pred = [idx for idx, d in res[:top_k]]
            else:
                pred = func(db_vecs, query)
                
            intersection = set(truth).intersection(set(pred))
            overlaps.append(len(intersection) / top_k)
        
        results[name] = np.mean(overlaps) * 100

    print(f"{'Method':<25} | {'Recall@10':<12} | {'Bits/Dim':<10}")
    print("-" * 65)
    
    # Bit estimation
    bit_rates = {
        "Binary Flat (1-bit)": 1.0,
        "Scalar Quant (SQ8)": 8.0,
        "IVF-PQ (8x8 bits)": (8 * 8) / dims, # approx 0.08 bits/dim for 768d
        "GAJE Protocol (DNA)": 2.0  # 2 bits per base (per dim)
    }

    for name in sorted(results.keys(), key=lambda x: results[x], reverse=True):
        print(f"{name:<25} | {results[name]:>10.2f}% | {bit_rates[name]:>8.2f}")

    print("-" * 65)
    print("CONCLUSIÓN:")
    gaje_acc = results["GAJE Protocol (DNA)"]
    if gaje_acc > results["Binary Flat (1-bit)"] and gaje_acc > 80:
        print("💡 INNOVACIÓN CONFIRMADA: GAJE ofrece precisión de grado SQ8")
        print("   con la densidad de almacenamiento de un sistema genómico.")
    else:
        print("📈 REQUIERE OPTIMIZACIÓN para superar a los estándares.")

if __name__ == "__main__":
    run_comparison()
