import numpy as np


def run_adc_simulation():
    print("🧪 GAJE PROTOCOL: ADC (ASYMMETRIC) SIMULATION 🧪")
    print("-" * 50)

    dims = 768
    num_records = 1000
    top_k = 10

    # 1. Generate normal distribution data
    data = np.random.normal(0, 0.5, (num_records, dims)).astype(np.float32)

    # 2. Define Gray Code Centroids (A, C, G, T)
    # Mapping: 00 -> A, 01 -> C, 11 -> G, 10 -> T
    # Based on our -0.34, 0, 0.34 thresholds for std=0.5
    centroids = {
        0b00: -0.6,  # A
        0b01: -0.2,  # C
        0b11: 0.2,  # G
        0b10: 0.6,  # T
    }

    def quantize_val(v):
        if v < -0.34:
            return 0b00
        if v < 0.0:
            return 0b01
        if v < 0.34:
            return 0b11
        return 0b10

    # 3. Compress DB into DNA (2-bit)
    db_dna = []
    for row in data:
        dna_row = [quantize_val(x) for x in row]
        db_dna.append(dna_row)

    # 4. Comparative Search (SDC vs ADC)
    q_idx = 42
    query_vec = data[q_idx]

    # --- GROUND TRUTH ---
    truth_indices = []
    for idx, row in enumerate(data):
        dist = np.linalg.norm(query_vec - row)
        truth_indices.append((idx, dist))
    truth_indices.sort(key=lambda x: x[1])
    truth_top_k = [x[0] for x in truth_indices[:top_k]]

    # --- SDC (Symmetric: DNA vs DNA) ---
    query_dna = [quantize_val(x) for x in query_vec]
    sdc_results = []
    for idx, dna_row in enumerate(db_dna):
        # Hamming distance estimate
        dist = sum([bin(a ^ b).count("1") for a, b in zip(query_dna, dna_row)])
        sdc_results.append((idx, dist))
    sdc_results.sort(key=lambda x: x[1])
    sdc_top_k = [x[0] for x in sdc_results[:top_k]]

    # --- ADC (Asymmetric: Float Query vs DNA Centroids) ---
    adc_results = []
    for idx, dna_row in enumerate(db_dna):
        # Compare original float query against centroids of the DNA strand
        reconstructed_centroids = np.array([centroids[bits] for bits in dna_row])
        dist = np.linalg.norm(query_vec - reconstructed_centroids)
        adc_results.append((idx, dist))
    adc_results.sort(key=lambda x: x[1])
    adc_top_k = [x[0] for x in adc_results[:top_k]]

    # 5. Results
    sdc_overlap = len(set(truth_top_k).intersection(set(sdc_top_k))) / top_k
    adc_overlap = len(set(truth_top_k).intersection(set(adc_top_k))) / top_k

    print(f"RESULTS (Top-{top_k} Overlap):")
    print(f"  SDC (Symmetric/Hamming): {sdc_overlap * 100:.2f}%")
    print(f"  ADC (Asymmetric/Float):  {adc_overlap * 100:.2f}% 🚀")
    print("-" * 50)

    improvement = ((adc_overlap / sdc_overlap) - 1) * 100 if sdc_overlap > 0 else 100
    print(f"PRECISION BOOST: {improvement:.2f}%")


if __name__ == "__main__":
    run_adc_simulation()
