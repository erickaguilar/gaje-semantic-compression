import numpy as np


def standard_binary_quantize(val):
    if val < -0.5:
        return 0b00  # 0
    if val < 0.0:
        return 0b01  # 1
    if val < 0.5:
        return 0b10  # 2
    return 0b11  # 3


def gray_code_quantize(val):
    # Gray Code Sequence: 00, 01, 11, 10
    if val < -0.5:
        return 0b00  # 0
    if val < 0.0:
        return 0b01  # 1
    if val < 0.5:
        return 0b11  # 3 (Adjacent to 01 by 1 bit)
    return 0b10  # 2 (Adjacent to 11 by 1 bit)


def hamming_distance(a, b):
    return bin(a ^ b).count("1")


def run_gray_comparison():
    print("🧬 GRAY CODE VS STANDARD BINARY ANALYSIS 🧬")
    print("-" * 50)

    # Test values from -1.0 to 1.0
    np.linspace(-1.0, 1.0, 100)

    # Reference point (e.g., center of 'C' cluster)
    ref_val = -0.25
    ref_std = standard_binary_quantize(ref_val)
    ref_gray = gray_code_quantize(ref_val)

    print(f"Reference Value: {ref_val} (Cluster C)")
    print(f"  Standard Bits: {bin(ref_std)}")
    print(f"  Gray Bits:     {bin(ref_gray)}")
    print("\n[Analysis] Hamming Distance to Neighbors:")

    # Compare with a value in the next cluster (Cluster G: 0.25)
    neighbor_val = 0.25

    # Standard
    n_std = standard_binary_quantize(neighbor_val)
    dist_std = hamming_distance(ref_std, n_std)

    # Gray
    n_gray = gray_code_quantize(neighbor_val)
    dist_gray = hamming_distance(ref_gray, n_gray)

    print(f"Target Value: {neighbor_val} (Cluster G - Adjacent)")
    print(f"  Standard (01 -> 10): Distance = {dist_std} bits (🛑 BIG JUMP)")
    print(f"  Gray (01 -> 11):     Distance = {dist_gray} bits (✅ SMOOTH)")

    # Correlation Test
    print("\n[*] Measuring Correlation (Value distance vs Bit distance)...")
    vals = np.random.uniform(-1, 1, 1000)

    std_dists = []
    gray_dists = []
    real_dists = []

    for v in vals:
        real_dists.append(abs(v - ref_val))
        std_dists.append(hamming_distance(ref_std, standard_binary_quantize(v)))
        gray_dists.append(hamming_distance(ref_gray, gray_code_quantize(v)))

    corr_std = np.corrcoef(real_dists, std_dists)[0, 1]
    corr_gray = np.corrcoef(real_dists, gray_dists)[0, 1]

    print(f"Correlation Standard: {corr_std:.4f}")
    print(f"Correlation Gray:     {corr_gray:.4f}")

    improvement = ((corr_gray / corr_std) - 1) * 100
    print(f"\n🚀 IMPROVEMENT: {improvement:.2f}% better semantic mapping")


if __name__ == "__main__":
    run_gray_comparison()
