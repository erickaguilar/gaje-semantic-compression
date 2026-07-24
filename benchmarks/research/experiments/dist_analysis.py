import numpy as np


def run_distribution_analysis():
    print("📊 ANALYZING VECTOR DISTRIBUTION 📊")
    dims = 768
    num_samples = 1000

    # Real embeddings often are NOT uniform or normal across all dimensions
    # Let's check the mean and std of our synthetic normal data
    data = np.random.normal(0, 0.5, (num_samples, dims))

    # Static thresholds: -0.5, 0.0, 0.5
    # Let's count how many values fall into each 2-bit bucket
    buckets = {
        "A ( < -0.5)": np.sum(data < -0.5),
        "C (-0.5 to 0)": np.sum((data >= -0.5) & (data < 0)),
        "G (0 to 0.5)": np.sum((data >= 0) & (data < 0.5)),
        "T ( > 0.5)": np.sum(data >= 0.5),
    }

    total = num_samples * dims
    print(f"Total elements: {total}")
    for k, v in buckets.items():
        print(f"  {k}: {v} ({v / total * 100:.2f}%)")


if __name__ == "__main__":
    run_distribution_analysis()
