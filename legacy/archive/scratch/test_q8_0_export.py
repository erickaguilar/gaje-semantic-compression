import numpy as np
import gaje.core._impl as dna_semantic_compression
from gaje.nn.stabilized import GenomicLayer


def test_q8_0_quantization():
    print("Testing Q8_0 Quantization and Dequantization via PyO3...")

    # Shape: 64 rows, 128 columns (divisible by 32 block size)
    out_features = 64
    in_features = 128

    # Generate random weights in range [-2.0, 2.0]
    np.random.seed(42)
    original_weights = np.random.uniform(
        -2.0, 2.0, size=(out_features, in_features)
    ).astype(np.float32)

    # Instantiate GenomicLayer in Q8_0 mode (quant_format = 2, bit_depth = 8)
    layer = GenomicLayer(
        name="test_layer",
        weights_f32_or_tensor=original_weights,
        bit_depth=8,
        quant_format=2,
        block_size=32,
    )

    print(f"Layer dna_database size in bytes: {len(layer.dna_database)}")
    expected_size = out_features * (in_features // 32) * 34
    print(f"Expected size in bytes: {expected_size}")
    assert len(layer.dna_database) == expected_size, "Database size mismatch!"

    # Dequantize using the native Rust dequantize function
    reconstructed = np.array(
        dna_semantic_compression.dequantize_q8_0_native(
            layer.dna_database, out_features, in_features
        ),
        dtype=np.float32,
    ).reshape(out_features, in_features)

    # Compute Cosine Similarity
    dot_prod = np.sum(original_weights * reconstructed, axis=-1)
    norm_orig = np.linalg.norm(original_weights, axis=-1)
    norm_recon = np.linalg.norm(reconstructed, axis=-1)
    cossim = dot_prod / (norm_orig * norm_recon + 1e-9)
    mean_cossim = np.mean(cossim)

    print(f"Mean Cosine Similarity between original and Q8_0: {mean_cossim:.6f}")
    assert mean_cossim > 0.999, f"Cosine similarity too low: {mean_cossim}"

    # Compute MSE
    mse = np.mean((original_weights - reconstructed) ** 2)
    print(f"MSE: {mse:.6f}")
    assert mse < 1e-4, f"MSE too high: {mse}"

    print("✅ Q8_0 Quantization and Dequantization test passed successfully!")


if __name__ == "__main__":
    test_q8_0_quantization()
