import numpy as np
import os
import sys

# Asegurar que se usa el paquete local
sys.path.insert(0, os.path.abspath("python"))

from gaje.core import _impl as engine
from gaje.processing.balancer import SignalToNoiseBalancer


def test_native_precision_mask():
    print("🔬 Probando 'generate_precision_mask_native'...")
    entropy = np.array([0.1, 0.5, 0.9, 1.2, 0.05, 2.0, 0.3])

    # Python Version (via Balancer class)
    balancer = SignalToNoiseBalancer()
    mask_py = balancer.generate_precision_mask(entropy, fidelity_level=0.8)

    # Native Version
    mask_native_bytes = engine.generate_precision_mask_native(entropy.tolist(), 0.8)
    mask_native = np.frombuffer(mask_native_bytes, dtype=np.uint8)

    print(f"   - Entropy: {entropy}")
    print(f"   - Mask (PY):     {mask_py}")
    print(f"   - Mask (Native): {mask_native}")

    assert np.array_equal(
        mask_py, mask_native
    ), "Mismatch between Python and Native mask generation!"
    print("✅ MÁSCARA DE PRECISIÓN NATIVA VALIDADA")


def test_native_active_dims():
    print("\n🔬 Probando 'get_active_dimensions_native'...")
    entropy = np.array([0.001, 0.05, 0.005, 0.1, 0.2, 0.0001])
    threshold = 0.01

    # Python expected
    expected_dims = np.where(entropy > threshold)[0].tolist()

    # Native
    native_dims = engine.get_active_dimensions_native(entropy.tolist(), threshold)

    print(f"   - Threshold: {threshold}")
    print(f"   - Expected Indices: {expected_dims}")
    print(f"   - Native Indices:   {native_dims}")

    assert expected_dims == native_dims, "Mismatch in active dimensions filtering!"
    print("✅ FILTRADO DE DIMENSIONES NATIVO VALIDADO")


if __name__ == "__main__":
    try:
        test_native_precision_mask()
        test_native_active_dims()
    except Exception as e:
        print(f"❌ Error durante la validación: {e}")
        sys.exit(1)
