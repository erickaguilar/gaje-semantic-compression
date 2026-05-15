import numpy as np
from gaje.nn.stabilized import GenomicLayer
import time

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def silu_prime(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))

def test_swiglu_drift_analysis():
    print("🔬 Analizando Drift de Señal en SwiGLU (2-bit Genomic)...")
    np.random.seed(42) # Fijar semilla para test determinista
    dim = 128
    out_dim = 256 # Típico de FFN up/gate
    
    # 1. Crear pesos "maestros" (F32)
    w_gate_f32 = np.random.randn(out_dim, dim).astype(np.float32) * 0.1
    w_up_f32 = np.random.randn(out_dim, dim).astype(np.float32) * 0.1
    
    # 2. Crear capas genómicas (2-bit)
    print("[*] Comprimiendo a 2 bits...")
    layer_gate = GenomicLayer("gate", w_gate_f32, anchor_threshold=0.75)
    layer_up = GenomicLayer("up", w_up_f32, anchor_threshold=0.75)
    
    # 3. Input de prueba
    x = np.random.randn(dim).astype(np.float32)
    
    # 4. Referencia F32 (Sin ruido de cuantización)
    gate_f32 = np.dot(w_gate_f32, x)
    up_f32 = np.dot(w_up_f32, x)
    swiglu_f32 = silu(gate_f32) * up_f32
    
    # 5. Salida Genómica (Con ruido de 2 bits)
    gate_gen = layer_gate.forward(x)
    up_gen = layer_up.forward(x)
    swiglu_gen = silu(gate_gen) * up_gen
    
    # 6. Medir Error
    mse_linear = np.mean((gate_gen - gate_f32)**2)
    mse_swiglu = np.mean((swiglu_gen - swiglu_f32)**2)
    
    print(f"\n[!] Resultados del Drift:")
    print(f"    - MSE Linear (Gate): {mse_linear:.8f}")
    print(f"    - MSE SwiGLU (Final): {mse_swiglu:.8f}")
    
    amplification = mse_swiglu / mse_linear
    print(f"    - Factor de Amplificación de Ruido: {amplification:.2f}x")
    
    # El objetivo de la estabilización será reducir este factor de amplificación
    assert amplification > 1.0, "SwiGLU debería amplificar el ruido de cuantización."

if __name__ == "__main__":
    test_swiglu_drift_analysis()
