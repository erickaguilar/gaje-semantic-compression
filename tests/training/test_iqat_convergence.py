import numpy as np
from gaje.nn.stabilized import GenomicLayer
from gaje.core import _impl as dna_core


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def test_iqat_swiglu_convergence():
    print("🔬 Validando Convergencia de IQAT (SwiGLU-Aware Refinement)...")
    dim = 64
    out_dim = 128

    # 1. Pesos F32 de referencia
    w_gate_f32 = np.random.randn(out_dim, dim).astype(np.float32) * 0.1
    w_up_f32 = np.random.randn(out_dim, dim).astype(np.float32) * 0.1

    # 2. Capas genómicas (2-bit)
    layer_gate = GenomicLayer("gate", w_gate_f32, anchor_threshold=0.75)
    layer_up = GenomicLayer("up", w_up_f32, anchor_threshold=0.75)

    # 3. Bloque Genómico (solo FFN para test)
    # Necesitamos mockear los otros componentes del bloque
    attn = dna_core.GenomicAttention(1, 1, dim, [], 1e-6, 10000.0)
    w_q = GenomicLayer("q", np.eye(dim).astype(np.float32)).linear
    w_k = GenomicLayer("k", np.eye(dim).astype(np.float32)).linear
    w_v = GenomicLayer("v", np.eye(dim).astype(np.float32)).linear
    w_o = GenomicLayer("o", np.eye(dim).astype(np.float32)).linear
    w_down = GenomicLayer("down", np.eye(out_dim, dim).astype(np.float32)).linear

    block = dna_core.RustGenomicBlock(
        0,
        attn,
        w_q,
        w_k,
        w_v,
        w_o,
        layer_gate.linear,
        layer_up.linear,
        w_down,
        [1.0] * dim,
        1e-6,
    )

    # 4. Input y Target
    x = np.random.randn(dim).astype(np.float32)
    # Target para la salida de SwiGLU (el "otro lado de la puerta")
    swiglu_target = silu(np.dot(w_gate_f32, x)) * np.dot(w_up_f32, x)

    # 5. Error Inicial
    print(f"[*] Suma de centroides gate (init): {np.sum(block.gate_gen.centroids):.6f}")
    gate_init = layer_gate.forward(x)
    up_init = layer_up.forward(x)
    swiglu_init = silu(gate_init) * up_init
    initial_error = np.mean((swiglu_init - swiglu_target) ** 2)
    print(f"[*] MSE SwiGLU Inicial: {initial_error:.8f}")

    # 6. Refinamiento IQAT
    lr = 0.005
    iterations = 50
    for i in range(iterations):
        block.refine_ffn(x.tolist(), swiglu_target.tolist(), lr)

    # 7. Error Final
    print(
        f"[*] Suma de centroides gate (final): {np.sum(block.gate_gen.centroids):.6f}"
    )
    gate_final = np.array(block.gate_gen.forward(x.tolist(), True))
    up_final = np.array(block.up_gen.forward(x.tolist(), True))
    swiglu_final = silu(gate_final) * up_final
    final_error = np.mean((swiglu_final - swiglu_target) ** 2)
    print(f"[*] MSE SwiGLU Final:   {final_error:.8f}")

    improvement = (initial_error - final_error) / initial_error * 100
    print(f"🚀 Mejora IQAT: {improvement:.2f}%")

    assert final_error < initial_error, "IQAT no redujo el error a través de SwiGLU."


if __name__ == "__main__":
    test_iqat_swiglu_convergence()
