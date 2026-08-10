import numpy as np
from gaje.nn.stabilized import GenomicLayer
from gaje.core import _impl as dna_core

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def run_refinement(lr, iterations=50):
    np.random.seed(42)
    dim = 64
    out_dim = 128

    w_gate_f32 = np.random.randn(out_dim, dim).astype(np.float32) * 0.1
    w_up_f32 = np.random.randn(out_dim, dim).astype(np.float32) * 0.1

    layer_gate = GenomicLayer("gate", w_gate_f32, anchor_threshold=0.75)
    layer_up = GenomicLayer("up", w_up_f32, anchor_threshold=0.75)

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

    x = np.random.randn(dim).astype(np.float32)
    swiglu_target = silu(np.dot(w_gate_f32, x)) * np.dot(w_up_f32, x)

    gate_init = layer_gate.forward(x)
    up_init = layer_up.forward(x)
    swiglu_init = silu(gate_init) * up_init
    initial_error = np.mean((swiglu_init - swiglu_target) ** 2)

    for i in range(iterations):
        block.refine_ffn(x.tolist(), swiglu_target.tolist(), lr)

    gate_final = np.array(block.gate_gen.forward(x.tolist(), True))
    up_final = np.array(block.up_gen.forward(x.tolist(), True))
    swiglu_final = silu(gate_final) * up_final
    final_error = np.mean((swiglu_final - swiglu_target) ** 2)

    improvement = (initial_error - final_error) / initial_error * 100
    print(f"LR: {lr:.5f} | Initial MSE: {initial_error:.8f} | Final MSE: {final_error:.8f} | Improvement: {improvement:.2f}%")
    return improvement

if __name__ == "__main__":
    for lr in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, -0.001, -0.005, -0.01]:
        run_refinement(lr)
