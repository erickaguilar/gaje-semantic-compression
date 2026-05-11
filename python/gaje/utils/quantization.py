import numpy as np
from gaje.core import _impl as dna_semantic_compression

def dequantize_q8_0(tensor, n_head=None, head_dim=None, is_q_or_k=False):
    """
    De-cuantiza un tensor Q8_0 de GGUF utilizando el kernel optimizado en Rust.
    Incluye des-permutación opcional para arquitecturas tipo Llama/Qwen.
    """
    in_features, out_features = tensor.shape
    data = tensor.data.tobytes()
    
    # Kernel Nativo (Rust)
    flat_weights = dna_semantic_compression.dequantize_q8_0_native(data, out_features, in_features)
    w = np.array(flat_weights, dtype=np.float32).reshape(out_features, in_features)
    
    # Des-permutación para Llama 3 / Qwen2 (GGUF)
    # Llama.cpp permuta Q y K para que el RoPE split sea contiguo en memoria.
    if is_q_or_k and n_head is not None and head_dim is not None:
        w_new = np.zeros_like(w)
        for h in range(n_head):
            for i in range(head_dim // 2):
                w_new[h * head_dim + i] = w[h * head_dim + 2 * i]
                w_new[h * head_dim + head_dim // 2 + i] = w[h * head_dim + 2 * i + 1]
        return w_new
    return w
