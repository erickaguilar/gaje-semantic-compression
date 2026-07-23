import numpy as np


def dequantize_q8_0(
    tensor, n_head=None, head_dim=None, is_q_or_k=False, rope_style="split"
):
    """
    De-cuantiza bloques Q8_0 (Block size 32) de GGUF.
    tensor: El objeto tensor de gguf
    """
    data_u8 = tensor.data
    # Cada bloque de 32 elementos ocupa 34 bytes (2 para delta f16 + 32 int8)
    # n_elements = (len(data_u8) / 34) * 32

    # GGUF tensors usually have shape [out, in] or [length]
    # We need the original dimensions to reshape
    if len(tensor.shape) == 2:
        in_f, out_f = tensor.shape
    else:
        out_f = tensor.shape[0]
        in_f = 1

        
    # Usamos la implementación nativa si está disponible por velocidad
    try:
        from gaje.core import _impl as dna_semantic_compression

        # La implementación nativa espera out_f e in_f
        # Nota: GGUF shape[0] es la dimensión más rápida (in_f en pesos lineales)
        # Pero depende de cómo lo cargue la librería gguf.
        # Por seguridad, usamos la lógica de bloques manual si la nativa falla.
        res = dna_semantic_compression.dequantize_q8_0_native(
            data_u8.tobytes(), out_f, in_f
        )
        return np.array(res).reshape(out_f, in_f)
    except Exception:
        # Fallback manual simplificado
        n_blocks = len(data_u8) // 34
        weights_f32 = np.zeros(n_blocks * 32, dtype=np.float32)

        for b in range(n_blocks):
            offset = b * 34
            delta = np.frombuffer(data_u8[offset : offset + 2], dtype=np.float16)[
                0
            ].astype(np.float32)
            qs = data_u8[offset + 2 : offset + 34].view(np.int8).astype(np.float32)
            weights_f32[b * 32 : (b + 1) * 32] = qs * delta

        return weights_f32.reshape(out_f, in_f)

def unpermute_to_split(weights, n_head, head_dim):
    """Deshace la permutación de RoPE usada en GGUF para Llama/Qwen, pasando de interleaved a split."""
    # weights: [out_features, in_features]
    # GGUF almacena los pesos como (n_head, head_dim // 2, 2)
    # Nosotros queremos (n_head, 2, head_dim // 2)
    out_f, in_f = weights.shape
    return weights.reshape(n_head, head_dim // 2, 2, in_f).transpose(0, 2, 1, 3).reshape(out_f, in_f)
