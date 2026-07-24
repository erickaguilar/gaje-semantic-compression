import os
import sys
import numpy as np
import gguf

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

from gaje.utils.quantization import unpermute_to_interleaved


def check_rope_permutation(gguf_path):
    print("🔍 [VERIFICACIÓN] ¿Necesita SmolLM2 F16 unpermute?")
    reader = gguf.GGUFReader(gguf_path)

    # SmolLM2-135M parameters
    n_head = 9
    n_embd = 576
    head_dim = n_embd // n_head

    layer_name = "blk.0.attn_q.weight"
    tensor = next(t for t in reader.tensors if t.name == layer_name)
    w_original = tensor.data.astype(np.float32)

    # El test de "unpermute" no se puede hacer contra sí mismo para ver si "lo necesita"
    # pero podemos ver si la estructura de los pesos sugiere permutación (pares alternos)
    # Sin embargo, lo más directo es comparar la inferencia con y sin.

    w_unpermuted = unpermute_to_interleaved(w_original.copy(), n_head, head_dim)

    diff = np.mean(np.abs(w_original - w_unpermuted))
    print(f"Diferencia media tras unpermute: {diff:.8f}")
    if diff < 1e-6:
        print("✅ Los pesos YA están en formato interleaved (o no están permutados).")
    else:
        print("⚠️ El unpermute CAMBIA los pesos significativamente.")


if __name__ == "__main__":
    check_rope_permutation("models/gguf/smollm2-135m-f16.gguf")
