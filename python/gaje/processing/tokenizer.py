import os
import numpy as np
from gaje.core import _impl as dna_semantic_compression
from transformers import AutoTokenizer


class GenomicTokenizer:
    """
    Tokenizer que integra el motor GAJE para producir embeddings genómicos de 2 bits.
    """

    def __init__(self, model_name_or_path, embedding_weights_f32):
        print(f"🧬 Inicializando Tokenizer Genómico para: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.vocab_size = len(self.tokenizer)

        # Genomizar los embeddings del vocabulario
        print(f"[*] Genomizando matriz de embeddings ({self.vocab_size} tokens)...")
        std = np.std(embedding_weights_f32)
        mean = np.mean(embedding_weights_f32)

        # Max-Lloyd 2-bit
        self.thresholds = [mean - 0.9816 * std, mean, mean + 0.9816 * std]
        self.centroids = [
            mean - 1.510 * std,
            mean - 0.4528 * std,
            mean + 0.4528 * std,
            mean + 1.510 * std,
        ]

        # Cargar en Rust
        dna_embeddings = [
            dna_semantic_compression.quantize_embedding(w.tolist(), self.thresholds)
            for w in embedding_weights_f32
        ]
        self.engine = dna_semantic_compression.GajeIndex([], self.centroids)
        self.engine.add_batch(dna_embeddings)

    def encode(self, text):
        """
        Convierte texto en una secuencia de 'Codones Semánticos' (Embeddings de 2 bits).
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # Recuperar los embeddings genomizados de Rust
        genomic_embeddings = []
        for tid in token_ids:
            # Re-extraemos el ADN del índice para simular el flujo del modelo
            # (En un modelo real esto sería la entrada a la primera capa)
            dna_strand = self.engine.database[
                tid * self.engine.stride : (tid + 1) * self.engine.stride
            ]
            genomic_embeddings.append(dna_strand)

        return token_ids, genomic_embeddings

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)


def test_genomic_tokenizer():
    import gguf

    model_path = "/data/data/com.termux/files/home/models/gguf/qwen2-0_5b-q8_0.gguf"

    if not os.path.exists(model_path):
        print(
            "❌ Error: No se encontró el modelo Qwen2 para extraer los pesos de embedding."
        )
        return

    # 1. Extraer pesos de embedding reales
    print("[*] Cargando pesos de embedding desde GGUF...")
    reader = gguf.GGUFReader(model_path)
    embd_tensor = next(t for t in reader.tensors if t.name == "token_embd.weight")
    weights_f32 = embd_tensor.data.astype(np.float32)

    # 2. Inicializar Tokenizer Genómico
    # Nota: Usamos el nombre del modelo en HF para cargar la lógica BPE
    g_tokenizer = GenomicTokenizer("Qwen/Qwen2-0.5B", weights_f32)

    # 3. Prueba de Fuego: Texto -> ADN
    frase = "El protocolo GAJE es el futuro de la IA móvil."
    print(f"\n📝 Frase original: '{frase}'")

    ids, dna_seq = g_tokenizer.encode(frase)

    print("\n✅ Tokenización Genómica Exitosa:")
    print(f"   • Tokens IDs: {ids}")
    print(
        f"   • ADN Semántico: {len(dna_seq)} strands de {len(dna_seq[0])} bytes cada uno."
    )
    print(f"   • RAM Total ADN: {(len(dna_seq) * len(dna_seq[0]))} bytes")
    print(
        f"   • RAM F32 Equiv: {(len(dna_seq) * weights_f32.shape[1] * 4)} bytes (16x ahorro)"
    )

    # 4. Simulación de Inferencia: Recuperar significado
    print("\n[*] Reconstruyendo señal para validación...")
    first_token_dna = dna_seq[0]
    # De-cuantizar usando los centroides entrenados
    rec_signal = dna_semantic_compression.dequantize_embedding(
        list(first_token_dna), weights_f32.shape[1], g_tokenizer.centroids
    )

    # Comparar con el peso original
    orig_signal = weights_f32[ids[0]]
    cos_sim = np.dot(orig_signal, rec_signal) / (
        np.linalg.norm(orig_signal) * np.linalg.norm(rec_signal)
    )

    print(
        f"   • Similitud Coseno (Token '{g_tokenizer.decode([ids[0]])}'): {cos_sim:.4f}"
    )


if __name__ == "__main__":
    test_genomic_tokenizer()
