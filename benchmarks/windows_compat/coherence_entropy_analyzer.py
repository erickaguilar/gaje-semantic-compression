"""
Benchmark de Analisis de Entropia adaptado para Windows.
Usa el modelo Qwen2-0.5B-Instruct GGUF local.
"""
import os
import sys
import numpy as np
import time

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, os.path.join(project_root, "python"))

from gaje.core import _impl as dna_semantic_compression
import gguf
from transformers import AutoTokenizer


class EntropyValidator:
    def dequantize_q8_0(self, data_u8, hidden_dim):
        """De-cuantiza bloques Q8_0 (Block size 32)."""
        print(f"[*] De-cuantizando pesos Q8_0 (Hidden: {hidden_dim})...")
        n_rows = data_u8.shape[0]
        n_blocks = hidden_dim // 32
        weights_f32 = np.zeros((n_rows, hidden_dim), dtype=np.float32)
        for i in range(n_rows):
            row_data = data_u8[i].view(np.uint8)
            for b in range(n_blocks):
                offset = b * 34
                delta_bytes = row_data[offset : offset + 2]
                delta = np.frombuffer(delta_bytes, dtype=np.float16)[0].astype(
                    np.float32
                )
                qs = row_data[offset + 2 : offset + 34].view(np.int8).astype(np.float32)
                weights_f32[i, b * 32 : (b + 1) * 32] = qs * delta
        return weights_f32

    def __init__(self, model_path):
        print(
            f"[DNA] Inicializando Validador de Entropia para: {os.path.basename(model_path)}"
        )
        self.reader = gguf.GGUFReader(model_path)

        # Detectar arquitectura del modelo
        arch = self.reader.fields["general.architecture"].parts[-1]
        if hasattr(arch, "tolist"):
            arch = arch.tolist()
        arch = (
            (
                "".join([chr(x) for x in arch])
                if isinstance(arch, list) and isinstance(arch[0], int)
                else str(arch[0] if isinstance(arch, list) else arch)
            )
            .strip()
            .replace("\x00", "")
        )

        self.hidden_dim = int(
            self.reader.fields[f"{arch}.embedding_length"].parts[-1][0]
        )
        tokenizer_name = (
            "Qwen/Qwen2-0.5B-Instruct"
            if arch == "qwen2"
            else "HuggingFaceTB/SmolLM2-135M-Instruct"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        print(f"    Arquitectura: {arch}, Hidden dim: {self.hidden_dim}")

        # 1. Cargar y De-cuantizar Pesos (Referencia)
        embd_tensor = next(
            t for t in self.reader.tensors if t.name == "token_embd.weight"
        )

        if embd_tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0:
            self.W_orig = self.dequantize_q8_0(embd_tensor.data, self.hidden_dim)
        elif embd_tensor.tensor_type == gguf.GGMLQuantizationType.F16:
            self.W_orig = (
                np.frombuffer(embd_tensor.data, dtype=np.float16)
                .reshape(-1, self.hidden_dim)
                .astype(np.float32)
            )
        else:
            self.W_orig = np.frombuffer(embd_tensor.data, dtype=np.float32).reshape(
                -1, self.hidden_dim
            )

        print(f"    [+] Pesos de Referencia (F32): {self.W_orig.shape}")

        # 2. Crear Capa Genomizada (GAJE 2-bit)
        print("[*] Genomizando capa de referencia para validacion...")
        std = np.std(self.W_orig)
        mean = np.mean(self.W_orig)
        self.thresholds = [mean - 0.9816 * std, mean, mean + 0.9816 * std]
        self.centroids = [
            mean - 1.510 * std,
            mean - 0.4528 * std,
            mean + 0.4528 * std,
            mean + 1.510 * std,
        ]

        dna_batch = [
            dna_semantic_compression.quantize_embedding(w.tolist(), self.thresholds)
            for w in self.W_orig
        ]
        self.engine = dna_semantic_compression.GajeIndex([], self.centroids)
        self.engine.add_batch(dna_batch)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def calculate_entropy(self, pk):
        """Shannon Entropy: -sum(p * log(p))"""
        pk = np.array(pk)
        pk = pk[pk > 0]
        return -np.sum(pk * np.log(pk))

    def calculate_kl_div(self, pk, qk):
        """KL Divergence: sum(p * log(p/q))"""
        pk = np.array(pk)
        qk = np.array(qk)
        qk = np.where(qk == 0, 1e-12, qk)
        mask = pk > 0
        return np.sum(pk[mask] * np.log(pk[mask] / qk[mask]))

    def analyze_fidelity(self, prompt="El protocolo GAJE"):
        print(f"\n[TEXT] Analizando fidelidad para: '{prompt}'")

        # Simular una activacion de entrada realista
        x_input = np.random.normal(0, 1.0, (self.hidden_dim,)).astype(np.float32)

        # B. Forward Pass Original (F32)
        start_orig = time.perf_counter()
        logits_orig = np.dot(self.W_orig, x_input)
        probs_orig = self.softmax(logits_orig)
        time_orig = (time.perf_counter() - start_orig) * 1000

        # C. Forward Pass Genomico (GAJE 2-bit) via flat_search como proxy
        start_gen = time.perf_counter()
        # Usamos la busqueda ADC como proxy del forward genomico
        results = dna_semantic_compression.dna_similarity_search_adc(
            x_input.tolist(),
            [
                list(
                    self.engine.database[
                        i * self.engine.stride : (i + 1) * self.engine.stride
                    ]
                )
                for i in range(self.W_orig.shape[0])
            ],
            self.centroids,
        )
        # Construir logits desde distancias (invertidas)
        logits_gen = np.zeros(self.W_orig.shape[0])
        for idx, dist in results:
            logits_gen[idx] = -dist  # Distancia inversamente proporcional
        probs_gen = self.softmax(logits_gen)
        time_gen = (time.perf_counter() - start_gen) * 1000

        # D. Metricas de Fidelidad
        cos_sim = np.dot(logits_orig, logits_gen) / (
            np.linalg.norm(logits_orig) * np.linalg.norm(logits_gen) + 1e-10
        )

        entropy_orig = self.calculate_entropy(probs_orig)
        entropy_gen = self.calculate_entropy(probs_gen)
        kl_div = self.calculate_kl_div(probs_orig, probs_gen)

        print("-" * 50)
        print("  REPORTE DE FIDELIDAD DE LOGITS")
        print("-" * 50)
        print(f"  Similitud Coseno:      {cos_sim:.6f}")
        print(f"  Divergencia KL:        {kl_div:.6f} (Menor es mejor)")
        print(f"  Entropia Orig:         {entropy_orig:.4f} bits")
        print(f"  Entropia GAJE:         {entropy_gen:.4f} bits")
        print(f"  Delta Entropia:        {abs(entropy_orig - entropy_gen):.4f} bits")

        print("\n  RENDIMIENTO")
        print(f"  Latencia F32:          {time_orig:.2f} ms")
        print(f"  Latencia GAJE:         {time_gen:.2f} ms")
        print("-" * 50)

        if kl_div < 0.1:
            print("  ESTADO: EXCELENTE. La senal semantica es casi identica.")
        elif kl_div < 0.5:
            print("  ESTADO: ACEPTABLE. Hay perdida menor de matices.")
        else:
            print("  ESTADO: NOTABLE. Diferencia significativa (esperada con 2-bit).")


if __name__ == "__main__":
    model_path = os.path.join(project_root, "models", "Qwen2-0.5B-Instruct-Q8_0.gguf")
    if os.path.exists(model_path):
        validator = EntropyValidator(model_path)
        validator.analyze_fidelity()
    else:
        print("[ERROR] Modelo Qwen2 no encontrado.")
