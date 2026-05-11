import os
import numpy as np
from gaje.core import _impl as dna_semantic_compression
import gguf
import time
from transformers import AutoTokenizer


class EntropyValidator:
    def dequantize_q8_0(self, data_u8, hidden_dim):
        """
        De-cuantiza bloques Q8_0 (Block size 32).
        Cada bloque: 1 float16 (delta) + 32 int8.
        """
        print(f"[*] De-cuantizando pesos Q8_0 (Hidden: {hidden_dim})...")
        # En Q8_0, cada bloque de 32 elementos ocupa 34 bytes (2 para delta + 32 para pesos)
        # data_u8 tiene forma [N, 34 * (hidden_dim/32)]
        n_rows = data_u8.shape[0]
        n_blocks = hidden_dim // 32

        # Re-estructurar para procesar bloques
        # Q8_0 GGUF tensor.data shape suele venir ya estructurado
        # pero necesitamos extraer los deltas (float16)
        weights_f32 = np.zeros((n_rows, hidden_dim), dtype=np.float32)

        for i in range(n_rows):
            row_data = data_u8[i].view(np.uint8)
            # Cada bloque es de 34 bytes
            for b in range(n_blocks):
                offset = b * 34
                # Primeros 2 bytes son delta en float16
                delta_bytes = row_data[offset : offset + 2]
                delta = np.frombuffer(delta_bytes, dtype=np.float16)[0].astype(
                    np.float32
                )
                # Siguientes 32 bytes son los pesos int8
                qs = row_data[offset + 2 : offset + 34].view(np.int8).astype(np.float32)
                weights_f32[i, b * 32 : (b + 1) * 32] = qs * delta

        return weights_f32

    def __init__(self, model_path):
        print(f"🧬 Inicializando Validador de Entropía para: {model_path}")
        self.reader = gguf.GGUFReader(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        self.hidden_dim = 896

        # 1. Cargar y De-cuantizar Pesos (Referencia)
        embd_tensor = next(
            t for t in self.reader.tensors if t.name == "token_embd.weight"
        )
        # En Q8_0, tensor.data es uint8 con bloques de 34 bytes
        self.W_orig = self.dequantize_q8_0(embd_tensor.data, self.hidden_dim)
        print(f"    [+] Pesos de Referencia (F32): {self.W_orig.shape}")

        # 2. Crear Capa Genomizada (GAJE 2-bit)
        print("[*] Genomizando capa de referencia para validación...")
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
        pk = pk[pk > 0]  # Evitar log(0)
        return -np.sum(pk * np.log(pk))

    def calculate_kl_div(self, pk, qk):
        """KL Divergence: sum(p * log(p/q))"""
        pk = np.array(pk)
        qk = np.array(qk)
        # Asegurar que no haya ceros en qk
        qk = np.where(qk == 0, 1e-12, qk)
        # Solo calcular donde pk > 0
        mask = pk > 0
        return np.sum(pk[mask] * np.log(pk[mask] / qk[mask]))

    def analyze_fidelity(self, prompt="El protocolo GAJE"):
        print(f"\n📝 Analizando fidelidad para: '{prompt}'")

        # Simular una activación de entrada realista
        x_input = np.random.normal(0, 1.0, (896,)).astype(np.float32)

        # B. Forward Pass Original (F32)
        start_orig = time.perf_counter()
        logits_orig = np.dot(self.W_orig, x_input)
        probs_orig = self.softmax(logits_orig)
        time_orig = (time.perf_counter() - start_orig) * 1000

        # C. Forward Pass Genómico (GAJE 2-bit)
        start_gen = time.perf_counter()
        logits_gen = np.array(self.engine.genomic_linear_forward(x_input.tolist()))
        probs_gen = self.softmax(logits_gen)
        time_gen = (time.perf_counter() - start_gen) * 1000

        # D. Métricas de Fidelidad
        cos_sim = np.dot(logits_orig, logits_gen) / (
            np.linalg.norm(logits_orig) * np.linalg.norm(logits_gen)
        )

        entropy_orig = self.calculate_entropy(probs_orig)
        entropy_gen = self.calculate_entropy(probs_gen)
        kl_div = self.calculate_kl_div(probs_orig, probs_gen)

        print("-" * 45)
        print("📊 REPORTE DE FIDELIDAD DE LOGITS")
        print("-" * 45)
        print(f"✅ Similitud Coseno:      {cos_sim:.6f}")
        print(f"✅ Divergencia KL:       {kl_div:.6f} (Menor es mejor)")
        print(f"✅ Entropía Orig:        {entropy_orig:.4f} bits")
        print(f"✅ Entropía GAJE:        {entropy_gen:.4f} bits")
        print(f"✅ Delta Entropía:       {abs(entropy_orig - entropy_gen):.4f} bits")

        print("\n⚡ RENDIMIENTO")
        print(f"⏱️ Latencia F32:         {time_orig:.2f} ms")
        print(f"⏱️ Latencia GAJE:        {time_gen:.2f} ms")
        print("-" * 45)

        if kl_div < 0.1:
            print("🔥 ESTADO: EXCELENTE. La señal semántica es casi idéntica.")
        elif kl_div < 0.5:
            print("⚠️ ESTADO: ACEPTABLE. Hay pérdida menor de matices.")
        else:
            print("❌ ESTADO: CRÍTICO. Colapso de información detectado.")


if __name__ == "__main__":
    model_path = "./data/models/qwen2-0_5b-instruct-fp16.gguf"
    if os.path.exists(model_path):
        validator = EntropyValidator(model_path)
        validator.analyze_fidelity()
    else:
        print("❌ Modelo no encontrado para validación.")
